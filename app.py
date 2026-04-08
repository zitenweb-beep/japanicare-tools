import os
import glob
import subprocess
import tempfile
import threading
import uuid
import time
from functools import wraps
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, send_from_directory)
import anthropic

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'japanicare-secret-key')

# 認証パスワード（環境変数になければ '0111'）
APP_PASSWORD = os.environ.get('APP_PASSWORD', '0111')
# APIキーはここを直接書き換えてください
ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY', 'ここにsk-から始まるキーを入れる')

# ── ジョブストア ──────────────────────────────────────────────────
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# ── Whisper（初回リクエスト時にロード） ──────────────────────────
whisper_model = None
whisper_lock  = threading.Lock()

def get_whisper():
    global whisper_model
    with whisper_lock:
        if whisper_model is None:
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return whisper_model

# ── 認証 ─────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == APP_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('portal'))
        error = 'パスワードが正しくありません'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def portal():
    return render_template('portal.html')

@app.route('/tools/<path:filename>')
@login_required
def serve_tool(filename):
    return send_from_directory('tools', filename)

@app.route('/kansan')
@login_required
def kansan():
    return render_template('kansan.html')

@app.route('/kansan/start', methods=['POST'])
@login_required
def kansan_start():
    data = request.json or {}
    url = data.get('url', '').strip()
    user_code = data.get('user_code', '利用者A').strip()
    date_str = data.get('date', '').strip()
    staff_code = data.get('staff_code', '担当職員').strip()

    if not url:
        return jsonify({'error': 'URLを入力してください'}), 400

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            'status': 'queued', 'step': '',
            'result': None, 'error': None,
            'created_at': time.time()
        }

    threading.Thread(
        target=run_kansan_job,
        args=(job_id, url, user_code, date_str, staff_code),
        daemon=True
    ).start()
    return jsonify({'job_id': job_id})

@app.route('/kansan/status/<job_id>')
@login_required
def kansan_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'ジョブが見つかりません'}), 404
    return jsonify(job)

# ── バックグラウンド処理 ──────────────────────────────────────────
def update_job(job_id, **kwargs):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)

def run_kansan_job(job_id, url, user_code, date_str, staff_code):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            update_job(job_id, status='running', step='音声をダウンロード中...')
            audio_base = os.path.join(tmpdir, 'audio')
            
            # 【重要】ffmpegの場所指定を削除し、システムに任せる設定
            dl = subprocess.run(
                ['yt-dlp', '-x', '--audio-format', 'mp3', '--audio-quality', '5',
                 '-o', audio_base + '.%(ext)s', '--no-playlist', '--quiet', url],
                capture_output=True, text=True, timeout=300
            )
            
            audio_files = glob.glob(audio_base + '.*')
            if not audio_files or dl.returncode != 0:
                error_detail = dl.stderr if dl.stderr else "URLが正しくないか、音声が見つかりません。"
                raise RuntimeError(f'ダウンロード失敗: {error_detail}')

            update_job(job_id, step='文字起こし中（2〜5分かかります）...')
            model = get_whisper()
            segments, _ = model.transcribe(
                audio_files[0], language='ja', beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            parts = [seg.text.strip() for seg in segments if seg.text.strip()]
            transcript = ''.join(parts)

            update_job(job_id, step='職員考察を生成中...')
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            
            prompt = f"""支援記録の「職員考察」を執筆してください。
支援日：{date_str}
対象：{user_code}
担当：{staff_code}
内容：{transcript[:4000]}

【ルール】
・150字程度。
・一文を短く。「この」「その」は使わない。
・野菜名はカタカナ。
・結論から書く。体言止め活用。"""

            msg = client.messages.create(
                model='claude-3-sonnet-20240229',
                max_tokens=500,
                messages=[{'role': 'user', 'content': prompt}]
            )
            kansan_text = msg.content[0].text.strip()

            update_job(job_id, status='done', step='完了',
                       result={'kansan': kansan_text,
                               'char_count': len(kansan_text),
                               'transcript': transcript})

    except Exception as e:
        update_job(job_id, status='error', error=str(e))

def cleanup_old_jobs():
    while True:
        time.sleep(3600)
        with jobs_lock:
            now = time.time()
            for k in [k for k, v in jobs.items()
                      if v.get('created_at', now) < now - 3600]:
                del jobs[k]

threading.Thread(target=cleanup_old_jobs, daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
