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
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-in-production')

APP_PASSWORD = os.environ.get('APP_PASSWORD', '0111')
ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY', '0111')

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
        if request.form.get('password') == APP_PASSWORD and APP_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('portal'))
        error = 'パスワードが正しくありません'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ── ポータル（トップページ） ──────────────────────────────────────
@app.route('/')
@login_required
def portal():
    return render_template('portal.html')


# ── 既存HTMLツールを配信 ─────────────────────────────────────────
# tools/ フォルダに置いた既存チェッカーHTMLをそのまま配信する
@app.route('/tools/<path:filename>')
@login_required
def serve_tool(filename):
    return send_from_directory('tools', filename)


# ── 職員考察ツール ────────────────────────────────────────────────
@app.route('/kansan')
@login_required
def kansan():
    return render_template('kansan.html')

@app.route('/kansan/start', methods=['POST'])
@login_required
def kansan_start():
    data       = request.json or {}
    url        = data.get('url', '').strip()
    user_code  = data.get('user_code', '利用者A').strip()
    date_str   = data.get('date', '').strip()
    staff_code = data.get('staff_code', '担当職員').strip()

    if not url:
        return jsonify({'error': 'URLを入力してください'}), 400
    if not ANTHROPIC_KEY:
        return jsonify({'error': 'ANTHROPIC_API_KEY が未設定です（管理者に連絡）'}), 500

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
        jobs[job_id].update(kwargs)

def run_kansan_job(job_id, url, user_code, date_str, staff_code):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:

            # Step 1: 音声ダウンロード
            update_job(job_id, status='running', step='音声をダウンロード中...')
            audio_base = os.path.join(tmpdir, 'audio')
            dl = subprocess.run(
                ['yt-dlp', '-x', '--audio-format', 'mp3', '--audio-quality', '5',
                 '-o', audio_base + '.%(ext)s', '--no-playlist', '--quiet', url],
                capture_output=True, text=True, timeout=300
            )
            audio_files = glob.glob(audio_base + '.*')
            if not audio_files or dl.returncode != 0:
                raise RuntimeError(
                    f'音声のダウンロードに失敗しました。URLを確認してください。\n{dl.stderr[-400:]}'
                )

            # Step 2: 文字起こし
            update_job(job_id, step='文字起こし中（2〜5分かかります）...')
            model = get_whisper()
            segments, _ = model.transcribe(
                audio_files[0], language='ja', beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            fillers = ['えー', 'あー', 'あのー', 'えーと', 'えっと', 'うーん', 'そのー']
            parts = []
            for seg in segments:
                text = seg.text.strip()
                for f in fillers:
                    text = text.replace(f, '')
                if text:
                    parts.append(text)
            transcript = ''.join(parts)

            # Step 3: 職員考察生成
            update_job(job_id, step='職員考察を生成中...')
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            prompt = f"""あなたは就労継続支援B型事業所「ジャパニケア札幌」の支援員です。
以下の音声文字起こしをもとに、支援記録の「職員考察」を執筆してください。

支援日：{date_str}
対象：{user_code}
担当：{staff_code}

音声文字起こし：
{transcript[:4000]}

【執筆ルール】
・120〜180字程度（端的にまとめる）
・利用者・職員の発話は「」でくくる
・当日の作業内容を必ず盛り込む
・利用者の体調・メンタル・様子を適切に記録する
・5W1Hを意識し、ほかのスタッフが読んでも現場の光景が浮かぶ内容にする
・個人名は使わず、{user_code}として表記する

【文体ルール（おがっち構文）】
・一文は30〜40字以内で短く切る
・指示語（この・その・あの・それ）は使わない。対象を直接書く
・「しかし」「だが」「けれども」は使わない。短文の連打で表現する
・作業内容など結論から書き始める
・体言止めを適宜活用する

【表記ルール】
・野菜名はカタカナ（タマネギ・レタス・キャベツ・ジャガイモ等）
・その他の漢字ひらがなの使い分けは「例解辞典」の表記ルールに準拠

職員考察の本文のみを出力してください。説明・前置き・見出しは不要です。"""

            msg = client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=500,
                messages=[{'role': 'user', 'content': prompt}]
            )
            kansan_text = msg.content[0].text.strip()

            update_job(job_id, status='done', step='完了',
                       result={'kansan': kansan_text,
                               'char_count': len(kansan_text),
                               'transcript': transcript})

    except subprocess.TimeoutExpired:
        update_job(job_id, status='error',
                   error='処理がタイムアウトしました。URLを確認して再試行してください。')
    except Exception as e:
        update_job(job_id, status='error', error=str(e))


# ── 古いジョブをクリーンアップ ───────────────────────────────────
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
