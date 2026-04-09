import os
import glob
import subprocess
import tempfile
import threading
import uuid
import time
import re
from functools import wraps
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, send_from_directory)
import anthropic

app = Flask(__name__)

# SECRET_KEY が未設定の場合は起動時に警告
_secret = os.environ.get('SECRET_KEY', '')
if not _secret:
    import secrets
    _secret = secrets.token_hex(32)
    print('WARNING: SECRET_KEY が未設定です。ランダム値を使用します（再起動でセッションが無効になります）。')
app.secret_key = _secret

# セッションをブラウザ閉じたら失効させる（永続化しない）
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE']   = True  # HTTPS のみ

APP_PASSWORD  = os.environ.get('APP_PASSWORD', '')
ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

# ── ブルートフォース対策（IPごとの失敗カウント） ──────────────────
_login_failures: dict[str, dict] = {}
_failures_lock  = threading.Lock()
MAX_FAILURES    = 10   # 失敗上限
LOCKOUT_SEC     = 600  # ロック時間（10分）

def _get_ip() -> str:
    return request.headers.get('X-Forwarded-For', request.remote_addr or '').split(',')[0].strip()

def _is_locked(ip: str) -> bool:
    with _failures_lock:
        rec = _login_failures.get(ip)
        if not rec:
            return False
        if rec['count'] >= MAX_FAILURES:
            if time.time() - rec['last'] < LOCKOUT_SEC:
                return True
            # ロック期間が過ぎたらリセット
            del _login_failures[ip]
    return False

def _record_failure(ip: str):
    with _failures_lock:
        rec = _login_failures.setdefault(ip, {'count': 0, 'last': 0})
        rec['count'] += 1
        rec['last']   = time.time()

def _clear_failure(ip: str):
    with _failures_lock:
        _login_failures.pop(ip, None)

# ── stand.fm URL 検証 ─────────────────────────────────────────────
ALLOWED_URL_PATTERN = re.compile(r'^https://stand\.fm/')

def is_valid_url(url: str) -> bool:
    return bool(ALLOWED_URL_PATTERN.match(url))

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
        ip = _get_ip()
        if _is_locked(ip):
            error = 'ログイン試行が多すぎます。しばらくお待ちください。'
        elif request.form.get('password') == APP_PASSWORD and APP_PASSWORD:
            _clear_failure(ip)
            session['authenticated'] = True
            return redirect(url_for('portal'))
        else:
            _record_failure(ip)
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
    date_str   = data.get('date', '').strip()
    staff_code = data.get('staff_code', '担当職員').strip()

    if not url:
        return jsonify({'error': 'URLを入力してください'}), 400
    if not is_valid_url(url):
        return jsonify({'error': 'stand.fm の URL のみ受け付けています'}), 400
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
        args=(job_id, url, date_str, staff_code),
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

def run_kansan_job(job_id, url, date_str, staff_code):
    import json as json_mod
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

            # Step 3: 利用者ごとに職員考察を生成
            update_job(job_id, step='職員考察を生成中...')
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            prompt = f"""あなたは就労継続支援B型事業所「ジャパニケア札幌」の支援員です。
以下の音声文字起こしには、複数の利用者の記録が含まれています。

支援日：{date_str}
担当：{staff_code}

音声文字起こし：
{transcript[:6000]}

【指示】
「本日の〇〇さん」「〇〇さんについて」などの区切りを手がかりに、利用者ごとのセクションを識別してください。
各利用者に対して「職員考察」を執筆し、以下のJSON形式のみで出力してください。
説明・前置き・コードブロック記号（```）は一切不要です。JSONだけを出力してください。

[
  {{"user": "Aさん", "kansan": "職員考察本文"}},
  {{"user": "Bさん", "kansan": "職員考察本文"}}
]

【各考察の執筆ルール】
・120〜180字程度（端的にまとめる）
・利用者・職員の発話は「」でくくる
・当日の作業内容を必ず盛り込む
・利用者の体調・メンタル・様子を適切に記録する
・5W1Hを意識し、ほかのスタッフが読んでも現場の光景が浮かぶ内容にする
・個人名は使わず「〇〇さん」として表記する

【文体ルール（おがっち構文）】
・一文は30〜40字以内で短く切る
・指示語（この・その・あの・それ）は使わない
・「しかし」「だが」は使わない。短文の連打で表現する
・結論から書き始める。体言止めを適宜活用する

【表記ルール】
・野菜名はカタカナ（タマネギ・レタス・キャベツ等）
・その他は「例解辞典」の表記ルールに準拠"""

            msg = client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=2000,
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw = msg.content[0].text.strip()

            # JSON をパース（```json ... ``` で囲まれている場合も対応）
            raw_clean = raw
            if '```' in raw_clean:
                raw_clean = raw_clean.split('```')[-2] if raw_clean.count('```') >= 2 else raw_clean
                raw_clean = raw_clean.replace('json', '', 1).strip()
            users_data = json_mod.loads(raw_clean)

            # 文字数を付加
            results = [
                {'user': item['user'],
                 'kansan': item['kansan'],
                 'char_count': len(item['kansan'])}
                for item in users_data
            ]

            update_job(job_id, status='done', step='完了',
                       result={'users': results, 'transcript': transcript})

    except subprocess.TimeoutExpired:
        update_job(job_id, status='error',
                   error='処理がタイムアウトしました。URLを確認して再試行してください。')
    except RuntimeError as e:
        # ユーザー向けメッセージ（yt-dlp失敗など）はそのまま表示
        update_job(job_id, status='error', error=str(e))
    except Exception:
        # 予期せぬエラーは内部情報を隠す
        update_job(job_id, status='error',
                   error='処理中に予期せぬエラーが発生しました。管理者にお問い合わせください。')


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
