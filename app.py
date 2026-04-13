import os
import glob
import subprocess
import tempfile
import threading
import uuid
import time
import re
import traceback
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

# ── 個別支援計画書 AI生成 ─────────────────────────────────────────
@app.route('/shienkeikaku/generate', methods=['POST'])
@login_required
def shienkeikaku_generate():
    import json as json_mod
    data = request.json or {}
    transcript = (data.get('transcript') or '').strip()

    if not transcript:
        return jsonify({'error': '文字起こしデータが空です'}), 400
    if not ANTHROPIC_KEY:
        return jsonify({'error': 'ANTHROPIC_API_KEY が未設定です（管理者に連絡）'}), 500

    # 長すぎる入力はカット（概ね Claude の入力上限と処理時間を考慮）
    if len(transcript) > 20000:
        transcript = transcript[:20000]

    prompt = f"""あなたは就労継続支援B型事業所「ジャパニケア札幌」のサービス管理責任者です。
以下の文字起こしは、利用者との個別支援計画書作成ヒアリング音声を書き起こしたものです。

【指示】
文字起こしデータから、個別支援計画書の各項目を抽出・整理し、下記JSON形式のみで出力してください。
説明・前置き・コードブロック記号（```）は一切不要です。JSONだけを出力してください。

【JSONスキーマ】
{{
  "intention": "本人・家族の意向（本文）",
  "policy": "総合的な支援の方針（本文）",
  "situation": "本人の現状（本文）",
  "longGoal": "長期目標（1〜5年スパンの目標）",
  "goals": [
    {{
      "issue": "課題",
      "goal": "目標",
      "self": "本人の取組内容",
      "staff": "職員の支援内容",
      "period": "6ヵ月",
      "note": "「体調チェックシート」の推移を観察する。",
      "etc": "その他（特記事項がなければ空文字）"
    }}
  ]
}}

【処理ルール】
1. フィラー語（えー、あのー、えっと、うーん、そのー、ま、まあ 等）を削除する
2. 言い直し・繰り返しを整理する
3. ヒアリングマニュアル①〜⑧の観点に従って要点を抽出する
   ① 長期目標（1年後・3年後・5年後の仕事像）
   ② 短期目標（半年後の状態）
   ③ 課題
   ④ 本人の取組内容
   ⑤ 職員の支援内容
   ⑥ 達成期間（原則6ヵ月）
   ⑦ 留意事項（体調チェックシートの推移観察）
   ⑧ 本人・家族の意向／総合的な支援の方針／本人の現状
4. 短期目標が文字起こしから複数読み取れる場合は goals 配列に2つ以上追加する。1つで十分なら1つのみ
5. etc（その他）は特記事項がなければ空文字

【文体】
- 「だ・である」調。簡潔に事実ベースで書く
- 一文は短く。だらだら書かない
- 受動・尊敬表現は最小限にする

【伏せ字ルール】
- 利用者名は実名を出さず「本人」または「ご本人」と表記する
- 職員名は「職員」と表記する
- 具体的な家族・地名・病院名は出さない

【文字起こしデータ】
{transcript}
"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=3000,
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = msg.content[0].text.strip()
        print(f'[SHIENKEIKAKU] Claude raw response (head): {raw[:300]}', flush=True)

        # ── JSON 抽出（Claude が ``` で囲む場合も対応） ──
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        raw_clean = match.group() if match else raw

        result = json_mod.loads(raw_clean)

        # スキーマの最低限の検証
        for key in ['intention', 'policy', 'situation', 'longGoal', 'goals']:
            if key not in result:
                result[key] = '' if key != 'goals' else []
        if not isinstance(result.get('goals'), list):
            result['goals'] = []

        return jsonify(result)

    except json_mod.JSONDecodeError:
        print(f'[SHIENKEIKAKU] JSON parse error: {raw[:500]}', flush=True)
        return jsonify({'error': 'AI応答の解析に失敗しました。もう一度「AI生成」を押してください。'}), 500
    except Exception:
        print(f'[SHIENKEIKAKU ERROR] {traceback.format_exc()}', flush=True)
        return jsonify({'error': '処理中にエラーが発生しました。時間をおいて再試行してください。'}), 500

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
・120〜200字厳守（120字未満は情報不足、200字超過は冗長。必ず文字数を数えて範囲内に収めること）
・利用者・職員の発話は「」でくくる
・当日の作業内容を必ず盛り込む
・利用者の体調・メンタル・様子を適切に記録する
・5W1Hを意識し、ほかのスタッフが読んでも現場の光景が浮かぶ内容にする
・個人名は使わず「〇〇さん」として表記する
・「他利用者」「他の利用者」は「ほかの利用者」に統一する
【文体ルール（おがっち構文）】
・一文は30〜40字以内で短く切る
・指示語（この・その・あの・それ）は使わない
・「しかし」「だが」は使わない。短文の連打で表現する
・結論から書き始める。体言止めを適宜活用する
・「〜れた」「〜られた」などの受動・尊敬表現は使わない。「〜した」「〜していた」の能動形で書く
・回りくどい表現を徹底排除する。以下は禁止パターン：
  NG「〜な様子だった」→OK「〜だった」（例：楽しそうな様子だった→楽しそうだった）
  NG「〜様子が見られた」→OK 事実を直接書く（例：意欲が低い様子が見られた→意欲が低かった）
  NG「〜することができた」→OK「〜できた」
  NG「〜を行うことができた」→OK「〜を行った」「〜できた」
  NG「〜という形で」→OK 削除して直接書く
  NG「〜といった感じだった」→OK 断定する
  NG「〜のほうに」→OK「〜に」
・修飾語は最小限。「非常に」「とても」「大変」は本当に必要なときだけ使う
【表記ルール】
・時刻は数字+コロン形式で表記する（例：10:00、10:30、14:00）
・野菜名はカタカナ（タマネギ・レタス・キャベツ等）
・その他は「例解辞典」の表記ルールに準拠
【漢字↔ひらがな統一ルール ※最重要・必ず全文に適用せよ】
以下は置換テーブルである。出力前に1語ずつ照合し、違反があれば必ず修正すること。
[ひらがなに開く（漢字で書いてはいけない）]
辛い→つらい／辛く→つらく／辛さ→つらさ
頂く→いただく／頂いた→いただいた
際→さい（例：「その際」→「そのさい」「〜の際に」→「〜のさいに」）
出来る→できる／出来た→できた
致す→いたす
下さる→くださる／下さい→ください
既に→すでに
殆ど→ほとんど
未だ→まだ
更に→さらに
但し→ただし
様々→さまざま
ライン→LINE（アプリ名称は正式表記）
〜な様子だった→〜だった（「様子だった」は冗長。「楽しそうだった」「落ち着いていた」のように直接書く）
〜様子が見られた→直接書く（「発話も少ない様子が見られた」→「発話も少なかった」。回りくどい観察表現を避け、事実を短く書く）
〜して見る→〜してみる
〜して置く→〜しておく
〜して欲しい→〜してほしい
[漢字で閉じる（ひらがなで書いてはいけない）]
ひきつづき→引き続き
おおむね→概ね
おこなう→行う／おこなった→行った
みられる→見られる／みられた→見られた
他利用者→ほかの利用者／他の利用者→ほかの利用者
【最終チェック指示】
JSON出力の直前に、全考察テキストを上記の置換テーブルと照合し、1語でも違反があれば修正してから出力せよ。"""
            msg = client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=2000,
                messages=[{'role': 'user', 'content': prompt}]
            )
            raw = msg.content[0].text.strip()
            print(f'[KANSAN] Claude raw response: {raw[:300]}', flush=True)

            # ── JSON パース（正規表現で確実に抽出） ──────────────────
            # Claude が ``` で囲む場合も、直接返す場合も対応
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                raw_clean = match.group()
            else:
                raw_clean = raw
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
    except Exception as e:
        # エラー詳細をログに出力（Railway の Deploy Logs で確認できる）
        print(f'[KANSAN ERROR] {traceback.format_exc()}', flush=True)
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
