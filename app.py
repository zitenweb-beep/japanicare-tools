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

# ── SNS投稿ジェネレーター ─────────────────────────────────────────
@app.route('/post-generator/generate', methods=['POST'])
@login_required
def post_generator_generate():
    import json as json_mod
    data = request.json or {}
    platform    = data.get('platform', 'x')
    theme_name  = (data.get('theme_name') or '').strip()
    theme_angle = (data.get('theme_angle') or '').strip()
    free_text   = (data.get('free_text') or '').strip()

    if not theme_name and not free_text:
        return jsonify({'error': 'テーマを選ぶか自由入力を記入してください'}), 400
    if not ANTHROPIC_KEY:
        return jsonify({'error': 'ANTHROPIC_API_KEY が未設定です（管理者に連絡）'}), 500

    # テーマブロックの構築
    if theme_name:
        theme_block = f'【テーマ】{theme_name}\n【切り口のヒント】{theme_angle}'
        if free_text:
            theme_block += f'\n【補足・現場エピソード】\n{free_text}'
    else:
        theme_block = f'【テーマ（自由入力）】\n{free_text}'

    # プラットフォーム別ルール
    if platform == 'x':
        platform_name = 'X (Twitter)'
        platform_rules = ('・X投稿用。全角140字ちょうど前後（130〜150字）で収める。\n'
                          '・改行は最大2箇所まで。冒頭で引きつけ、終わりに余韻。')
    else:
        platform_name = 'Threads'
        platform_rules = ('・Threads投稿用。180〜240字で少しゆとりを持たせる。\n'
                          '・改行を効果的に使い、リズムを作る（3〜5箇所）。')

    prompt = f"""あなたはジャパニケア札幌（合同会社JAPANICARE運営の就労継続支援B型事業所）の広報担当です。
以下の条件で、SNS投稿案を【3案】作ってください。

{theme_block}

【プラットフォーム】{platform_name}
{platform_rules}

【主体と店舗の関係（絶対に混同しないこと）】
・投稿の主体は、**就労継続支援B型事業所「ジャパニケア札幌」**（合同会社JAPANICARE運営）。
・**マノメオ（MANOMEWO）** は、ジャパニケア札幌が運営する「カフェ・バーを併設した雑貨店」の店舗名。
・関係：ジャパニケア札幌（事業所）＞ マノメオ（店舗）。ジャパニケア札幌の一機能としてマノメオがある。
・主語は原則「ジャパニケア札幌」または「私たち」。マノメオは、店舗・場所・働く現場の文脈でのみ登場させる（例：「マノメオで…」「マノメオに届いた…」）。
・「マノメオの厨房」「マノメオのカウンター」のように場所を細かく限定する書き方は冗長。基本は「マノメオ」単体で十分。
・「ジャパニケア札幌では〜」「マノメオで〜」のように、事業所の話か店舗の話かを自然に書き分ける。
・利用者・支援・工賃・運営・思想・哲学 → 主語は「ジャパニケア札幌」「私たち」
・店舗の出来事・お客様とのやりとり・商品・空間 → 主語は「マノメオ」
・マノメオが主語になるのは投稿の3本に1本以下が目安。基本はジャパニケア札幌の発信。

【ジャパニケア札幌の核となるスタンス（必ず滲ませる）】
・「全方位全肯定」「福祉をカジュアルに、もっと身近に」「クールでスタイリッシュな福祉」
・「私たちは障害者とはつきあっていない。おつきあいしているのは一人の人」というフラットな視線
・『障害者』というラベルを無力化したい。「困っている人がいたら、できる範囲で助け合おう」レベルの自然さ
・代表は現役のパニック障害。『いつかの自分が必要としていた居場所』を作っている
・工賃・居場所・主体性・凡々たる非凡・自分軸・誠実に一歩ずつ、などの語彙が自然に出る
・「真の支援員はお客様でした」「みんなでみんなを応援する」という関係観

【絶対NG】
・上から目線の支援者ポジション（「利用者様に寄り添い…」など）
・同情を誘う訴求（「障害者が作ったから応援を」など）
・きれいごとの定型文
・感動ポルノ、啓発的で説教くさい語り口
・ハッシュタグ羅列（1〜2個までなら可）
・絵文字の多用（0〜1個）
・過剰な丁寧語（「〜させていただきます」「〜のではないでしょうか」）
・利用者の実名（「利用者Aさん」のように伏せる）

【文体ルール：おがっち構文（厳守）】
・一文は30〜40字以内。情報を詰め込まない。
・指示語（この／その／あの／それ／これ）は極力使わない。対象を直接書く。
・「しかし／だが／けれども」は使わない。逆接は短文の連打か「──」で表現。
・体言止めを効果的に使い、余韻と強さを出す。
・結論ファースト。事実・結論を先に出し、理由は後。
・感情を説明せず、感情が伝わる事実を書く（例：「感動した」→「眠れなかった」）。
・五感（匂い・温度・触感・音）で書く。抽象論より具体的な場面。
・一人称は必ず「私」を使う（「僕」は使わない）。従業員全員で共有するツールのため。
・文末は必ず「です・ます調」で統一する（「〜です」「〜ます」「〜ません」）。「〜だ」「〜である」の断定調は使わない。
・ただし、過剰な丁寧語（「〜させていただきます」「〜のではないでしょうか」）は避ける。シンプルな「です・ます」で親密・対等・直球のトーン。
・体言止めは使ってOK（余韻と強さのため）。

【バズり型の素材（おがっち構文に溶かして使う）】
・冒頭は場面描写・時刻・固有名詞でリアリティを出す
・具体的なエピソード一点から思想へ展開
・最後に静かな問いかけで終わる

【出力形式】
以下のJSON形式のみで出力してください。説明・前置き・コードブロック記号（```）は一切不要です。
3案それぞれ、切り口や語り出しを変えて変化をつけてください。

{{
  "posts": [
    {{"text": "案1の本文"}},
    {{"text": "案2の本文"}},
    {{"text": "案3の本文"}}
  ]
}}"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=2500,
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = msg.content[0].text.strip()
        print(f'[POST_GEN] Claude raw response (head): {raw[:300]}', flush=True)

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        raw_clean = match.group() if match else raw
        result = json_mod.loads(raw_clean)

        # 文字数を実測値で付与
        posts = result.get('posts', [])
        for p in posts:
            p['char_count'] = len(p.get('text', ''))

        return jsonify({'posts': posts})

    except json_mod.JSONDecodeError:
        print(f'[POST_GEN] JSON parse error: {raw[:500]}', flush=True)
        return jsonify({'error': 'AI応答の解析に失敗しました。もう一度お試しください。'}), 500
    except Exception:
        print(f'[POST_GEN ERROR] {traceback.format_exc()}', flush=True)
        return jsonify({'error': '処理中にエラーが発生しました。時間をおいて再試行してください。'}), 500

# ── note執筆ツール ────────────────────────────────────────────────
@app.route('/note-generator/generate', methods=['POST'])
@login_required
def note_generator_generate():
    data = request.json or {}
    staff_name  = (data.get('staff_name') or '').strip()
    theme_name  = (data.get('theme_name') or '').strip()
    theme_angle = (data.get('theme_angle') or '').strip()
    free_text   = (data.get('free_text') or '').strip()

    if not staff_name:
        return jsonify({'error': 'スタッフの名前を入力してください'}), 400
    if not theme_name and not free_text:
        return jsonify({'error': 'テーマを選ぶか自由入力を記入してください'}), 400
    if not ANTHROPIC_KEY:
        return jsonify({'error': 'ANTHROPIC_API_KEY が未設定です（管理者に連絡）'}), 500

    # テーマブロックの構築
    if theme_name:
        theme_block = f'【テーマ】{theme_name}\n【切り口のヒント】{theme_angle}'
        if free_text:
            theme_block += f'\n【補足・現場エピソード】\n{free_text}'
        # タイトル文脈用のテーマラベル
        theme_label = theme_name
    else:
        theme_block = f'【テーマ（自由入力）】\n{free_text}'
        theme_label = free_text[:30]

    # 導入・結び（固定テンプレート）
    intro = (
        f'ジャパンにケアを！札幌市にある就労継続支援B型事業所'
        f'「ジャパニケア札幌」の{staff_name}です。\n'
        f'本日は、{theme_label}についてお話ししていきます。'
    )
    closing = (
        f'本日の話はここまで。いかがでしたか。\n'
        f'見学・体験は、随時募集しています。お電話のほか、各種SNSのDMからお問い合わせください。\n'
        f'以上、ジャパニケア札幌の{staff_name}。本日も、最後までおつきあいいただき、ありがとうございました。'
    )

    # AI生成部分のターゲット字数（導入・結びを除いた本文）
    intro_len = len(intro)
    closing_len = len(closing)
    body_target_min = 2000 - intro_len - closing_len - 100
    body_target_max = 2000 - intro_len - closing_len + 150

    prompt = f"""あなたはジャパニケア札幌（合同会社JAPANICARE運営の就労継続支援B型事業所）の広報担当・{staff_name}です。
note記事の【本文（中間部分）】を執筆してください。導入・結びは別途固定テンプレートで自動挿入されるため、本文のみを出力してください。

{theme_block}

【出力する本文の位置づけ】
・導入（自動挿入・書かない）：「ジャパンにケアを！札幌市にある就労継続支援B型事業所『ジャパニケア札幌』の{staff_name}です。本日は、〜についてお話ししていきます。」
・あなたが書くのは、↑この導入に続く本文。
・結び（自動挿入・書かない）：「本日の話はここまで。いかがでしたか。見学・体験は…」
・つまり、いきなり本題に入ってOK。再度の自己紹介や「本日は〜」の再掲はしない。最後も「いかがでしたか」等で閉じない（結びは自動で付く）。

【本文の字数】
・{body_target_min}字以上、{body_target_max}字以内。
・導入＋本文＋結びで全体2000字前後を目指すため、本文は必ずこの範囲に収める。

【主体と店舗の関係（絶対に混同しないこと）】
・記事の主体は、就労継続支援B型事業所「ジャパニケア札幌」（合同会社JAPANICARE運営）。
・マノメオ（MANOMEWO）は、ジャパニケア札幌が運営する「カフェ・バー併設の雑貨店」の店舗名。
・関係：ジャパニケア札幌（事業所）＞ マノメオ（店舗）。
・主語は原則「ジャパニケア札幌」または「私たち」。マノメオは店舗・場所の文脈でのみ登場させる。
・利用者・支援・工賃・運営・思想・哲学 → 主語は「ジャパニケア札幌」「私たち」
・店舗の出来事・お客様とのやりとり・商品・空間 → 主語は「マノメオ」

【ジャパニケア札幌の核となるスタンス（必ず滲ませる）】
・「全方位全肯定」「福祉をカジュアルに、もっと身近に」「クールでスタイリッシュな福祉」
・「私たちは障害者とはつきあっていない。おつきあいしているのは一人の人」というフラットな視線
・『障害者』というラベルを無力化したい。「困っている人がいたら、できる範囲で助け合おう」レベルの自然さ
・代表は現役のパニック障害。『いつかの自分が必要としていた居場所』を作っている
・工賃・居場所・主体性・凡々たる非凡・自分軸・誠実に一歩ずつ、などの語彙が自然に出る
・「真の支援員はお客様でした」「みんなでみんなを応援する」という関係観

【絶対NG】
・上から目線の支援者ポジション（「利用者様に寄り添い…」など）
・同情を誘う訴求（「障害者が作ったから応援を」など）
・きれいごとの定型文／感動ポルノ／説教くさい語り口
・ハッシュタグ／絵文字の多用
・過剰な丁寧語（「〜させていただきます」「〜のではないでしょうか」）
・利用者の実名（「利用者Aさん」のように伏せる）
・自己紹介の再掲（導入で済んでいる）
・「本日は〜」「いかがでしたか」等の導入・結びの再掲
・読者への媚び（「ぜひフォローを！」等の露骨な呼びかけ）

【文体ルール：おがっち構文（厳守）】
・一文は30〜40字以内。情報を詰め込まない。
・指示語（この／その／あの／それ／これ）は極力使わない。対象を直接書く。
・「しかし／だが／けれども」は使わない。逆接は短文の連打か「──」で表現。
・体言止めを効果的に使い、余韻と強さを出す。
・結論ファースト。事実・結論を先に出し、理由は後。
・感情を説明せず、感情が伝わる事実を書く（例：「感動した」→「眠れなかった」）。
・五感（匂い・温度・触感・音）で書く。抽象論より具体的な場面。
・一人称は「私」または「私たち」。「僕」は使わない（事業所全体の広報のため）。
・文末は「です・ます調（できます調）」で統一する。「〜です」「〜ます」「〜ません」「〜できます」。「〜だ」「〜である」の断定調は使わない。
・ただし、過剰な丁寧語は避ける。シンプルな「です・ます」で親密・対等・直球のトーン。
・体言止めは使ってOK（余韻のため）。

【改行ルール】
・2〜3行ごとに必ず空行を入れる（noteの読みやすさのため）。
・1段落は2〜3文までを目安に。
・段落間は \\n\\n（空行1つ）で区切る。

【構成のヒント】
・導入に続けて、いきなり具体的な場面・エピソード・問いから入る。
・エピソード→気づき→思想 の順で展開する。
・最後は静かな問いかけや、余韻のある一文で締める（ただし「いかがでしたか」は結びで自動挿入されるので使わない）。

【表記ルール（JAPANICARE職員考察ツールと統一）】
・時刻は数字+コロン形式（10:00、14:30）
・「ヶ」は「ヵ」に統一（3ヵ月／数ヵ月）
・以下は必ずひらがなに開く：
  辛い→つらい／頂く→いただく／際→さい／出来る→できる／下さい→ください
  既に→すでに／殆ど→ほとんど／未だ→まだ／更に→さらに／但し→ただし／様々→さまざま
  〜して見る→〜してみる／〜して置く→〜しておく／〜して欲しい→〜してほしい
・「良い」は「いい」または「よい」に開く（文脈で判断）
・以下は漢字で閉じる：
  ひきつづき→引き続き／おこなう→行う／みられる→見られる
  他利用者／他の利用者→ほかの利用者
・LINEは正式表記

【出力形式】
・本文テキストのみを出力。JSONや説明・前置き・コードブロック記号（```）は一切不要。
・見出し記号（#、■、【】）は使わない。段落改行のみで構成。
・本文の冒頭は、いきなり場面・エピソード・問いから始める。

さあ、本文を書いてください。
"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=4000,
            messages=[{'role': 'user', 'content': prompt}]
        )
        body = msg.content[0].text.strip()
        print(f'[NOTE_GEN] Claude body head: {body[:200]}', flush=True)

        # コードブロック記号が混入していたら除去
        body = re.sub(r'^```[a-zA-Z]*\n?', '', body)
        body = re.sub(r'\n?```$', '', body)
        body = body.strip()

        # 導入 + 本文 + 結びを結合（本文前後は空行で区切る）
        full_text = f'{intro}\n\n{body}\n\n{closing}'

        return jsonify({
            'text': full_text,
            'char_count': len(full_text),
            'body_char_count': len(body),
            'intro_char_count': intro_len,
            'closing_char_count': closing_len
        })

    except Exception:
        print(f'[NOTE_GEN ERROR] {traceback.format_exc()}', flush=True)
        return jsonify({'error': '処理中にエラーが発生しました。時間をおいて再試行してください。'}), 500

# ── SNS投稿ジェネレーター 音声文字起こし ──────────────────────────
@app.route('/post-generator/transcribe', methods=['POST'])
@login_required
def post_generator_transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': '音声ファイルがありません'}), 400

    audio_file = request.files['audio']

    # ファイル拡張子を元のファイル名から推測（webm/mp4/m4aなど）
    filename = audio_file.filename or 'audio.webm'
    ext = '.webm'
    for e in ['.webm', '.mp4', '.m4a', '.mp3', '.wav', '.ogg']:
        if filename.lower().endswith(e):
            ext = e
            break

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        model = get_whisper()
        segments, _ = model.transcribe(
            tmp_path, language='ja', beam_size=5,
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
        transcript = ''.join(parts).strip()

        if not transcript:
            return jsonify({'error': '音声を認識できませんでした。もう一度お試しください。'}), 400

        return jsonify({'transcript': transcript})

    except Exception:
        print(f'[POST_TRANSCRIBE ERROR] {traceback.format_exc()}', flush=True)
        return jsonify({'error': '文字起こし中にエラーが発生しました。'}), 500
    finally:
        if tmp_path:
            try: os.remove(tmp_path)
            except: pass

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
[カタカナ小書き文字の統一]
ヶ→ヵ（例：3ヶ所→3ヵ所／3ヶ月→3ヵ月／数ヶ月→数ヵ月）
※「〜ヶ」という表記は必ず「〜ヵ」に統一する。例外なし。
[文脈で判断するひらがな化]
良い→「いい」または「よい」（文脈で判断）
・会話調・口語的な文脈：「いい」（例：体調がいい／気分がいい／調子がいい）
・書き言葉・フォーマルな文脈：「よい」（例：よい結果となった／よい機会だった）
※「良い」と漢字で書いてはいけない。必ず「いい」か「よい」に開く。
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
