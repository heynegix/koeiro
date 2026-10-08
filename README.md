# Anime Voice Changer — 登録声 / 逐次変換・一括変換（LavaSR）

マイク入力を、登録した声にリアルタイム変換して仮想ケーブル等へ出力します。
Windows 10/11 と Linux に対応。Original（素通し）・Female DSP・AI Voice
（MeanVC2＋一括変換＋LavaSR帯域復元）を切り替えて使います。

## 変換方法

- **逐次変換**：話しながら順に処理。遅延は小さいが帯域復元なし
- **一括変換（LavaSR）**：0.8秒の無音で発話を区切り、発話全体を変換してから
  LavaSRで帯域復元。最大60秒。話し終えてから数秒〜十数秒待つ方式
- **比較用：一括＋全部入り／自然寄せ**：聞き比べ用の実験モード。速度・声質の保証なし

## 必要なもの

- Python 3.12（Windows / Linux共通）
- PortAudio（Linuxのみ別途：`sudo apt install libportaudio2 portaudio19-dev`等）
- FFmpeg（声の登録時の音声読込に使用）
- 出力先の仮想オーディオデバイス（任意。Windows: VB-CABLE等、Linux: PipeWire/PulseAudioのnull-sink等）
- 約8GB以上の空き容量（AIモデル＋作業環境）

## セットアップ

Windows PowerShell / Linux bash のどちらでも、同名の手順です。

```sh
# 1. GUI用環境
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt        # Windows: .venv\Scripts\python

# 2. AIワーカー環境（MeanVC2＋LavaSR）
python3.12 -m venv vc_models/meanvc2/.venv
vc_models/meanvc2/.venv/bin/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
vc_models/meanvc2/.venv/bin/python -m pip install -r vc_models/meanvc2/requirements.lock.txt
# Linuxでは requirements.lock.txt から win32-setctime の1行を除いてください

# 3. AIモデル本体の配置（MeanVC2 checkpoint一式を vc_models/meanvc2/repo 配下へ）
# 4. ASR（任意・テキスト連動を使う場合のみ）
.venv/bin/python tools/install_asr.py

# 5. 起動
.venv/bin/python app.py
```

初回起動時は設定・ログは自動生成されます（`settings.json`・`logs/`は配布物に含みません）。

## 声の登録と選択

1. 声ライブラリの「＋ 音声ファイルから声を追加」から、本人の録音（BGMなし・20秒以上推奨）を登録
2. 一覧から声を選ぶ。アプリ標準の「標準ボイス」は最初から入っていて、
   登録なしでそのまま使えます（自分の登録ではないので「登録済み」とは表示しません）
3. ホームの「変換したい声」からも同じ声を選べます
4. 出力音声カードの「変換を試聴」で、ファイル変換の仕上がりを事前確認できます

登録した声・録音・ログは公開されません。詳しい同梱範囲は `PUBLISHING.md`、
第三者の権利表示は `THIRD_PARTY_NOTICES.md`、自作部分の許諾は `LICENSE` を参照。

## モニター（変換後の音を自分の耳で確認）

「設定」のモニターで、変換後の音を別のデバイス（ヘッドホン等）で聴けます。

- 開始前でも変換中でもON/OFF・デバイス・音量を変更できます（再Startは不要）
- メインの出力先と同じデバイスを選ぶと二重に聞こえるため警告が出ます
- 「このデバイスでテスト音を鳴らす」で、そのデバイスから短いチャイムを1回鳴らせます
- モニターは best-effort です。開始に失敗しても、変換・出力は止まりません

モニターは聴き取り用であり、遅延測定用ではありません（途切れ・溢れの回数を表示します）。

## 使い方ガイド

初回起動時に、スキップ可能なガイドが自動で開きます。各ページは実際の画面へ移動し、
説明している項目をハイライトします。スキップした場合は次回の起動でもう一度表示され、
「次回から表示しない」にチェックすると以降は表示されません。右下の
「？ 使い方ガイド」ボタンからいつでも開けます。

## 注意

- 人間の試聴評価・実機DiscordまでのEnd-to-End遅延保証はありません（モニターは聴き取り用）
- キャラクター音声（Beatrice等）は同梱しません。公式配布から利用者自身が取得してください
