# Koeiro（声彩）

マイクの声を、登録した声にリアルタイムで変換するボイスチェンジャーです。
Windows 10/11 と Linux で動作し、変換後の音は仮想オーディオデバイス経由で
配信ソフトや通話アプリから使えます。

## できること

- **逐次変換** — 話しながら順に変換。遅延は小さい
- **一括変換（LavaSR）** — 0.8秒の無音で発話を区切り、発話全体を変換してから
  LavaSR で帯域復元（最大60秒）。話し終えてから数秒〜十数秒待つ方式
- **Original / Female DSP** — 素通しと DSP 加工
- 声の登録・ライブラリ・試聴、モニター、初回ガイド

## 必要なもの

- Python 3.12（Windows / Linux 共通）
- FFmpeg（声の登録時の音声読込に使用）
- PortAudio（Linux のみ別途：`sudo apt install libportaudio2 portaudio19-dev` 等）
- 出力先の仮想オーディオデバイス（任意。Windows: VB-CABLE 等、Linux: PipeWire/PulseAudio の null-sink 等）
- 空き容量 約8GB（AIモデル＋作業環境）

## セットアップ

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

# 4. 起動
.venv/bin/python app.py
```

テキスト連動（任意）を使う場合のみ、`.venv/bin/python tools/install_asr.py` を実行します。
設定とログは起動時に自動生成され（`settings.json`・`logs/`）、配布物には含みません。

## 使い方

1. 「声ライブラリ」の「＋ 音声ファイルから声を追加」で、本人の録音（BGMなし・20秒以上推奨）を登録
2. 変換したい声を選ぶ。アプリ標準の「標準ボイス」は最初から入っていて、登録なしで使えます
3. ホームで出力先を仮想ケーブル等にして Start。相手には選んだ声で届きます
4. 「変換を試聴」で、ファイル変換の仕上がりを事前確認できます

初回起動時はスキップ可能なガイドが開きます（右下の「？ 使い方ガイド」からいつでも）。
「設定」のモニターを使うと、変換後の音を別のデバイス（ヘッドホン等）で聴けます。
モニターは聴き取り用で、遅延測定用ではありません。

## 使用モデルとライセンス

自作部分は MIT です（[LICENSE](LICENSE)）。使っているAIモデル・ライブラリは次のとおりです。

| 用途 | モデル / ライブラリ | ライセンス |
|---|---|---|
| 声の変換 | MeanVC2（Vocos ボコーダ、WavLM/ECAPA 前処理を含む） | Apache-2.0。前処理チェックポイントは上流の表示が競合（MIT / CC BY-SA 3.0） |
| 帯域復元 | LavaSR（vendored vocos） | Apache-2.0（vocos は MIT） |
| Female DSP | Signalsmith Stretch / Linear | MIT |
| 音声入出力・UI | sounddevice（PortAudio）、PySide6 / Qt | MIT、LGPLv3 |
| F0解析（オフラインのみ） | FCPE / torchfcpe | MIT（アプリ実行時は NumPy 実装のみを使用） |
| テキスト連動（任意） | sherpa-onnx、ReazonSpeech / Vosk 日本語 | Apache-2.0 |

- 帰属表示の全文は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- 同梱しないもの：キャラクター音声（Beatrice 等。公式配布から利用者自身が取得）、
  研究用モデル（Seed-VC = GPLv3、Vevo2 = CC-BY-NC-ND-4.0、EZ-VC = CC-BY-NC-4.0 等）、
  LavaSR 同梱の `encodec`（メタデータが CC BY-NC 4.0 表示。アプリは未使用）
- 登録した声・録音・ログは公開・同梱されません。同梱範囲は [PUBLISHING.md](PUBLISHING.md)

## 注意

- 人間の試聴評価・実機DiscordまでのEnd-to-End遅延保証はありません（モニターは聴き取り用）
- キャラクター音声（Beatrice等）は同梱しません。公式配布から利用者自身が取得してください
- 登録する声は、使う権利のある本人の音声に限ってください
