# Koeiro（声彩）

マイクの声を、登録した声にリアルタイムで変換するボイスチェンジャーです。
Windows 10/11 と Linux で動作し、変換後の音は仮想オーディオデバイス経由で
配信ソフトや通話アプリから使えます。

## できること

- **逐次変換** — 話しながら順に変換。遅延は小さい
- **一括変換（LavaSR）** — 0.8秒の無音で発話を区切り、発話全体を変換してから
  LavaSR で帯域復元（最大60秒）。話し終えてから数秒〜十数秒待つ方式
- **一括変換-最速** — 待ちを約0.6秒に縮めた速い版
- **Original / Female DSP** — 素通しと DSP 加工
- 声の登録・選択、モニター、初回ガイド

## 入手方法（推奨）

GitHub Releases（https://github.com/heynegix/koeiro/releases）から
OS用の配布物をダウンロードして使います。新しい版が出ると、アプリの
「設定 → アプリの更新」から自動で更新できます。

- Windows：`koeiro-vX.Y.Z-windows.zip` を展開し、`py -3.12 tools/setup_release.py` を実行
  （初回のみ。Python 3.12・git・uv が必要）。終わったら `.venv\Scripts\python app.py` で起動
- Linux：`koeiro-vX.Y.Z-linux.tar.gz` を展開し、`python3.12 tools/setup_release.py` を実行
  （初回のみ）。終わったら `.venv/bin/python app.py` で起動

展開先は書き込み可能な場所にしてください（更新時に差し替えます）。
以下はソースから実行する開発者向け手順です。

## 必要なもの（ソース実行時）

- Python 3.12（Windows / Linux 共通）
- FFmpeg（声の登録時の音声読込に使用）
- PortAudio（Linux のみ別途：`sudo apt install libportaudio2 portaudio19-dev` 等）
- 出力先の仮想オーディオデバイス（任意。Windows: VB-CABLE 等、Linux: snd-aloop（ALSA Loopback）等）
- 空き容量 約8GB（AIモデル＋作業環境）

## セットアップ（ソース実行。開発者向け）

### AIエージェントに任せる（コピペ用）

下の文章をそのまま Codex / Claude Code 等に貼ると、セットアップを全自動で行います。
通信量 約3GB・所要 20〜60分を見込んでください。

```text
Koeiro（このREADMEのあるディレクトリを作業場所にする）をセットアップしてください。
OSを判定し、WindowsならPowerShell、Linuxならbashで実行してください。
[.venv, vc_models/*/.venv, settings.json, logs/, models/user_voices/] は消さない・壊さないこと。gitへのコミット・pushは禁止。

1. Python 3.12系があるか確認（Windows: `py -3.12 --version`、Linux: `python3.12 --version`）。
   無ければOS標準の方法で用意してください（Windows: python.org か `winget install Python.Python.3.12`、
   Linux: deadsnakes 等）。git と uv も必要です。無ければ同様に用意してください。
2. `tools/setup_release.py --dry-run` で手順全体を把握してから本実行してください
   （Windows: `py -3.12 tools/setup_release.py`、Linux: `python3.12 tools/setup_release.py`）。
   モデル取得（約2GB）も既定で含まれます。後回しにする場合だけ `--no-models` を付けてください。
3. 1ステップでも失敗したら中断し、実行コマンドと末尾30行の出力を報告して指示を仰いでください。
   自己判断で版の固定・手順の省略・別手段への切替をしないでください。
4. 成功したら起動確認をしてください。GUI用Python（Windows: `.venv\Scripts\python`、
   Linux: `.venv/bin/python`）で `app.py --smoke-test` を実行します（画面なしの場合は
   `QT_QPA_PLATFORM=offscreen` を先に設定。Windows: `$env:QT_QPA_PLATFORM='offscreen'`）。
   エラーなく終了することと `logs/app.log` の末尾を確認して報告してください。
```

手で進める場合は以下を参照してください。

Windows は PowerShell、Linux は bash で実行します。`py` は
Windows の Python ランチャー、`python3.12` は Linux の Python 3.12 です。

```powershell
# Windows
# 1. GUI用環境
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# 2. AIワーカー環境（MeanVC2＋LavaSR）
py -3.12 -m venv vc_models\meanvc2\.venv
.\vc_models\meanvc2\.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.\vc_models\meanvc2\.venv\Scripts\python -m pip install -r vc_models\meanvc2\requirements.lock.txt

# 3. AIモデル本体の配置（下の「モデル配置」を参照）
# 4. 起動
.\.venv\Scripts\python app.py
```

```sh
# Linux
# 1. GUI用環境
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 2. AIワーカー環境（MeanVC2＋LavaSR）
python3.12 -m venv vc_models/meanvc2/.venv
vc_models/meanvc2/.venv/bin/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
vc_models/meanvc2/.venv/bin/python -m pip install -r vc_models/meanvc2/requirements.lock.txt
# Linuxでは requirements.lock.txt から win32-setctime の1行を除いてください
# （Windows専用パッケージのため。そのままではLinuxでインストールに失敗します）

# 3. AIモデル本体の配置（下の「モデル配置」を参照）
# 4. 起動
.venv/bin/python app.py
```

`vc_models/meanvc2/requirements.lock.txt` はリポジトリに同梱しています。
最小構成で試す場合は、報告済みの組み合わせ（numpy / scipy / librosa /
einops 0.8.0 / x-transformers 2.2.11 / safetensors / soundfile / psutil /
matplotlib / s3prl）でも推論が動作します。

### GPU（CUDA）で推論する
計算デバイスは設定の「実行デバイス」（自動 / CPU / GPU）で選びます。
初期値は自動で、GPUが見えれば使い、なければCPUにフォールバックします。
GPUを使うには、ワーカー環境の torch を CUDA 版に入れ替えてください
（計算式・重み・手順は同一で、TF32 は品質保持のため無効化されます）。

```powershell
# Windows（CUDA 12系の例。GPUとドライバに合う cuXXX を選んでください）
.\vc_models\meanvc2\.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

```sh
# Linux
vc_models/meanvc2/.venv/bin/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

CUDA が無い環境で「GPU」を選ぶと、ワーカーが明示エラーで停止します
（黙ってCPUに落ちません）。切り替えは停止中にどうぞ。

## モデル配置

アプリは起動時にダウンロードしません。以下を事前に配置してください。

### MeanVC2 本体（変換に必須）
- `vc_models/meanvc2/repo` に公式リポジトリ（https://github.com/ASLP-lab/MeanVC2）を配置
- チェックポイント（Hugging Face `ASLP-lab/MeanVC2` から取得）:
  - `repo/preprocess/ckpts/fastu2pp_80ms.pt`、`fastu2pp_160ms.pt`
  - `repo/ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors`、
    `meanvc2_120ms_40ms.safetensors`
  - `repo/ckpts/vocos/vocos.pt`

### 声登録用チェックポイント（声登録を使う場合のみ）
`tools/register_voice.py`（アプリ内の登録ダイアログから実行）は、AIワーカー
環境の Python で動作し、以下を要求します（`s3prl` はワーカー環境に含む）。
いずれも `vc_models/meanvc2/repo/preprocess/ckpts/` に置きます。

- `wavlm_large.pt` — kNN-VC 作者の公式ミラーから取得してください
  （`https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt`）。
  上流 MeanVC2 の案内にあった Microsoft の配布 URL は現在 404 を返します
- `wavlm_large_finetune.pth` — 公式 ECAPA 配布を `gdown` で取得します
  （Google Drive ID `1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP`）

### LavaSR（帯域復元。一括変換に必須）
同梱スクリプトが公式リポジトリ・公式チェックポイント・分離ベンダー一式を
用意し、`src/vc/lavasr_runtime.json` の SHA-256 と照合します。

```powershell
# Windows
.\.venv\Scripts\python tools\install_lavasr.py
.\.venv\Scripts\python tools\install_lavasr.py --check-only  # 検証のみ
```

```sh
# Linux
.venv/bin/python tools/install_lavasr.py
.venv/bin/python tools/install_lavasr.py --check-only  # 検証のみ
```

内容：`https://github.com/ysharma3501/LavaSR.git` の clone、
Hugging Face `YatharthS/LavaSR` の `enhancer_v2/*`・`denoiser/*` の取得、
`uv` による vendor（`vocos@matcha`・`encodec`）の分離インストール。
`uv` と `git` が必要です。

テキスト連動（任意）を使う場合のみ、`.venv/bin/python tools/install_asr.py` を実行します。
設定とログは起動時に自動生成され（`settings.json`・`logs/`）、配布物には含みません。

## 使い方

1. 右の「ボイスプリセット」で声を選び「選択」。アプリ標準の「標準ボイス」は最初から入っていて、登録なしで使えます。
   自分の声を追加するには「＋ 声を追加」から、本人の録音（BGMなし・20秒以上推奨）を登録
2. 中央の②で話し方（逐次変換／一括変換／一括変換-最速）を選ぶ
3. ホームで出力先を仮想ケーブル等にして Start。相手には選んだ声で届きます。
   Discordで使う場合は、Discord側のノイズ抑制をOFFにすることをおすすめします
   （変換後の声がノイズ抑制で途切れ・劣化するため）

初回起動時はスキップ可能なガイドが開きます（右下の「？ 使い方ガイド」からいつでも）。
「設定」のモニターを使うと、変換後の音を別のデバイス（ヘッドホン等）で聴けます。
モニターは聴き取り用で、遅延測定用ではありません。

## Linuxで通話アプリへ流す（仮想デバイス）

PortAudio（ALSA）には PipeWire / PulseAudio の仮想 sink が見えないため、
ALSA Loopback（snd-aloop）を使います。カーネルモジュールで、音質の劣化はありません。

```sh
sudo modprobe snd-aloop
echo snd-aloop | sudo tee /etc/modules-load.d/snd-aloop.conf  # 再起動後も有効
```

1. アプリの「デバイス一覧を更新」を押すと、出力先に Loopback が出ます
2. 出力先に Loopback の再生側を選びます（例：`Loopback: PCM (hw:1,0)`）
3. Discord等のマイク設定で、Loopback の録音側を選びます（例：`Loopback: PCM (hw:1,1)`）
4. `aplay -l` / `arecord -l` で Loopback が見えない場合は、カーネルが
   `snd-aloop` に対応しているか確認してください

## トラブルシューティング

- **後から接続したデバイス（Bluetooth ヘッドセット等）が出ない**:
  「デバイス一覧を更新」を押すと PortAudio を再初期化して再検出します。
  ハンズフリー等で 16 kHz のデバイスしか出ない場合、そのままでは
  `-9997 Invalid sample rate` になります（アプリは 48 kHz / 44.1 kHz 前提）。
  対応プロファイル・サンプルレートのデバイスを選んでください
- **声が出ない・止まる**: 入力・出力が同じ方式か（Windows は WASAPI 推奨）、
  入力レベルが緑〜黄（-20dB 前後）かを確認してください

## 遅延の内訳（実測）

計算速度（RTF）と遅延は別物です。GPUは前者だけを縮めます。

- **逐次変換**：モデル1440ms＋抑揚補完1760ms＝約3.1秒の固定遅延が常にかかります
  （ストリーミング実測）。初回はさらに約1秒の溜めが加わります
- **一括変換**：無音待ち0.5〜0.8秒＋変換（RTF×秒数）が話し終わってからかかります。
  自然寄せレシピ＋LavaSR仕上げです
- **一括変換-最速**：推論1ブロック（バッファ480ms）と補完先読み短縮（320ms）
  で固定遅延を約0.6秒に抑えた速さ優先のモードです

## 開発用メモ

- `models/meanvc2_120` / `ref20` / `ref60` の開発用プロファイルは削除済みです。
  アプリが提供するのは登録声（`models/user_voices`）のみです

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

- 遅延は環境に依存します。DiscordまでのEnd-to-End遅延の保証はありません
- キャラクター音声（Beatrice等）は同梱しません。公式配布から利用者自身が取得してください
- 登録する声は、使う権利のある本人の音声に限ってください
