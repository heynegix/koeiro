# Koeiro on macOS (Apple Silicon・実験的)

Mac mini (M4) で CPU のみで動作確認した手順です。
マイク → AI声変換 → 出力、BlackHole 経由で別アプリへ送れます。

## 動作環境

- macOS 14+ / Apple Silicon（M1/M2/M3/M4）
- Python 3.12 系
- CPU 推論のみ（CUDA/MPS は未使用）
- 音声は Core Audio（PortAudio）経由

## セットアップ

```bash
git clone https://github.com/heynegix/koeiro.git
cd koeiro
brew install portaudio ffmpeg
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3.12 tools/setup_release.py
.venv/bin/python app.py
```

`tools/setup_release.py` は macOS では torch を通常の PyPI から入れます（CPU index は macOS 用 wheel がないため）。

## 音声ルーティング

- 入力：マイク（Core Audio）
- 出力：BlackHole 等の仮想デバイス（例：`BlackHole 2ch`）
- モニター：ヘッドホン（出力専用ストリーム）
- 入力と出力は同じホスト API（Core Audio）にしてください。
- 48 kHz / 256 frames が安定です。

## Female DSP（ネイティブ）

```bash
# Apple Silicon の場合
python3 tools/build_native.py --platform mac-arm64
# Intel Mac の場合
python3 tools/build_native.py --platform mac-x64
```

生成物は `src/processors/native/female_dsp_arm64.dylib`（+ `.json`）です。
C++ 側は Windows の `__declspec` / SSE を macOS で無効化してあり、ARM64 では no-op の浮動小数モードで動きます。

## 制限事項

- GPU 推論なし（CPU のみ）。MPS 対応は今後の課題です。
- frozen バンドル／自動アップデートなし。ソース実行のみです。
  - `pick_asset(..., platform="mac", kind="frozen")` は `None` を返します。
- 設定は `~/Library/Application Support/Koeiro/`（frozen 時）です。
- BlackHole は別途インストールしてください。

## トラブルシュート

- `PortAudio` がない：`brew install portaudio` 後に `.venv` を作り直してください。
- デバイスが出ない：`--list-devices` で Core Audio の名前を確認してください。
- 音が途切れる：入出力を同じ 48 kHz、バッファ 256 にそろえてください。
