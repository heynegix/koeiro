<p align="center">
  <img src="assets/icon.png" alt="Koeiro Icon" width="160">
</p>

<h1 align="center">Koeiro（声彩）</h1>

<p align="center">
  <strong>登録した声へリアルタイムに変換する次世代ローカルAIボイスチェンジャー</strong><br>
  Windows 10/11 & Linux 対応 ｜ MeanVC2 + LavaSR 帯域復元 ｜ Discord RPC 連携 ｜ 低スペック・CPU環境対応
</p>

<p align="center">
  <a href="https://negix.booth.pm/items/8981333"><img src="https://img.shields.io/badge/BOOTH-配布ページ-EC407A?logo=booth&logoColor=white" alt="BOOTH"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-0078D6" alt="Platform">
  <img src="https://img.shields.io/badge/PyTorch-CPU%20%2F%20CUDA%2012-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## 📖 概要

**Koeiro（声彩 / こえいろ）** は、マイクから入力された音声をリアルタイムで目的の声へと変換するAIボイスチェンジャーです。

超軽量な音声変換モデル **MeanVC2** に加え、高品質な超解像・帯域復元技術 **LavaSR** を統合。配信（OBS）、通話（Discord / Zoom）、VRChat などのVRメタバースに向けて、低遅延かつクリアなボイス変換を提供します。

GUIとAI推論ワーカーをプロセス分離する設計を採用しており、GUI側の軽快なレスポンスとAI推論の安定動作を両立。GPU非搭載のミニPC（Intel N100 / N150等）やCPU環境でも軽快に動作するよう、CPU 経路を実測して調整しています。

---

## 🚀 インストール & クイックスタート

### 🌟 【一番おすすめ】BOOTH版（ワンクリック簡単導入）
Pythonのインストールや面倒なコマンド操作なしですぐに使いたい方は、**BOOTH版のインストーラー**の利用を強くおすすめします。

👉 **[BOOTH 配布ページはこちら（negix.booth.pm/items/8981333）](https://negix.booth.pm/items/8981333)**

- ダウンロードしてインストーラーを起動するだけでセットアップが完了します。
- 初めてAIボイスチェンジャーを触る方や、手軽に配信・通話で使いたい方に最適です。

---

### 📦 GitHub Releases 版（ZIP / 手動セットアップ）

#### Windows の場合
1. [Releases](https://github.com/heynegix/koeiro/releases) から `koeiro-vX.Y.Z-windows.zip` をダウンロードし、任意のフォルダ（※日本語を含まない英数字パス推奨）に展開。
2. PowerShell を開き、展開先ディレクトリで初回セットアップを実行：
   ```powershell
   py -3.12 tools/setup_release.py
   ```
3. セットアップ完了後、アプリを起動：
   ```powershell
   .\.venv\Scripts\python app.py
   ```

#### Linux の場合
1. [Releases](https://github.com/heynegix/koeiro/releases) から `koeiro-vX.Y.Z-linux.tar.gz` をダウンロードして展開。
2. ターミナルで依存パッケージの導入と初回セットアップを実行：
   ```bash
   # PySide6 の実行ライブラリ（libgl1 等）も一緒に入れます
   sudo apt install -y libportaudio2 portaudio19-dev ffmpeg \
     libgl1 libegl1 libglib2.0-0 libxkbcommon0 libdbus-1-3 fontconfig
   python3.12 tools/setup_release.py
   ```
3. セットアップ完了後、アプリを起動：
   ```bash
   .venv/bin/python app.py
   ```

> [!TIP]
> アプリ起動後は「調整」引き出しの **「① 配線」** で **入力デバイス（マイク）** と **出力デバイス（VB-CABLE やヘッドホン等）** を指定してください。

---

## ✨ 主な機能

* **用途に合わせた変換モード**
  * **逐次変換（Streaming）**: 発話しながらリアルタイム・低遅延に変換。通話やゲームプレイに最適。
  * **一括変換（LavaSR 高音質）**: 無音区間（約0.8秒）で自動分割し、文全体を一括変換後に LavaSR で高域を復元。極めて自然で明瞭な音質を実現。
  * **一括変換 - 最速**: 判定待ち時間を約0.6秒まで短縮したレスポンス重視のバッチ変換。
  * **Original / Female DSP**: AIを介さない素通し（スルー）モード、および軽量DSPによるピッチ・フォルマント加工。
* **直感的な声の登録 & 切り替え**
  * 数十秒〜数分の音声ファイル（wav/mp3等）を追加するだけで、お好みのキャラクターボイスを即座に作成・選択可能。
* **Discord Rich Presence（RPC）連携**
  * Discord のステータスに「Koeiro で変換中」「使用中のボイス名」をリアルタイム表示。
* **CPU で実測した推論、GPU（CUDA）は実測済みの声だけ試す**
  * AIボイスは GPU 非搭載のミニPC（Intel N100 / N150 等）で実測した CPU 経路を既定にしています。GPU（CUDA）経路は Kaggle の Tesla T4 で計測し、CPU と -76 dB（最大 6.9e-4）まで一致して 2.0〜2.7 倍速いことを確認、cpu/cuda クリップの聴き比べも実施済みです。「自動」は GPU が見えるとき、実測済みの MeanVC2 に CUDA を選びます（CPU に固定したいときは「CPU」）。TF32 を制御して再現性を保ちます。
* **UI設計（ステージ1枚＋引き出し）**
  * メイン画面は入力レベル・遅延・ステータス表示に集中。詳細設定は「調整」「実験」の引き出しに収納。
  * 初回起動時のスポットライトガイドや、アプリ内からのワンクリック自動アップデートに対応。
* **話し方の演出・オーディオ処理**
  * **感情・抑揚プリセット**: Calm / Gentle / Energetic / Sad / Anime Light / Anime Expressive / Feminine を搭載。
  * **入力の整え（音量の自動均一化）**: 小声をAIが認識しやすい音量へ自動リフト（無音は増幅しません）。
  * **出力の仕上げ & ノイズ除去**: つなぎ目のノイズや耳に刺さる息音を穏やかに抑制。
  * **AI出力フォルマント**: ピッチを維持したまま声の共振を高域へ寄せ、明るい声色に補正。
* **配信・通話を加速するショートカット**
  * **シーン機能**: マイク、ボイス、配線、エフェクトをプロファイルとして保存・瞬時呼出。
  * **ホットキー**: `Ctrl+1/2/3` でモード変更、`Ctrl+D` で Original ↔ 直前変換をトグル。
  * **音声コマンド（β）**: ウェイクワード「コエイロ」＋「止めて」「次の声」などのローカル音声操作。
* **拡張パック（DLC）& 実験機能**
  * レイテンシ・ガバナ、囁き検出モード、LAN内GPUを活用する分散推論、話者自動振り分け、Kokoro-82M による音声合成（TTS）などを搭載。

---

## 🛠️ 動作要件

| 項目 | 要件 |
| :--- | :--- |
| **OS** | Windows 10 / 11, Linux (Ubuntu 22.04+ 推奨) |
| **Python** | 3.12 系 |
| **必要容量** | 空き容量 約8GB 以上（モデル群を含む） |
| **外部ツール** | **FFmpeg**（音声入出力用）、**git** |
| **Linux 追加依存** | `libportaudio2`, `portaudio19-dev` ＋ PySide6 の実行ライブラリ（`libgl1` など。下の Linux 手順の `apt install` をそのまま使ってください） |
| **仮想オーディオ** | 変換音声を他アプリへ渡すための仮想デバイス（任意）<br>・Windows: [VB-CABLE](https://vb-audio.com/Cable/) 等<br>・Linux: `snd-aloop`（ALSA Loopback）等 |

---

## 🧑‍💻 開発者向けセットアップ（ソースコードから実行）

```bash
# 1. リポジトリのクローン
git clone https://github.com/heynegix/koeiro.git
cd koeiro

# 2. 仮想環境の作成と有効化
python3.12 -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux:
source .venv/bin/activate

# 3. 依存ライブラリのインストール
pip install -r requirements.txt
pip install -r requirements-ai.txt

# 4. アプリケーションの起動
python app.py
```

AIボイスの GPU（CUDA）経路は Kaggle の Tesla T4 で計測済みです（CPU と -76 dB まで一致、
2.0〜2.7 倍速。cpu/cuda クリップの聴き比べも実施済みで、「自動」は GPU が見えるときに
実測済みの MeanVC2 へ CUDA を選びます。CPU 固定は「CPU」）。手元の GPU で同じ計測を
やり直す手順は `notes/kaggle_gpu_verification.md` にあります。

---

## 🗣️ 音声合成（TTS・任意機能）

軽量なテキスト読み上げモデル **Kokoro-82M**（Apache-2.0）を利用可能です。  
必要な場合は以下のスクリプトを実行してモデルを取得してください：

```bash
python tools/install_tts.py
```

---

## 📄 ライセンス

本プロジェクトは [MIT License](LICENSE) のもとで公開されています。  
使用しているサードパーティ製モデルおよびライブラリのライセンス詳細については [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) をご確認ください。
