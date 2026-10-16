# Kaggle で GPU（CUDA）経路を検証する

issue #6 の報告で、実行デバイスを GPU にしたときの AI 起動失敗が再現しました。
開発機に GPU が無い（torch 2.5.1+cpu）ため、この経路はこれまで一度も実機で走らせていません。
`src/vc/config.py` の `CUDA_VERIFIED_BACKENDS` が空だったのはそれが理由で、「自動」は MeanVC2 を
CPU に留めていました。2026-10-15 の聴感確認を受けて `('meanvc2',)` を追加済みです
（「検証後に」参照）。

これは聴感の検証ではありません。音声デバイスを開かず、同じ音声を通した出力を数で突き合わせるだけです。

## 0. push 前の修正を測る（clone は main を取るため）

ノートブックは GitHub の `main` を clone します。**まだ push していない修正は clone に入らない**
ので、push 前に測る場合は変更したソースを Kaggle Dataset にして上書きしてください。重みは §2 の
curl、音声とモデル定義は clone 側で揃うため、ソースだけで足ります。

```bash
# ローカルで。モデル・音声・作業用ファイルは含めない
tar -czf koeiro-worktree.tgz src tools tests README.md notes
```

この `koeiro-worktree.tgz` を Kaggle の Datasets → New Dataset にアップロードし、§2 の
`git clone` の直後に次を足します（`<slug>` はアップロードした Dataset のスラッグ）。

```bash
!tar -xzf /kaggle/input/<slug>/koeiro-worktree.tgz -C /kaggle/koeiro
```

push 済みならこの節は不要で、clone だけで最新が入ります。

## 1. ノートブック

Kaggle Notebook、Accelerator を **GPU T4 / GPU L4**、Internet **ON**。
セクション 2 以降を Code cell に順に実行します。

## 2. コードと重み

同梱の声 `user_3b11387911244c6e9a013b42fa88a458`（reference.wav / fixed_embedding.npy /
runtime.json）は公開リポジトリに入っているので、追加のアップロードは不要です。

```bash
!git clone -q https://github.com/heynegix/koeiro /kaggle/koeiro
!git clone -q https://github.com/ASLP-lab/MeanVC2 /kaggle/koeiro/vc_models/meanvc2/repo
!git -C /kaggle/koeiro/vc_models/meanvc2/repo checkout -q 13acf84c1bf135ea5edad9c245b345289b06b33e
```

`setup.json` に刻んだ revision と同じコミットに checkout してください。
重みは `runtime.json` が sha256 を要求するので、名前の違うファイルは混ざりません。

```bash
!mkdir -p /kaggle/koeiro/vc_models/meanvc2/repo/preprocess/ckpts /kaggle/koeiro/vc_models/meanvc2/repo/ckpts/pretrained_models /kaggle/koeiro/vc_models/meanvc2/repo/ckpts/vocos
!curl -sL -o /kaggle/koeiro/vc_models/meanvc2/repo/preprocess/ckpts/fastu2pp_160ms.pt https://huggingface.co/ASLP-lab/MeanVC2/resolve/main/fastu2pp_160ms.pt
!curl -sL -o /kaggle/koeiro/vc_models/meanvc2/repo/ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors https://huggingface.co/ASLP-lab/MeanVC2/resolve/main/meanvc2_120ms_40ms.safetensors
!curl -sL -o /kaggle/koeiro/vc_models/meanvc2/repo/ckpts/vocos/vocos.pt https://huggingface.co/ASLP-lab/MeanVC2/resolve/main/vocos.pt
```

## 3. 依存

`requirements.lock.txt` は CPU ビルドの torch を刻んでいるので、torch / torchaudio /
torchvision の 3 行だけ除外し、同じバージョン番号の CUDA 版を**先に**入れます（順番が逆だと
torchmetrics や torchdiffeq が依存として PyPI の torch を巻き込みます。バージョンをずらすと
CPU 側の結果と比べられません）。`^torch` のような広い grep は torchdiffeq / torchmetrics まで
消してしまい、MeanVC2 の `src.model` が import 時に `torchdiffeq` を要求するため失敗します。

```bash
!pip install -q --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
!grep -vE '^(torch|torchaudio|torchvision)==' /kaggle/koeiro/vc_models/meanvc2/requirements.lock.txt > /kaggle/requirements.txt
!pip install -q -r /kaggle/requirements.txt
!python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. 計測

同じ音声（既定はその声の reference.wav）を CPU 経路と CUDA 経路に通し、出力を差し引きます。

```bash
!cd /kaggle/koeiro && python tools/benchmark_meanvc2_realtime.py --devices cpu cuda --seconds 20 --threads 4 --report /kaggle/working/gpu.json
```

`/kaggle/working/gpu.json` と `gpu.cpu.wav` / `gpu.cuda.wav` が出力です。issue に貼るのは
`comparison` と 2 本の `rtf`、それと `routes` の `torch` / `cuda_name` の値です。

## 5. 結果の読み方

- `routes` に `error` がある: GPU 経路はまだ立ち上がりません。メッセージ全文と traceback を
  セル出力ごと貼ってください（`--devices cuda` だけで実行すると GPU 経路の失敗が
  いちばんはっきり出ます）。
- `comparison.identical` が true: 両経路がビット単位で一致。`CUDA_VERIFIED_BACKENDS` に
  `'meanvc2'` を足せる状態です。
- `identical` が false でも差が小さい: float32 の丸め差としては自然です。判定の目安は
  `max_abs_diff` が 1e-3 より十分小さく、`diff_to_signal_db` が -60 dB 前後（この目安自体は
  実測値ではないので、数値を貼って相談してください）。そのうえで実際に聴く確認へ進みます。
- 差が大きい、または音がおかしい: GPU 経路は却下。`自動` が CPU を選んでいるので運用は壊れません。

## 実測結果（2026-10-15 / Kaggle Tesla T4）

private ノートブック `heynegix/koeiro-meanvc2-cuda-verification` を GPU セッションで上から
実行した結果です。v3（この文書を更新した後のソースを埋め込んだもの）を本記録とします。
v2 でも同じ計測をして、比較値は 16 桁まで一致しました。セッション全体で約 2 分（うち計算は
約 110 秒）、音声デバイスは開いておらず、聴感の確認でもありません。入力は同梱の
`reference.wav`（28.2 秒）の先頭 20 秒、320,000 サンプル。環境は python 3.13.15 /
torch 2.11.0+cu128 / 4 vCPU / Tesla T4（2 枚見えているうち 1 枚だけ使用）です。埋め込んだ
ソースの sha256 は run.log に出ています。この記録を書いた後に `src/vc/meanvc2.py` の docstring へ
上の 2 回分の数値を折り込んだため作業ツリーのダイジェストは埋め込み時と違いますが、変わったのは
説明文だけで計算部分は同一です。

| 構成 | CPU RTF | CUDA RTF | 倍率 | CPU chunk平均 | CUDA chunk平均 |
| --- | --- | --- | --- | --- | --- |
| 出荷条件（VC 6 ブロック + Vocos 36 フレーム） | 0.191 | 0.093 | 2.05 | 30.5 ms | 14.9 ms |
| グループなし（VC 1 + Vocos 1） | 0.518 | 0.284 | 1.82 | 82.8 ms | 45.5 ms |

- 一致: 出荷条件で `max_abs_diff 6.9e-4`、`diff_to_signal_db -76.36 dB`（`identical` は false）。
  グループなしでは `6.4e-4` / `-79.98 dB`。float32 の丸め差の範囲で、出力は両方とも有限、
  RMS は 0.0982 で一致しました。
- 段ごとの平均（出荷条件、1 チャンクあたり）: features 14.9 → 9.3 ms、VC 10.4 → 4.8 ms、
  Vocos 5.0 → 0.66 ms。GPU が効くのは VC と Vocos で、Vocos は 7.5 倍です。
- GPU で残るのは kaldi フロントエンド（CPU 実行）と毎ホップの転送で、14.9 ms のうち 9.3 ms が
  features 段（チャンク平均の 62%）でした。ここが次の上限になります。
- 同じ計測の v2 では出荷条件が CPU 0.159 / CUDA 0.078（2.04 倍）でした。共有 VM なので
  絶対時間は ±15% ほど動きますが、倍率は 2.0 前後で安定しています。
- `traced_devices_moved` は CUDA で 1、CPU で 0。設計どおり、GPU のときだけ wenet の位置
  エンコーダが持つ device 定数を書き換えていることを実測で確認できました。
- 起動（warmup）は CUDA で 5 秒前後、CPU で 1 秒弱。オフライン計測なので聴感の対象では
  ありませんが、初回起動の待ちは GPU のほうが長くなります。
- 出荷条件の chunk p95 は CPU 87.7 ms / CUDA 33.9 ms。遅延のばらつきも GPU で小さくなります。
- まとめ: CUDA 経路は実機で動作し、出荷条件で**CPU の約 2 倍**（2.04 倍 / 2.05 倍）。ただし
  ビット単位一致ではないため追加当時は保留とし、2026-10-15 の聴感確認を経て
  `CUDA_VERIFIED_BACKENDS` に追加済みです（下記）。

### whole-file（一括変換）ルートと ASR キャプチャ（v4 → v5 / 2026-10-15）

private ノートブック `heynegix/koeiro-meanvc2-captured-whole-file-conversion` で、配信ルート
そのもの（`OfflineAsrStage` / `OfflineVcStage`、`tools/benchmark_meanvc2_offline.py`）を T4 で
通しました。ステージを CUDA graph へ焼いて再生する buffered ルートのうち ASR 側が v4 で
キャプチャに失敗し、原因を特定して `src/vc/meanvc2.py` を修正、v5 で成立を確認しました。

- 失敗（v4）: 6 回の試行すべて同じ場所で `RuntimeError: Cannot copy between CPU and CUDA
  tensors during CUDA graph capture unless the CPU tensor is pinned.`。op は
  `aten::to(pe, dtype=6, layout=0, device=torch.device("cpu"))`。トレースの位置エンコーダは
  定数 device 版の `pe.to(...)` を持ち、`retarget_devices` は device 定数を書き換えますが、
  `pe` は素の属性なので freeze 後も CPU に残り、eager では毎ホップの未ピン H2D コピーとして
  通っていました（記録中は拒否）。
- 修正: `load()` の freeze / retarget 直後に `pos_enc.pe = pos_enc.pe.to(self.device)` を行い、
  `self.position_table` をそのテンソルへ結ぶ。CPU では同じオブジェクトが返るため値も挙動も
  不変（ローカル再確認: フロントエンド 2 種の bit 一致、meanvc2 テスト 118 件緑）。
- v5 実測（セッション 144.2 s、torch 2.11.0+cu128 / T4。run.log と `.build/kaggle_replay_out5/`）:

| 経路 | 20 s 音声の秒 | RTF | 倍率 |
| --- | --- | --- | --- |
| CPU whole | 3.427 | 0.171 | 1.00 |
| CUDA whole | 1.380 | 0.069 | 2.48 |
| CUDA buffered（replay） | 0.809 | 0.0405 | CPU whole 比 4.24 |

- `replay_stats`: cuda は `captured true / asr_graphs 1 / vc_graphs 4 / replays 64`、
  cpu は `captured false / replays 0`（CPU に記録は無い）。
- `buffered_comparison.identical` は cpu / cuda とも true（`max_abs_diff 0.0`）。replay を
  含む cuda 出力が eager とビット一致です。
- probe（`tools/probe_meanvc2_capture.py`）: `retargeted device constants: 1`、ASR stage /
  VC 全形状 / host scalars / global・thread_local・relaxed の全モードが OK。FAILED は
  device scalars だけになり、失敗点は `int(offset)` の readback
  （`cudaErrorStreamCaptureUnsupported`）。トレースが offset をホストへ読み返すため、
  **素の device スカラは原理的に記録できない**ことが実機で確定しました。host scalars が
  正しい処方です。
- 参考: 同じ v5 のリアルタイム計測は出荷条件 CUDA RTF 0.072（v3 0.093、cpu 比 2.68）、
  グループなし 0.244（v3 0.284）。pe コピーが消えた分の改善を含みます（共有 VM の揺れ ±15%）。
- cpu vs cuda の差は v4 と同一の `max_abs_diff 2.95e-3 / -65.39 dB`。加えて v4 と v5 の
  `offline.cpu.wav` / `offline.cuda.wav` はサンプル値がビット一致でした（ファイル差は
  libsndfile が書く PEAK チャンクの時刻 2 バイトだけで、約 43 分の実行間隔と一致）。
  修正は実機でも数値を一切動かしていません。埋め込んだ `src/vc/meanvc2.py` の sha256 は
  `3849cdbd…` でローカルと一致。
- ローカル（修正後）: フルスイート 1250 passed / 13 skipped（`-X dev -W default`、3 分 54 秒）。
  meanvc2 env では 36 + 82 + 21 件と、2 フロントエンドの buffered bit 一致チェックが緑。
- 出力回収は `kernel_output` の `download_zip`（`curl -H "Authorization: Bearer
  <~/.kaggle/access_token>"`）→ `Expand-Archive` が確実でした。

### 試聴用クリップ（v6 / 2026-10-15）

同じノートブックの v6 で、聴感確認の材料を出力しました。入力はプロモ用に本人が提供した
`promo/public/audio/voice_demo_original.wav`（48 kHz / 3.44 s、リポジトリ管理外）で、3 回を
0.35 s の無音で区切って 11.017 s に整形し、`stage_voice` が sha256 `4246a214…` 付きで
埋め込みます（`--source` で両ベンチマークへ渡す）。出力は cpu/cuda のペア 3 組:

- `grouped_default.{cpu,cuda}.wav`（配信構成のストリーミング、16 kHz）— アプリのルート
- `ungrouped.{cpu,cuda}.wav`（チャンク 1 個ずつ、16 kHz）
- `offline.{cpu,cuda}.wav`（一括変換、48 kHz）

cuda 側は `captured true / asr_graphs 1 / vc_graphs 4 / replays 38`、`buffered_comparison`
は cpu/cuda とも bit 一致。cpu vs cuda 差は -73.7 / -76.1 / -67.3 dB で v5 と同じ水準、
RTF は 2.48 / 1.82 /（whole）2.35 倍、buffered は CPU whole 比 3.11 倍。回収先は
`.build/kaggle_replay_listen/`（セッション 358673527、cancel 済み）。

聴感: 本人が `grouped_default` / `ungrouped` / `offline` の cpu/cuda ペア 3 組をローカル再生で
比較し、区別できないと確認しました（2026-10-15、ユーザー試聴。Kaggle 側は音声デバイスを
開かずクリップを書くだけ、実機マイク・Discord 試験ではありません）。これを受けて
`CUDA_VERIFIED_BACKENDS` を `('meanvc2',)` にしました。

## ローカルで済んでいる確認

```bash
python tools/verify_meanvc2_trace_devices.py
```

- 同梱の `fastu2pp_160ms.pt` と `vocos.pt` のグラフを走査しました（2026-10-15、torch 2.5.1+cpu）。
  結果は ASR 側に device を書き込んだノードが 1 個（`aten::to` の `cpu`）、Vocos に 0 個、
  そして device を渡されていない確保はどちらにも 0 個です。
- だから GPU に載せるには 2 段必要でした。`torch.jit.freeze` の前に `.to(device)` して重みが
  グラフ定数になる前に動かし（報告の 1 番目のエラー）、さらにグラフが書き込んだ device 自体を
  `retarget_devices` で実際の実行先へ書き換えます。ここが報告の 2 番目
  （`found at least two devices, cpu and cuda:0`）の原因で、wenet の位置エンコーダが
  `pe.to(device=cpu)` として計測機の device をそのまま持っていました。
  CPU 経路では書き換えは 0 件、つまり何も動きません。
- ソルバのノイズは CPU のジェネレータから引いてデバイスへ移すようにしました。
  CUDA の generator は別アルゴリズムなので、これをしないと 2 つの経路は引き算できません。
- 変更前後の CPU 出力は `max_abs_diff 0.0`（2026-10-15 再測: `git show HEAD:src/vc/meanvc2.py` を
  `src.vc.meanvc2` として読み込んだ計測と、作業ツリーの計測に同じ参照音声 4.8 秒を
  `--devices cpu` で通した。出力長 76,800 サンプル一致、`np.array_equal` も true）。
- `retarget_devices` は `torch.jit.freeze` の後でも効きますが、**一度でもグラフを実行した後は
  実行計画がキャッシュされ、書き換えが反映されません**。バックエンドは `load()` の中で freeze
  直後に呼び、初回実行は `warmup()` なので順序は満たしています（`tests/test_meanvc2_quality_math.py`
  の `RetargetDeviceTests` がこの順序を固定）。
- Kaggle の CPU 経路は CUDA ビルドの torch 上で動きます。Windows の `+cpu` ビルドと同じかは
  上記のローカル再現と突き合わせて判断してください。

## 検証後に（2026-10-15 の実測を受けた状態）

1. 完了: `CUDA_VERIFIED_BACKENDS` を `('meanvc2',)` にしました。2 経路は -73.7〜-76.1 dB
   （一括変換は -65〜-67 dB）でビット一致ではないため、本人が v6 の cpu/cuda クリップ 3 組を
   ローカル再生で比較し、差を聞き分けられないことを確認しています（2026-10-15、ユーザー試聴。
   実機マイク・Discord 試験ではありません）。`src/vc/config.py` の 1 行と
   `tests/test_compute_device.py::test_auto_only_reaches_a_gpu_that_someone_measured` を
   更新済みです（同テストは「自動」が未測定の声を GPU に載せない規則を固定しています）。
2. 完了: `src/vc/meanvc2.py` の docstring と GUI の実行デバイス表記を実測に合わせました
   （「未検証」→「実測済み」、ツールチップに -76 dB と 2.0 倍）。
3. 完了: README の GPU 記述を実測値に更新しました。

## 6. Kaggle MCP（エージェントから操作する・任意）

この検証自体はセル実行でも足りますが、エージェントにノートブックの作成・実行・出力回収まで
任せるなら MCP（Model Context Protocol）サーバーを使います。

1. 公式（ホスト型・推奨）: `https://www.kaggle.com/mcp`。サーバーのインストールは不要で、
   OAuth 対応クライアントか `KGAT_` トークンの Bearer 認証で接続します。コンペ・データセット・
   ノートブック（セッション開始/状態/取消、`save_notebook` は保存して上から実行）・モデル・
   ベンチマークを扱えます（https://www.kaggle.com/docs/mcp）。
2. コミュニティ（ローカル stdio）: `uvx kaggle-mcp-server`
   （https://github.com/Galaxy-Dawn/kaggle-mcp）。公式に無いディスカッション系ツール付き。
   `KAGGLE_API_TOKEN` か `~/.kaggle/kaggle.json` を使います。

クライアント設定の例。VS Code は `.vscode/mcp.json`（キーは `servers`）:

```json
{
  "servers": {
    "kaggle": { "type": "http", "url": "https://www.kaggle.com/mcp" }
  }
}
```

Claude Desktop / Cursor など `mcpServers` 形式のクライアントで公式ホスト型を使う場合は
`npx mcp-remote` 経由:

```json
{
  "mcpServers": {
    "kaggle": {
      "command": "npx",
      "args": ["mcp-remote", "https://www.kaggle.com/mcp"]
    }
  }
}
```

ローカル stdio 版（コミュニティ）なら:

```json
{
  "mcpServers": {
    "kaggle": {
      "command": "uvx",
      "args": ["kaggle-mcp-server"],
      "env": { "KAGGLE_API_TOKEN": "KGAT_xxxxxxxxxxxx" }
    }
  }
}
```

トークンは Kaggle の Settings → API → Create New API Token（`KGAT_...`）。クライアント設定に
直書きせず、環境変数から読ませるか、公式ホスト型なら
`--header "Authorization: Bearer $KAGGLE_API_TOKEN"` のように渡してください。
`~/.kaggle/kaggle.json`（username/key）は旧来の形式で、CLI も受け付けます。

### この環境への導入記録（2026-10-15）

- サーバーは `uv tool install kaggle-mcp-server --with "mcp<2"` で導入しました。PyPI 0.2.0 は
  `mcp` 2.x で `mcp.server.fastmcp` が無くなって起動しないため、`mcp<2`（1.30.0）を添えています。
  素の `uvx kaggle-mcp-server` は今のところ同じ理由で失敗します。
- 実行ファイル: `%USERPROFILE%\.local\bin\kaggle-mcp-server.exe`
- 起動確認済み: initialize → `serverInfo kaggle 1.30.0`、tools/list → 51 ツール
  （dataset_create / file_upload / kernel_push / kernel_session_create|status|output|cancel /
  competition_* / discussions_* など）。トークン無しでも一覧までは応答します。
- 登録先: `%USERPROFILE%\.agents\mcp.json`（Freebuff デスクトップは
  `{cwd}/.agents/mcp.json` → `{cwd}/../.agents/mcp.json` → `{homedir}/.agents/mcp.json` の順に読む）。
  `KAGGLE_API_TOKEN` の値だけを `KGAT_...` に置き換え、Freebuff を再起動するとツールが現れます。

注意:

- ホスト型の `save_notebook` はデータソース指定が落ちるという報告が Kaggle の product-feedback に
  あります（https://www.kaggle.com/discussions/product-feedback/727647）。保存後に
  ノートブックの入力（コンペ/データセット）を確認してください。
- MCP が扱うのは Kaggle 上のノートブック/データで、ローカルのこのリポジトリを操作するものでは
  ありません。ローカルからの自動化は `pip install kaggle` の CLI でも足ります。
- この環境の Kaggle MCP は接続済みです（2026-10-15）。ただし `kernel_push` は title / text /
  language / kernel_type / is_private しか受け付けず、**GPU（`machine_shape`）と
  `enable_internet` を指定できません**。指定しないと CPU・ネットワーク無効のノートブックに
  なるため、保存だけは同じ kagglesdk から `machine_shape='NvidiaTeslaT4'` と
  `enable_internet=True` を付けて行いました。実行・状態・出力の回収は MCP の
  `kernel_session_*` で足ります。

### エージェントから回したときの手順と落とし穴（2026-10-15 に実際に通した）

1. ノートブックのソースは `.build/kaggle_gpu_driver.py`。`@@PATCHED@@` の位置に、作業ツリーの
   `src/vc/meanvc2.py` と `tools/benchmark_meanvc2_realtime.py` を base64 と sha256 で埋めてから
   `.build/kaggle_meanvc2_gpu.ipynb` を作ります（clone は main を取るので、未 push の修正は
   この方法で渡します。実行時は書き換えたファイルの sha256 をログに出しています）。
2. `.build/kaggle_push.py` で保存（1 回目は新規、2 回目以降は `slug` 必須）。
3. MCP の `kernel_session_create` で実行開始。ノートブックは上から実行されます
   （papermill、`allow_errors=False` なのでセル内の例外はその場で止まります）。
4. `kernel_session_status` をポーリングし、`COMPLETE` になったら `kernel_session_output` の
   各ファイル URL（署名付き）から `summary.json` と `run.log`、`*.cpu.wav` / `*.cuda.wav` を取ります。
5. 落とし穴は 3 つ。(a) 同じ title で保存し直すと 409 で拒否される（`slug` を使う）。
   (b) 前のセッションが残っていると `kernel_session_create` が 409 になる。
   `CancelKernelSession` は `kernel_session_id`（作成応答の `metadata.kernelSessionId`）を
   直接渡すと通りました（`get_kernel_session_status` は session id を返さないため、
   MCP の `kernel_session_cancel` は空の id を送って 403 になります）。
   (c) セッションのログは status 応答の `log` フィールドに JSON 行で入るので、失敗時はそこを読みます。

### opencode への導入記録（2026-10-15）

Freebuff とは別に、opencode のグローバル設定
`%USERPROFILE%\.config\opencode\opencode.json` の `mcp` へ `kaggle`（local stdio）を登録しました。
実行ファイルは同じ `%USERPROFILE%\.local\bin\kaggle-mcp-server.exe` です。

トークンは設定に書かず、kagglesdk がネイティブに読む `~/.kaggle/access_token`
（内容は `KGAT_...` の 1 行）を唯一の入力場所にしました。解決順は
`KAGGLE_API_TOKEN` 環境変数 → `~/.kaggle/access_token` → `~/.kaggle/access_token.txt` です
（`kagglesdk/kaggle_env.py`）。このファイルには Freebuff 側で使っていたトークンを入れてあります。
差し替えるときはこの 1 ファイルを上書きするだけで足ります。opencode は設定を起動時に一度だけ
読むので、トークン変更後は opencode の再起動が必要です。

なお、同じ値のユーザー環境変数 `KAGGLE_API_TOKEN` が別途設定されていて
ファイルより優先される状態だったため、重複とローテーション時の取り違えを避けるべく削除し、
`WM_SETTINGCHANGE` のブロードキャストまで行いました（2026-10-15）。Freebuff は
`.agents/mcp.json` が自前の `env` でトークンを渡しているので、この削除の影響を受けません。

この環境で確認したこと（2026-10-15）:

- initialize + tools/list → 51 ツール（トークン未設定でも一覧までは応答）。パッケージは
  kaggle_mcp_server 0.2.0 / mcp 1.30.0 / kagglesdk 0.1.37。
- `~/.kaggle/access_token` 経由・環境変数なしで認証が要る読み取りを実行: `kernels_list`
  （koeiro 検索 →「No kernels found.」= エラーなし）と `kernel_session_status`
  （`heynegix/koeiro-meanvc2-cuda-verification` と
  `heynegix/koeiro-meanvc2-captured-whole-file-conversion` がどちらも `COMPLETE`）。
  プライベートノートブックが読めるので、このトークンは有効です。
- `opencode debug config` が exit 0 で、`mcp.kaggle` を含む解決済み設定を出力。

注意: `kernel_push` の制限（`machine_shape` / `enable_internet` を指定できない）は opencode 側でも
同じで、回避手順も §6 の既存記録のままです。`uv tool upgrade kaggle-mcp-server` は `mcp` 2.x を
巻き込んで起動しなくなるため、`--with "mcp<2"` 付きで入れたままにしてください。
