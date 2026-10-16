# PUBLISHING — 配布前の整備手順と同梱範囲

## 同梱するもの（allowlist）

- `app.py`、`src/`、`tests/`、`tools/install_*.py`
- アプリ名は **Koeiro**（旧名 Anime Voice Changer）。凍結ビルドの worker 実行ファイルは
  `KoeiroWorker.exe` の名前で作る（`src/runtime_paths.py` の `WORKER_EXE` と一致必須）
- `requirements.txt`（GUI）、`vc_models/meanvc2/requirements.lock.txt`（ワーカー）
- `models/meanvc2_*/runtime.json`（設定のみ。音声・埋め込みは含まない）
- 同梱する声：`models/user_voices/<default-voice>/` の `reference.wav`・
  `fixed_embedding.npy`・`runtime.json`・サニタイズ済み `profile.json` のみ
- 非音声イベントゲートのモデル：`models/nonvoice/` の `silero_vad.onnx`（Silero VAD, MIT）・
  `yamnet.onnx`・`yamnet_class_map.csv`（audiomagic/yamnet-onnx, Apache-2.0）。
  凍結前に `tools/setup_release.py --nonvoice-models` で sha256 固定のまま取得し、
  凍結ビルドに埋め込む（モデルが無い環境ではゲートが無効になるだけで、変換は止まらない）
- `LICENSE`、`THIRD_PARTY_NOTICES.md`、`README.md`、このファイル

## 同梱しないもの

- `vc_models/` のうち MeanVC2・LavaSR・runner 以外（研究用モデル約70GB）、
  `vc_models/cache/`、開発用 venv（`.venv-ai`・`.venv-prosody` 等）
- `vc_models/post_lavasr/vendor/encodec*` と `vendor/bin/encodec.exe`（wheel の
  メタデータが CC BY-NC 4.0 表示。アプリは vendored `vocos` だけを使い
  `encodec` を import しないため、同梱しない）
- `models/user_voices/` の既定声以外、`models/` の研究用・ASR取得物（利用者が
  `tools/install_asr.py` で取得）、Beatrice系モデル一式
- `recordings/`、`logs/`、`validation/`、`test-results/`、`checkpoints/`、
  `.build/`、`settings.json`、実行時キャッシュ、`__pycache__` 類
- 上記は `.gitignore` で除外済み。フォルダ配布時はこの一覧で選別すること

## bundle.json（凍結ビルド用マーカー）

凍結バンドルでは `models/bundle.json` で公開プロファイルと処理方法を制限する。
開発ツリーには置かないこと（置くと手元の声が隠れる）。
リリースアセット（ソースツリー方式）には含めない。

```json
{
  "profiles": ["user_<default-voice-id>"],
  "deliveries": ["streaming", "utterance_lavasr", "utterance_fastest"]
}
```

## 公開前チェックリスト

- [ ] `LICENSE`・`THIRD_PARTY_NOTICES.md` を同梱した
- [ ] `README.md` の「使用モデルとライセンス」表と `THIRD_PARTY_NOTICES.md` を
      実際の同梱物に合わせて更新した（MeanVC2＝Apache-2.0、LavaSR＝Apache-2.0、
      WavLM-Large/ECAPA 前処理チェックポイントの帰属表示）
- [ ] Windows ビルドの PySide6(Qt) LGPLv3 義務を満たした（ライセンス文の同梱、
      Qt の入手先の明記、再リンク可能な動的リンクの維持）
- [ ] 既定声以外の `models/user_voices/*` が配布物に無い
- [ ] 凍結ビルドの前に `tools/setup_release.py --nonvoice-models` を通し、非音声ゲートの
      モデル3点（`models/nonvoice/`）を同梱した
- [ ] `settings.json`・`logs/`・`recordings/` が配布物に無い
- [ ] Beatrice/JVS/GPL/CC-BY-NC系の資産が配布物に無い
- [ ] ソース公開の場合：Git履歴に音声・重み・画面画像・絶対パス・秘密情報が
      残っていない（`git rev-list --objects --all` で wav/mp3/m4a/pt/pth/
      safetensors/onnx/bin/env を検索。残っていれば history rewrite が必要）
- [ ] 配布先（BOOTH・GitHub等）の規約・年齢表示・サポート窓口を記載した
- [ ] タグに `notes/release_notes_<tag>.md`（リリース本文用）を入れた。BOOTH ZIP は手元で
      `tools/build_booth_zip.py` で作った（CI に作らせるときだけ `notes/guide-<tag>-*.txt` も
      タグに入れ、workflow_dispatch の booth を有効にする）
- [ ] Windows は setup.exe（＋AIパックを出すなら its parts）を作り、`-checksums.txt` を
      一緒に添付した（実行ファイル版の自動更新は両方が無い版を適用しない）
- [ ] 添付前に `tools/verify_release_assets.py --tag <tag>` を通した
      （`release` ワークフローは添付前に自動で通し、落ちれば添付しない）
- [ ] AIパックに `vendor/encodec` の実体が入っていないこと
      （`tools/build_ai_pack.py` の `check_stage` が検査する）
- [ ] **版数を上げた後は凍結ビルドを作り直した**（`src/version.py` はバンドルの中に入る。
      名前だけ新版の tar を添付すると、中身は旧版と答えて自分自身へ更新をかけようとする）
- [ ] Linux の実行ファイル版を出すなら `dist/linux/` の3点（app tar・パック2分割・
      checksums）を添付した。パックの `KoeiroWorker` に実行ビットが付いていることも
      確認する（Windows 上で tar すると NTFS がモードを持たないので、
      `tools/build_linux_bundles.py` の `executable_member` が名前で 0755 を戻す）
- [ ] 配布物相当のツリーで `tools/setup_release.py --dry-run` が通り、
  Windows と Linux の両方でセットアップ後に起動・変換・停止を確認した
- [ ] フォルダ配布時は `settings.json`・`logs/` を取り除いた（起動時に再生成される）
- [ ] uv のキャッシュを消した環境では各 venv の `pyvenv.cfg` の `home` が
      実在の Python を指している（指していないとワーカーが起動しない）

## GitHubリリース配布（Windows / Linux）

通常の配布はソースビルドではなく GitHub Releases です。アプリ内蔵の自動更新は
この形式を前提にしています（`src/app_update.py`）。

- リポジトリ：`heynegix/koeiro`
- タグ：`vX.Y.Z`（必要なら `-preview.N`）。アプリの表示・比較元は
  `src/version.py` の `APP_VERSION` です。**タグを打つ前にここを新タグに
  更新すること**（更新忘れは自動更新の無限通知になります）
- 資産内容：追跡ツリーそのもの＋直下の `release.json`（`tools/build_release_asset.py`
  が生成。初回起動時に `tools/setup_release.py` が各OS用の仮想環境とモデルを揃えます）、
  Windows の setup.exe と AIパック、Linux の凍結バンドルと AIパック。すべて
  `release` ワークフローがタグ push で作って添付します
- アセット名（OS判定・更新可否はこの名前で行います。変えないこと）：
  - Windows：`koeiro-<tag>-windows.zip`
  - Linux：`koeiro-<tag>-linux.tar.gz`
- `release.json` の例：`{"app": "Koeiro", "version": "0.11.0",
  "platform": "windows", "tag": "v0.11.0"}`（`version` は `v` なし）
- torch同梱の凍結バイナリは作りません：ワーカー環境だけで2GB超のため、
  単一アセットの2GB制限に収まらないからです（実測はPUBLISHINGの履歴ではなく
  `tools/setup_release.py --dry-run` と各環境で確認）。完全凍結は将来の選択肢で、
  コード側の分岐（`is_frozen`・`KoeiroWorker.exe`）は維持します

```sh
# 例：v1.0.0 を出す
# 1. src/version.py の APP_VERSION を 1.0.0 に更新してコミット
# 2. リリース本文も書いてコミット（無いと本文は空の下書きになる）
#    notes/release_notes_v1.0.0.md
# 3. タグを打って push。あとは release ワークフローが一式を作り、検査してから添付する
git tag v1.0.0; git push origin v1.0.0
# 4. 手元確認だけなら Actions → release → Run workflow（tag を入れて dry_run のまま。
#    リリースには触れず、Artifacts に全部残る）
# 5. BOOTH に上げる ZIP は手元で作る（BOOTH 節）
```

`release` ワークフローのジョブは prepare（タグ確定・版数一致）、selftest、source-assets、
windows-setup、windows-ai-pack、linux-app、linux-pack、release の8つ。BOOTH ZIP を作る booth
ジョブは既定では動かず、workflow_dispatch で booth を有効にしたときだけ加わります。
添付するのは booth 以外の成果物（リリース本文は `notes/release_notes_<tag>.md` があればそれを
使い、無ければ空の下書き）で、添付の前に `tools/verify_release_assets.py` が一式の有無・
sha256・更新の選別を通せなければ止まります。

更新時は仮想環境（`.venv`・`vc_models/meanvc2/.venv`・`vc_models/post_lavasr`）を
引き継ぐため、再ダウンロードは発生しません。依存関係の固定が変わった版では、
更新後に `tools/setup_release.py` を再実行してください。

## Windows インストーラ（setup.exe）と実行ファイル版の自動更新

実行ファイル版（`Koeiro.exe`）の更新は **setup.exe の再適用**で行います。凍結
バンドル内をファイル差し替えする方式は採用できません：起動中の `Koeiro.exe` と
`_internal\*.dll` は Windows にロックされ、自分自身を含むフォルダを退避できないためです。
インストーラはインストール先の外からファイルを差し替えるので、この問題がありません。

```powershell
# タグ push なら release ワークフローの windows-setup ジョブが同じ手順を回す。
# 手元で作り直すときは（PyInstaller と Inno Setup が必要）:
.\.venv\Scripts\python tools\build_windows_setup.py --tag v1.0.0 --rebuild
# 手元で作った場合だけ添付する:
gh release upload v1.0.0 dist\setup\koeiro-v1.0.0-windows-setup.exe `
                           dist\setup\koeiro-v1.0.0-windows-checksums.txt --clobber
```

- CI（`windows-setup` ジョブ）は windows-latest に GUI 用 `.venv` を作って `--rebuild` を回し、
  `ISCC.exe` がランナーに無ければ `choco install innosetup` で入れます

- アセット名（変えないこと）：`koeiro-<tag>-windows-setup.exe` と
  `koeiro-<tag>-windows-checksums.txt`（`<sha256>  <ファイル名>` の sha256sum 形式）
- 実行ファイル版のアプリは **setup.exe と checksums.txt の両方がある版にしか更新しません**。
  checksums が無い・合わない場合は「自動更新できません」と出して止まります（検証できない
  exe をサイレントで走らせないため）。ダウンロードは `%LOCALAPPDATA%\Koeiro\updates\`
- インストール先は既定で `%LOCALAPPDATA%\Programs\Koeiro`（ユーザー単位・UACなし）。
  Program Files にしないのは、毎回同意を要求するとサイレント更新が成立しないためです
- `AppId` は `installer\Koeiro.iss` に固定値で入れてあります。**変えると別アプリ扱いに
  なって併存インストールになる**ので、版が上がってもそのままにします
- 同梱物の検査：`tools/build_windows_setup.py` の `check_stage` が `koeiro-root.txt`
  （ビルド元の絶対パス）・`settings.json`・`logs`・`recordings` を弾きます。
  `koeiro-root.txt` を除くと `runtime_paths.install_root()` は **インストール先フォルダ**を
  指すようになるので、AI環境は `%LOCALAPPDATA%\Programs\Koeiro\vc_models` のように
  exe の隣に置く形になります（`KOEIRO_ROOT` で上書き可）

## BOOTH 向け配布

BOOTH の上限は **1ファイル 1.2GB・1商品あたり合計 10GB**（2026-03-09 時点。
`booth.pixiv.help` のアップロード容量制限）。torch 環境（実測 2.2GB）とモデル
（`vc_models/meanvc2/repo/preprocess/ckpts` だけで 1.8GB 超）を1ファイルに収めることは
できないので、**OSごとに1 ZIP** にまとめます（v0.13.4 実測）。

| ZIP | 中身 | 大きさ |
|---|---|---|
| `Koeiro_<tag>_Windows.zip` | setup.exe・`-windows-checksums.txt`・ai-pack.exe・ai-pack-1.bin・ai-pack-checksums・入手案内 | 748.2 MB |
| `Koeiro_<tag>_Linux.zip` | app tar・ai-pack 2分割・`-linux-checksums.txt`・入手案内 | 931.0 MB |

```powershell
# dist/setup・dist/ai-pack・dist/linux が揃った状態で（入手案内.txt は実測範囲を書いた自前のもの）
.venv\Scripts\python tools\build_booth_zip.py --tag v1.0.0 `
  --guide-windows notes\guide-v1.0.0-windows.txt --guide-linux notes\guide-v1.0.0-linux.txt
# 出来た ZIP の点検だけ（中身の名校験・フラット・stored・破損検査）
.venv\Scripts\python tools\build_booth_zip.py --tag v1.0.0 --check
```

- 既定ではこの手順を手元で回します（`release` ワークフローの booth ジョブは
  workflow_dispatch で明示的に有効にしたときだけ動き、成果物は Artifacts に出ます）
- このツールが検査するもの：同梱の checksums と各メンバーの sha256、1ファイル 1.2GB・
  1商品 10GB の上限、ルートのフラットさ、メンバーの重複。v0.13.4 の2枚は
  `--check` を通ります（2026-10-14 にこの手順で作り直して確認）
- v0.13.4 の実測: `Koeiro_v0.13.4_Windows.zip` **748,246,799 バイト**（6 メンバー）、
  `Koeiro_v0.13.4_Linux.zip` **931,060,549 バイト**（5 メンバー）。どちらも最大メンバーは
  `ai-pack-1`（`.bin` 696,688,088 / `.tar.gz` 451,326,384 バイト）で、1ファイル上限の内側

- どちらも上限の内側（最大の中身は ai-pack-1.bin 696.7 MB）。2ファイルを1商品にしても、
  Windows用と Linux用で商品を分けても容量は足ります
- ZIP は **無圧縮（stored）**。中身がすでに gzip / Inno で圧縮済みなので、deflate は
  CPU を溶くだけでサイズは減りません
- **アーカイブのルートはフラット**にします。`ai-pack.exe` は `ai-pack-1.bin` を同じ
  フォルダで探しますし、`install-ai-pack.sh` は2つの tar と checksums を同じフォルダで
  検証して展開するためです（階層を1つ入れると買い手が「データが見つからない」になる）
- ZIP 自身の sha256 も `Koeiro_<tag>_booth-checksums.txt` に出すので、アップロード後の
  突き合わせ（転送で壊れていないか）に使えます
- 入手案内.txt は**実測した範囲のことだけ**書きます（声の登録が凍結ビルドではできない、
  Linux の実機音声デバイス確認をしていない、など）。書かないのは許せても、間違ったことを
  書くのは返品とトラブルになります

BOOTH 販売でもアプリ内蔵の自動更新は GitHub Releases を見にいきます（購入者に
ネットワークが必要。オフライン完結にしたい場合は setup を再配布する）。

## AI ボイスパック（凍結ワーカー）

`KoeiroWorker.exe` は torch をinternalに凍結した AI ワーカーで、AI環境（venv）も
Python も要りません。アプリ本体の setup とは別の installer にして、**同じアプリフォルダに
展開**します（`{app}\KoeiroWorker.exe` + `{app}\_internal_worker\` + `{app}\vc_models\` +
`{app}\models\user_voices\`）。

```powershell
# 1. ワーカーを凍結する（AI環境の Python でビルド。GUI用 .venv ではない）
uv pip install --python vc_models\meanvc2\.venv\Scripts\python.exe pyinstaller
vc_models\meanvc2\.venv\Scripts\python.exe -m PyInstaller KoeiroWorker.spec --noconfirm
#    → dist\KoeiroWorker\KoeiroWorker.exe + _internal_worker\（実測 約1.01GB・ビルド 約14分）

# 2. 検証（同じ WAV を venv ワーカーと凍結ワーカーで通して差分を見る）
.venv\Scripts\python tools\verify_frozen_worker.py --wav promo\public\audio\voice_demo_original.wav

# 3. AIパックの installer を作る（スライス 1.05GB = 1050000000 バイト）
.venv\Scripts\python tools\build_ai_pack.py --tag v0.13.2
#    → dist\ai-pack\koeiro-<tag>-ai-pack.exe / -1.bin … と -checksums.txt
```

実測（v0.13.2）: ステージング 6650ファイル 1385.6 MB → 圧縮後 `koeiro-v0.13.2-ai-pack.exe` 2.8 MB +
`koeiro-v0.13.2-ai-pack-1.bin` 664.4 MB（計2ファイル）。アプリ本体の setup 46.3 MB と合わせて
**BOOTH には3ファイル**（各1.2GB上限に十分収まる）。インストール先は 1.6 GB。

この3手順は `release` ワークフローの windows-ai-pack（Windows）と linux-pack（Linux）が
CI でもそのまま回します（手元で作った場合は `gh release upload` で添付）。

- **変換結果は開発環境とビット同一**: `voice_demo_original.wav` 3.44秒を 一括変換(LavaSR)
  で通すと、環境venvワーカー / `dist` の凍結ワーカー / インストール済みパック、3者の出力 WAV
  の sha256 がすべて一致（`1526b85b1eb60414`、max_abs_diff 0.0）。逐次変換（streaming）は
  この照合をしていない
- 同じ WAV を使う再確認: `tools\verify_frozen_worker.py`（デバイスを開かない・人力聴取なしの
  app↔worker RPC である旨をレポートに明記する）

- **中身の決め打ち**: 同梱する重みは `models/user_voices/<既定声>/runtime.json` が要求する
  3点（`fastu2pp_160ms.pt`・`meanvc2_120ms_40ms.safetensors`・`vocos.pt`）だけ。
  `tools/build_ai_pack.py` の `check_stage` が runtime.json と突き合わせて、欠けていれば弾く
- **同梱しないもの**: AI環境の venv（凍結で代用）、声の登録が使う
  `wavlm_large.pt` / `wavlm_large_finetune.pth`（合計 約2.4GB）、`vendor/bin`
- **`encodec` は同梱できない**: wheel のメタデータが **CC BY-NC 4.0**。しかし
  `vc_models/post_lavasr/vendor/vocos/feature_extractors.py` が module 冒頭で
  `from encodec import EncodecModel` するため、実行時に import されていた（importtime 実測）。
  この配布物では **自作のスタブ**（`tools/build_ai_pack.py` の `ENCODEC_STUB`）に置き換える。
  実物→スタブの差し替えで 一括変換(LavaSR) の出力 WAV が **ビット同一**（sha256 一致）であることを
  実測で確認済み。スタブは実際に使われたら `NotImplementedError` で音を出さずに落ちる
- 購入者への手順: 分割物を**すべて同じフォルダ**に置いて part1 を実行。part1 だけ渡しても壊れる
- **声の登録はこのパックでは動かない**: 登録は `tools/register_voice.py` を AI環境の
  Python で実行する作り（`src/gui/voice_registration.py`）。凍結版で登録まで可能にするのは
  次の検討事項（ワーカー exe に登録モードを持たせる案が最も安い）

### 凍結ワーカーで判明した制約（spec に書いてあるが、消されると再踏する）

- `console=True` が必須。`console=False` は stdout が使えず、プロトコルが死ぬ
  （GUI は `CREATE_NO_WINDOW` で起動するので窓は出ない）
- `x_transformers` / `einops` / `einx` / `ema_pytorch` の **.py ソースを同梱する**。
  TorchScript が `@torch.jit.script` の展開に `inspect.getsource` を使うため、PYZ だけだと
  `Can't get source for <function softclamp>` で読み込み失敗する
- `contents_directory='_internal_worker'` で GUI の `_internal` と衝突させない
- ワーカーは `KOEIRO_ROOT` / `KOEIRO_ASSETS` を GUI から受け取る（`src/vc/client.py`）。
  これがないと `asset_root()` が自分のバンドル内を見てモデルを見つけられない
- PyInstaller は 1回 約14分。hidden imports の当たりは `.build/` の importtime 実測から取った

## Linux 版（Docker か GitHub Actions で作る）

PyInstaller は「動かした OS の実行ファイルしか作れない」ので、Linux 用の凍結ビルドは
Linux で作ります。経路は2つで、出てくるものは同じです（`tools/build_linux_bundles.py`）。

- **手元（Windows + Docker Desktop）**: `--engine docker`。ビルドはコンテナ内の `/tmp` で
  行い、出来た tar だけを stdout に流して受け取ります（NTFS バインドマウントに 1GB
  書くのは遅いため）。Docker のディスクイメージは C: に載るので、成果物を増やすたびに
  手元の空きが削れる点だけ注意
- **Linux 上（CI ランナーや WSL）**: `--engine local`。同じスクリプトをそのまま hosts 上で
  実行します。apt をスクリプト内では走らせないで、ワークフロー側のステップで同名の一覧を
  入れてください（`RUNTIME_APT` が単一の出所）

```powershell
# アプリ本体（凍結 GUI）
.venv\Scripts\python tools\build_linux_bundles.py --tag v0.13.3 --kind app
# AI ワーカーを凍結して AIパック（2分割 tar.gz）まで
.venv\Scripts\python tools\build_linux_bundles.py --tag v0.13.3 --kind worker
```

```sh
# Linux 上で、自分で凍結したワーカーからパックだけ
vc_models/meanvc2/.venv/bin/python -m PyInstaller KoeiroWorker.spec --noconfirm
python tools/build_linux_bundles.py --tag v0.13.3 --kind pack --worker-dist dist/KoeiroWorker
```

GitHub Actions 版は `release` ワークフロー（タグ push か workflow_dispatch、既定 dry_run）に
統合してあり、`linux-app` ジョブが上記1・2番目、`linux-pack` ジョブが3番目に対応します。

- アセット名（変えないこと）:
  - `koeiro-<tag>-linux-app.tar.gz` … 凍結 GUI（`Koeiro` + `_internal`）
  - `koeiro-<tag>-linux-ai-pack-1.tar.gz` / `-2.tar.gz` … ワーカー本体とモデル（BOOTH の
    1ファイル上限に合わせるため2分割。`install-ai-pack.sh` が両方を展開する）
  - `koeiro-<tag>-linux-checksums.txt` … 上の sha256（`sha256sum -c` 形式）。
    **アプリバンドルを含む一式のときだけこの名前**（実行ファイル版の自動更新が読む名前）。
    パックだけのビルドは `koeiro-<tag>-linux-ai-pack-checksums.txt` に出す。CI では app と pack が
    別ジョブなのでどちらも自分の名前で書き、組み立ての段で
    `tools/build_linux_bundles.py --kind checksums` が3行の1枚に統合してから添付する
  - ※ ソースツリーの `koeiro-<tag>-linux.tar.gz` とは**別名**。更新確認は名前完全一致で
    選ぶので混同しません（`src/app_update.py: pick_asset`）
- 買い手側の手順: tar を展開して `./Koeiro/Koeiro`。Debian/Ubuntu 系ではシステムの
  **libportaudio2** と Qt の実行ライブラリ（`libgl1 libegl1 libglib2.0-0 libxkbcommon0
  libdbus-1-3 fontconfig`）を要求します（空の Debian コンテナで
  `QT_QPA_PLATFORM=offscreen Koeiro --smoke-test` を走らせてこの一覧を実測。
  デスクトップ用途のディストリビューションなら大半は既に入っています）
- AIパックには `install-ai-pack.sh` が同梱されます: 第1引数のフォルダ（`Koeiro` が有るところ）
  へ2つの tar を展開します。チェックサム1枚に3成果物ぶんを書くので、このスクリプトは
  `-linux-ai-pack-` の行だけ `sha256sum -c` に通します（買い手が全部を同じフォルダに置いた
  とき、同梱されていない app.tar の行で `set -e` がインストールを止めていた）
- **ビルド元記録 `koeiro-root.txt` はバンドルから削除**して出します（Windows の絶対パスが
  入ったままだと、インストール先が見つからず AI が動かなくなる）
- Linux 凍結版の自動更新: Windows のような再インストール程序が無いので、**新版バンドルを
  落として sha256 検証 → 解凍 → インストール先ディレクトリごと差し替え → 再起動**。
  実行中の exe を含むディレクトリを rename するのは Linux では合法なので成立します
  （`src/update_swap.py`、`preserve_dirs` で `vc_models`（AIパック）と `recordings` を残す）
- Windows 側と同じ検査を通ります: `tools/build_ai_pack.py` の `stage_pack` / `check_stage`
  を名前付き引数（`worker_exe="KoeiroWorker"`, `contents_dir="_internal_worker"`）で共用。
  encodec のスタブ差し替えと runtime.json の sha256 照合は Linux でも同じ
- **凍結ワーカーが Linux で動くことは実測済み**（空の `python:3.11-bookworm` に libportaudio2
  だけ入れて `tools/verify_frozen_worker.py --only frozen`、保存済み WAV を RPC で1回変換）。
  出力は Windows の venv ワーカーと同じ入力に対しサンプル数一致・最大差 18/32768
  （≒ -66 dBFS、平均差 2.9e-6）。プラットフォーム側の浮動小数の丸め差で、 packaging の
  ズレではありません。**試聴判定ではなく「同じ計算に乗っている」だけの検査**です
- v0.13.3 の Linux 配布物（CI 製）実測: app tar **101.7 MiB** / パック **405.6 + 360.8 MiB**
  （ステージング 6158ファイル 1673.3 MB）。BOOTH の1ファイル上限 1.2 GB に対して3点とも余裕。
  インストール済みは 1.9 GB（`du -sh` を bookworm コンテナで実測）
- v0.13.4 の Linux 配布物（Windows + Docker 経路で作成）実測: app tar **101,322,360 バイト**、
  パック **451,326,384 + 378,408,433 バイト**（ステージング 6,158 ファイル 1,673.3 MB）。
  bookworm コンテナで `sha256sum -c` → `install-ai-pack.sh` → `--smoke-test`（exit 0、
  バンドルは `v0.13.4` と報告）→ 保存 WAV 3.439 秒の変換まで通した（165,072 サンプル、
  帯域復元のフォールバックなし、レポートは `dist/linux-install-verify-0134/`）
- v0.13.3 の Linux 実績（表示は `tools` と同じ MiB 基準）: app tar **101.7 MB**、
  AIパック **406.0 MB + 360.8 MB**（ステージング 6,158ファイル 1,673.3 MB）。
  3点の sha256 は GitHub が計算した digest と一致することを確認済み
- **`src/vc` を変えたらワーカーを作り直す**。凍結ワーカーは自分の `src/vc` を中に取り込んで
  いるので、モデルの重りが同じでも実行ファイルの中身は古くなります。`dist/linux/*.tar.gz`
  の member 時刻と `git log -1 --format=%ci` を見較べれば、作り直しが要るか分かります
- 凍結 Linux ビルドの CI 初実行は v0.13.3（当時の `linux-assets`、**app 1m39s / pack 7m26s**、
  ubuntu-22.04）。通るまでに落とした原因は上の「判明していること」と、コミット履歴の
  `linux-assets` 系メッセージに1件ずつ書いてあります。統合後の `release` ワークフローでは
  linux-app / linux-pack の2ジョブとして動き、タグ push（または dry_run=false の手動実行）で
  リリースへ自動で上がります。CI の pack ジョブは `tools/setup_release.py` で環境ごと
  作り直すため、ワーカーの実行 Python が手元（3.11）とずれます

### 判明していること（Linux ビルド時に効く）

- **ビルドしたマシンの glibc が買い手の下限になる**。ubuntu-24.04 ランナーで凍結すると
  `libpython3.12.so.1.0: version GLIBC_2.38 not found` で Debian 12 が起動できませんでした
  （実測: bookworm コンテナで落、ビルド側のランナーでは通る）。ビルドは ubuntu-22.04、
  起動確認は一段古い Debian bookworm のコンテナで行います（`release` ワークフローの
  「Debian bookworm で起動確認」ステップ）
- `Koeiro.spec` は `sys.platform == 'win32'` のときだけ `female_dsp_x64.dll` と
  `assets/icon.ico` を入れます（Windows 専用資産）
- コンテナの stdout は成果物専用。apt や pip の出力が1行でも混ざると gzip のメンバヘッダが
  壊れて `not in gzip format` になります（`_script` はすべて `1>&2`、`container()` が
  先頭2バイト `1f 8b` を検査して弾きます）

## ブランド画像

- `assets/icon.png` / `assets/logo.png` が原本（コミット済み）。編集時はこの2点のみ
- 派生品は `tools/build_brand.py` で再生成：`assets/icon.ico`（Windowsタスクバー・
  将来の凍結exe埋め込み用 `--icon assets/icon.ico`）、`assets/brand.png`（README上部）、
  `assets/social.png`（GitHub Settings → Social preview 用、1280x640）
- リポジトリアバターはGitからは設定できないため、上記 social.png を手で登録してください
