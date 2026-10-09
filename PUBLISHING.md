# PUBLISHING — 配布前の整備手順と同梱範囲

## 同梱するもの（allowlist）

- `app.py`、`src/`、`tests/`、`tools/install_*.py`
- アプリ名は **Koeiro**（旧名 Anime Voice Changer）。凍結ビルドの worker 実行ファイルは
  `KoeiroWorker.exe` の名前で作る（`src/runtime_paths.py` の `WORKER_EXE` と一致必須）
- `requirements.txt`（GUI）、`vc_models/meanvc2/requirements.lock.txt`（ワーカー）
- `models/meanvc2_*/runtime.json`（設定のみ。音声・埋め込みは含まない）
- 同梱する声：`models/user_voices/<default-voice>/` の `reference.wav`・
  `fixed_embedding.npy`・`runtime.json`・サニタイズ済み `profile.json` のみ
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

## bundle.json（配布ビルド用マーカー）

凍結バンドルでは `models/bundle.json` で公開プロファイルと処理方法を制限する。
開発ツリーには置かないこと（置くと手元の声が隠れる）。

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
- [ ] `settings.json`・`logs/`・`recordings/` が配布物に無い
- [ ] Beatrice/JVS/GPL/CC-BY-NC系の資産が配布物に無い
- [ ] ソース公開の場合：Git履歴に音声・重み・画面画像・絶対パス・秘密情報が
      残っていない（`git rev-list --objects --all` で wav/mp3/m4a/pt/pth/
      safetensors/onnx/bin/env を検索。残っていれば history rewrite が必要）
- [ ] 配布先（BOOTH・GitHub等）の規約・年齢表示・サポート窓口を記載した
- [ ] Windows（PyInstaller one-dir＋Worker exe）とLinux（venv方式）の両方で
      起動・変換・停止を確認した
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
- アセット名（OS判定・更新可否はこの名前で行います。変えないこと）：
  - Windows：`koeiro-<tag>-windows.zip`
  - Linux：`koeiro-<tag>-linux.tar.gz`
- パッケージ直下に `release.json` を同梱：`{"app": "Koeiro", "version": "<tag>",
  "platform": "windows" | "linux"}`。更新の検証はこのファイルで行います
- Windows：PyInstaller one-dir。`Koeiro.exe` と `KoeiroWorker.exe` が
  パッケージ直下（`sys.executable` の親）に来る構成にし、展開先ごと
  差し替え可能なこと（`Program Files` 等の要管理者権限の場所は避ける）
- Linux：venv 同梱方式。パッケージ直下に `app.py` と `.venv/` を置き、
  `.venv/bin/python app.py` で起動できること。ビルドは利用者層に近い
  ディストリ（例：Ubuntu 22.04 相当）で行い、glibc 互換に注意
- 両 OS とも `models/bundle.json` で公開プロファイル・処理方法を制限する
  （上記の例を参照）

```sh
# 例：v1.0.0 を出す
# 1. src/version.py の APP_VERSION を 1.0.0 に更新してコミット
# 2. Windows / Linux の配布物を用意し、上記の名前にする
gh release create v1.0.0 koeiro-v1.0.0-windows.zip koeiro-v1.0.0-linux.tar.gz \
  --title v1.0.0 --notes "変更点…"
```

リリース前チェック（上記チェックリストに加えて）：

- [ ] `src/version.py` が新タグと一致している
- [ ] 両アセットの直下に `release.json` があり `version` がタグと一致する
- [ ] 旧版を入れた環境で「更新を確認」が新版を提示し、再起動後に起動すること
- [ ] 更新後も追加した声・`settings.json` が残っていること
