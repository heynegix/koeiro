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
- [ ] `settings.json`・`logs/`・`recordings/` が配布物に無い
- [ ] Beatrice/JVS/GPL/CC-BY-NC系の資産が配布物に無い
- [ ] ソース公開の場合：Git履歴に音声・重み・画面画像・絶対パス・秘密情報が
      残っていない（`git rev-list --objects --all` で wav/mp3/m4a/pt/pth/
      safetensors/onnx/bin/env を検索。残っていれば history rewrite が必要）
- [ ] 配布先（BOOTH・GitHub等）の規約・年齢表示・サポート窓口を記載した
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
  が生成）。初回起動時に `tools/setup_release.py` が各OS用の仮想環境とモデルを揃えます
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
# 2. タグを打って push（アセット作成・添付は Actions が自動で行う）
git tag v1.0.0; git push origin v1.0.0
# 3. 手動確認だけなら Actions → release-assets → Run workflow（dry_run のまま）
```

更新時は仮想環境（`.venv`・`vc_models/meanvc2/.venv`・`vc_models/post_lavasr`）を
引き継ぐため、再ダウンロードは発生しません。依存関係の固定が変わった版では、
更新後に `tools/setup_release.py` を再実行してください。
