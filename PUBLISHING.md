# PUBLISHING — 配布前の整備手順と同梱範囲

## 同梱するもの（allowlist）

- `app.py`、`src/`、`tests/`、`tools/install_*.py`
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
  "deliveries": ["streaming", "utterance_lavasr"]
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
