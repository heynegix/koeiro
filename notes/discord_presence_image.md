# Discord プレゼンス画像の「？」表示について

## 症状

Discord の「プレイ中」カード（アクティビティ右側）の大きい画像と、通話ポップアウトの
「プレイ中」行の小さいアイコンが「？」になる。名前の横など別の場所のアイコンは正常。

## 原因（切り分け済み）

1. **画像の指定先がローカルのURLだった**
   `src/presence/activity.py` の `ICON_URL` が `http://localhost:9999/...` を指していた。
   Discord は外部URLの画像を **Discord 側のメディアプロキシ**が取得する仕様なので、
   手元のPCでしか解決できない `localhost` は取得できず「？」になる。
   （`https://files.catbox.moe/...` を試していた時期もある。ホスト側の削除・期限切れや、
   プロキシのキャッシュに残った失敗も「？」として出る）

2. **`.ico` はDiscordが描画しない形式**
   Discord のドキュメント（Activity Asset Image）では、アップロード済みアセットは
   PNG / JPEG / WebP、外部URLは「publicly accessible image（GIF・animated WebP・AVIF を含む）」。
   `assets/icon.ico` は `image/vnd.microsoft.icon` で返るため、カード画像には使えない。

3. **大きい画像と小さい画像は別スロット**
   `large_image` だけ指定しても、カード右下と通話ポップアウト行が使う `small_image` は
   空のままなので、そちらだけ「？」が残る。

## 現在の実装

`src/presence/activity.py` が画像の指定を1か所（`ICON_IMAGE`）に集約し、
`large_image` と `small_image` の**両方**に同じ画像を入れています。

- 既定: 公開済みの `assets/icon.png` を指す外部URL
  （実機確認で Discord が `mp:external/...` に書き換えて返す＝プロキシが取得できている）
- 上書き: 環境変数 `KOEIRO_ICON`（デプロイ済みビルドをポータルのキー名へ切り替える用途）
- 指定が空文字なら `assets` を送らない（描けない約束をしない）

### ポータルのアップロードアセット方式（より安定）

Discord デベロッパーポータル → 対象アプリ → Rich Presence → Art Assets に画像を上げ、
その「キー名」を `ICON_IMAGE` に書きます。外部URLと違い Discord 自身が解決するため、
プロキシの取得失敗やキャッシュに左右されません。

## 確認手順

1. `src/presence/activity.py` の `ICON_IMAGE` を確認する
2. `.venv/Scripts/python -m pytest tests/test_presence.py`（画像スロットのテスト）
3. 実機: `.venv/Scripts/python tools/verify_discord_presence.py --hold 30`
   → Discord が返す `assets` が `mp:external/...`（またはアセットキー）になっているかを見る
4. 表示は目視で確認する。**送信が受理されたことと、アイコンが描画されることは別**
   （受理はスクリプトで分かるが、描画の最終確認は人の目でしかできない）

## 参考

- [src/presence/activity.py](../src/presence/activity.py)
- [src/presence/client.py](../src/presence/client.py)
- [tests/test_presence.py](../tests/test_presence.py)（activity 生成のテスト）
- Discord ドキュメント: RPC の SET_ACTIVITY と Activity Assets / Activity Asset Image
