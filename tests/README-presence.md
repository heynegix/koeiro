# Discord プレゼンスの確認範囲

`tests/test_presence.py` は **Discord に接続しません**。activity の組み立ては純粋関数として、
IPC はソケットペア上の偽 Discord に対して検証します。実クライアントを使う確認は
`tools/verify_discord_presence.py` だけです。

## 何がどこまで分かるか

| 確認 | 何を証明するか | 何を証明しないか |
|---|---|---|
| `pytest tests/test_presence.py` | 送る内容・フレーミング・再接続・GUI配線 | Discord が実際に受理したか |
| `tools/verify_discord_presence.py` | Discord が SET_ACTIVITY を受理し、`assets` を解決したこと | **アイコンが描画されること** |
| 目視（Discord のプロフィール） | アイコンが実際に出ていること | — |

受理と描画は別です。`mp:external/...` が返っていても、クライアント側のキャッシュや
一時的な取得失敗で「？」のまま見えることがあります。最終確認は目視で行ってください。

## 実機での確認手順

1. Discord デスクトップアプリを起動しておく（`\\.\pipe\discord-ipc-0` ができる）
2. `.venv/Scripts/python tools/verify_discord_presence.py --hold 30`
   - `--hold` は Discord の返信を待つ秒数。30 秒あれば Discord を見に行けます
   - 接続を閉じるとアクティビティは消えます（何も残しません）
3. 出力の `reply:` にある `assets` を確認する
   - `mp:external/...` = 外部URLをプロキシが解決した
   - `koeiro_...` のような素のキー = ポータルのアップロードアセットを Discord が解決した
4. Discord のプロフィールを開いて、カードの大画像と通話ポップアウト行の両方を見る

Discord 側の制限で、アクティビティ更新は 15 秒に1回です。連続実行すると
`reply` が返らないことがあります（handshake は通ります）。

## アイコンの指定

- 指定は `src/presence/activity.py` の `ICON_IMAGE` 1か所
- 環境変数 `KOEIRO_ICON` で上書き可（ポータルのキー名を試すのに便利）
- PNG / JPEG / WebP を使うこと。`assets/icon.ico` は Discord が描画しません
- 詳しい切り分けは [../notes/discord_presence_image.md](../notes/discord_presence_image.md)

## DSH サンドボックス内でテストする場合

サンドボックスは名前付きパイプの再オープンを拒否するため、次の3件は
`PermissionError: [Errno 13]` で落ちます。**コードの不具合ではありません**（制限外では全て通ります）。

- `test_open_ipc_reports_a_busy_pipe_instead_of_blocking_on_it`
- `test_a_pipe_read_takes_what_is_buffered_and_no_more`
- 登録ワーカー系（`tests/test_voice_library.py` のパイプ付き子プロセス2件）
