"""First-run and on-demand tutorial guide. Skippable, remembers its dismissal.

Each page names the window page it belongs to and the control it is talking about, so
the guide walks the real application: the main window switches to that page and
highlights that control while the page is shown. Skipping only closes the guide (it
opens again on the next launch) unless "次回から表示しない" is ticked, and the
"？ 使い方ガイド" button (bottom right, and the ヘルプ entry) reopens it at any time.

The six steps follow the decisions a session needs, in order: where the voice comes
in, whose voice it becomes, whether to wait, and when to start. Each step keeps to
plain sentences about this app only.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel,
                               QPushButton, QStackedWidget, QVBoxLayout, QWidget)

# `page` is a Sidebar key, `target` the MainWindow attribute to highlight.
# `voice_panel` is the persistent right column: visible on every page, so it is
# highlighted without switching pages.
PAGES = (
    dict(title="ようこそ",
         page="home", target=None,
         body="マイクの声を、選んだ声に変えてそのまま出力します。相手に届くのは変換後の声だけです。\n\n"
              "最初に3つの操作を済ませてください。マイクを選ぶ。声と待ち方を選ぶ。Startを押す。\n"
              "この説明は閉じてかまいません。開き直すときは右下の"
              "「？ 使い方ガイド」を押してください。"),
    dict(title="① どのマイクを使うか",
         page="home", target="input_device",
         body="声の入り口を選びます。ふだん使っているマイクを選んでください。\n\n"
              "入力と出力先は同じ方式にそろえます。Windowsなら両方とも [Windows WASAPI] です。\n"
              "Discordに流すときは、右の声パネルの下にある出力デバイスで仮想ケーブル（CABLE Input）を選びます。\n"
              "Linuxで通話アプリへ流すときは `sudo modprobe snd-aloop` で Loopback を有効化します。詳しくはREADMEです。\n"
              "マイクを足したら「デバイス一覧を更新」を押してください。\n"
              "中央の入力レベルが -20dB 前後で振れていれば準備OKです。"),
    dict(title="② 誰の声にするか",
         page="home", target="voice_panel",
         body="右の声パネルで、なりたい声を選びます。\n\n"
              "名前が分かっていれば検索欄に打ってください。分からなければ「すべて」から眺めて、"
              "気になったカードの「選択」を押します。最初は標準ボイスでかまいません。\n"
              "自分の声を使いたいときは「＋ 声を追加」から登録します。\n"
              "選んだ声は②の「変換したい声」に出ます。"),
    dict(title="② 待つか待たないか",
         page="home", target="route_box",
         body="声の下の3つのボタンで、待つかどうかを決めます。\n\n"
              "逐次変換は話しながら変換します。待たない代わりに帯域復元はありません。\n"
              "一括変換は話し終えてから変換し、帯域を復元します。待つ代わりに高音質です。\n"
              "一括変換-最速は待ちを約0.6秒に縮めた版です。\n"
              "迷ったら一括変換のまま Start を押してください。"),
    dict(title="③ Start して話す",
         page="home", target="start_button",
         body="中央下の Start を押すと変換が始まります。止めるときは Stop です。\n\n"
              "中央のオーブをクリックしても同じ操作ができます。\n"
              "一括変換では「話終わるまで待ってから出力」と表示されます。"
              "途中で区切りたいときは「今の発話を変換」を押してください。\n"
              "選んだ声が、選んだ出力先に届きます。"),
    dict(title="困ったときは",
         page="settings", target="monitor_toggle",
         body="うまくいかないときは順に確かめてください。\n\n"
              "まず設定の Monitor でヘッドホンを選び、「このデバイスでテスト音を鳴らす」を押します。"
              "音が出れば出力側は正常です。\n"
              "声が荒いときは、入力レベルとマイクとの距離を見直してください。\n"
              "音が切れるときは Buffer を大きくし、他のアプリの負荷を下げてください。\n"
              "どこを触ればよいか分からないときは「推奨設定を適用」を押してください。\n"
              "このガイドは右下の「？ 使い方ガイド」からいつでも開けます。"),
)


class TutorialDialog(QDialog):
    """Six swipe-free pages with Back / Next / Skip and a hide-next-time box."""

    def __init__(self, parent=None, on_page=None):
        super().__init__(parent)
        self.on_page = on_page
        self.setWindowTitle("使い方ガイド")
        self.setMinimumSize(520, 380)
        layout = QVBoxLayout(self)
        self.stack = QStackedWidget()
        for spec in PAGES:
            page = QWidget()
            inner = QVBoxLayout(page)
            heading = QLabel(spec["title"])
            heading.setObjectName("sectionTitle")
            text = QLabel(spec["body"])
            text.setWordWrap(True)
            text.setTextFormat(Qt.TextFormat.PlainText)
            text.setAlignment(Qt.AlignmentFlag.AlignTop)
            inner.addWidget(heading)
            inner.addWidget(text, 1)
            self.stack.addWidget(page)
        layout.addWidget(self.stack, 1)
        self.position = QLabel()
        self.position.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.position.setObjectName("muted")
        layout.addWidget(self.position)
        hint = QLabel("このガイドを閉じたあと、実際に操作してみてください。")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.back_button = QPushButton("戻る")
        self.skip_button = QPushButton("スキップ")
        self.skip_button.setToolTip("ガイドを閉じます。次回の起動でもう一度表示されます。")
        self.next_button = QPushButton("次へ")
        self.hide_box = QCheckBox("次回から表示しない")
        row.addWidget(self.back_button)
        row.addWidget(self.skip_button)
        row.addStretch(1)
        row.addWidget(self.hide_box)
        row.addWidget(self.next_button)
        layout.addLayout(row)
        self.back_button.clicked.connect(self._back)
        self.skip_button.clicked.connect(self.reject)
        self.next_button.clicked.connect(self._next)
        self._refresh()

    @property
    def page(self):
        return self.stack.currentIndex()

    @property
    def hide_next_time(self):
        return self.hide_box.isChecked()

    def _refresh(self):
        last = self.stack.count() - 1
        index = self.stack.currentIndex()
        self.position.setText(f"{index + 1} / {self.stack.count()}")
        self.back_button.setEnabled(index > 0)
        self.next_button.setText("はじめる" if index == last else "次へ")
        spec = PAGES[index] if 0 <= index < len(PAGES) else None
        if spec is not None and callable(self.on_page):
            self.on_page(spec)

    def _back(self):
        self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1))
        self._refresh()

    def _next(self):
        if self.stack.currentIndex() >= self.stack.count() - 1:
            self.accept()
            return
        self.stack.setCurrentIndex(self.stack.currentIndex() + 1)
        self._refresh()

    def skip(self):
        """Headless/test helper with the same semantics as the Skip button."""
        self.reject()
