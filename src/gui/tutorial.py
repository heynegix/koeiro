"""First-run and on-demand tutorial guide. Skippable, remembers its dismissal.

Each page names the window page it belongs to and the control it is talking about, so
the guide walks the real application: the main window switches to that page and
highlights that control while the page is shown. Skipping only closes the guide (it
opens again on the next launch) unless "次回から表示しない" is ticked, and the
"？ 使い方ガイド" button (bottom right, and the ヘルプ entry) reopens it at any time.

The six steps follow the decisions a session needs, in order: where the voice comes
in, whose voice it becomes, whether to wait, and when to start. Each step says why
the decision matters, how to make it, and what is gained.
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
         body="このアプリは、マイクの声を選んだ声に変えて出力するボイスチェンジャーです。\n\n"
              "決めることは3つだけ：① どのマイクを使うか → ② 誰の声にするか・待つか → "
              "③ Startを押す。\n"
              "このガイドは「スキップ」でいつでも閉じられます。右下の"
              "「？ 使い方ガイド」から何度でも開けます。"),
    dict(title="① どのマイクを使うか",
         page="home", target="input_device",
         body="あなたの声の入り口を決めます。マイク入力で使うマイクを選んでください。\n\n"
              "・入力と出力先は同じ方式にそろえる（Windowsは両方 [Windows WASAPI] を推奨）\n"
              "・Discordに流すときは、右の声パネルの下の出力デバイスで仮想ケーブル（CABLE Input）を選ぶ\n"
              "・Linuxで通話アプリへ流すときは `sudo modprobe snd-aloop` で Loopback を有効化（詳細はREADME）\n"
              "・デバイスを増やしたら「デバイス一覧を更新」を押す\n"
              "・得るもの：中央の入力レベルが緑〜黄（-20dB前後）に振れれば準備OK"),
    dict(title="② 誰の声にするか",
         page="home", target="voice_panel",
         body="右の声パネルで、なりたい声を選びます。探し方は3つあります。\n\n"
              "・検索欄に名前を打つ／「すべて・標準ボイス・追加した声」で絞る\n"
              "・カードの「選択」を押す（最初は標準ボイスのままで大丈夫です）\n"
              "・自分の声に変えたいときは「＋ 声を追加」から登録\n"
              "・得るもの：②の「変換したい声」に選んだ声が出る"),
    dict(title="② 待つか待たないか",
         page="home", target="route_box",
         body="②の声の下の3つのボタンで、話し方（待つかどうか）を決めます。\n\n"
              "・逐次変換：話しながら順に処理。待たないが帯域復元なし\n"
              "・一括変換：話し終えてから変換し、帯域を復元。待つが高音質\n"
              "・一括変換-最速：待ちを約0.6秒に縮めた速い版\n"
              "・得るもの：遅延と音質のどちらを取るか、自分で選べる"),
    dict(title="③ Start して話す",
         page="home", target="start_button",
         body="中央下の Start を押すと変換が始まり、Stop で止まります。\n\n"
              "・中央のオーブをクリックしても開始・停止できます\n"
              "・一括変換では「話終わるまで待ってから出力」と表示されます\n"
              "・待たずに確定したいときは「今の発話を変換」\n"
              "・得るもの：選んだ声が、選んだ出力先に届く"),
    dict(title="困ったときは",
         page="settings", target="monitor_toggle",
         body="うまくいかないときの順番：耳で確かめる → 推奨に戻す → 聞き直す。\n\n"
              "・変換後を自分の耳で確認：設定の Monitor でヘッドホンを選び、"
              "「このデバイスでテスト音を鳴らす」で確認できます\n"
              "・声が荒い・ガラガラする：入力レベル、マイクとの距離を見直す\n"
              "・遅延・音切れ：Bufferを大きくし、他のアプリの負荷を下げる\n"
              "・どこを触ればよいか分からない：「推奨設定を適用」を押す\n"
              "・このガイドは右下の「？ 使い方ガイド」からいつでも開けます"),
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
