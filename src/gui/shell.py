"""Sidebar navigation, the centre orb and the voice panel.

The window is a fixed three-column layout: navigation on the left, the live
stage in the centre, and the voice list on the right. Every widget the rest of
the application already drives is still built by MainWindow; this module only
owns the shell, so the voice logic and its tests stay where they are.
"""
from math import cos, pi, sin

from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QLinearGradient, QPen, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

# What each page is for, in the reader's words rather than the code's. Home is the
# session flow (devices, voice, Start); settings holds everything needed only when
# something is wrong or needs fine control.
PAGE_PURPOSE = {
    'home': '① マイク → ② 声と話し方 → ③ Start。この順で進めれば変換が始まります。',
    'settings': 'デバイス・音量・モニターと、声の微調整・診断です。普段は開く必要はありません。',
}

# Dark navy shell: the swatch navy is the page, lifted navy cards carry the
# content, near-white text, one cyan accent. Neutrals do the structure work;
# cyan marks live conversion, keyboard focus and the running state only.
# Text gray passes AA on the page (6.47), on cards (5.22) and on pills (4.71).
BG = '#0B1020'
CARD = '#1B2440'
BORDER = '#2E3A5C'
BORDER_DARK = '#3A4666'
TEXT = '#F2F4FA'
TEXT_MUTED = '#8E97AC'
PRIMARY = '#F2F4FA'
PRIMARY_TEXT = '#0B1020'
LIVE = '#35E0FF'
PILL = '#232C47'
TRACK = '#263052'
DANGER = '#B91C1C'
WARN = '#FFCF7A'


class LevelBar(QWidget):
    """A 20-bar level meter drawn in-process; no audio work happens on this thread."""

    def __init__(self, parent=None, bars=20, vertical=False):
        super().__init__(parent)
        self.bars = bars
        self.vertical = vertical
        self._levels = [0.0]*bars
        self._peak = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(45)
        self._timer.timeout.connect(self._decay)
        self._timer.start()
        # A compact strip: enough to read a level at a glance, not a wall of bars.
        self.setFixedHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_level(self, value):
        """Feed a 0..1 amplitude.

        The strip shows the newest peak across its width rather than a per-bar history:
        the GUI thread samples at 30 Hz, so a fake per-bar walk would look like signal
        detail that was never measured.
        """
        level = max(0.0, min(1.0, float(value)))
        self._peak = max(self._peak, level)
        for index in range(len(self._levels)):
            self._levels[index] = level*(1.0-0.04*index)

    def set_levels(self, values):
        self._levels = [max(0.0, min(1.0, float(item))) for item in values][:self.bars]
        self.update()

    def _decay(self):
        self._peak *= 0.86
        if self._peak < 0.002:
            self._peak = 0.0
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect()
        painter.fillRect(rect, QColor(BG))
        count = len(self._levels) or 1
        gap = 2
        span = (rect.height()-gap*(count-1))/count
        bar_height = max(1, int(span))
        for index, level in enumerate(self._levels):
            y = int(index*(span+gap))
            filled = int(max(0.0, min(1.0, level))*rect.width())
            painter.fillRect(0, y, rect.width(), bar_height, QColor(TRACK))
            if filled:
                painter.fillRect(0, y, filled, bar_height, QColor('#FFFFFF'))
        # The newest peak is drawn across the strip so a single reading is visible even
        # though the per-bar history is not sampled at audio rate.
        peak_width = int(max(0.0, min(1.0, self._peak))*rect.width())
        if peak_width > 0:
            painter.fillRect(rect.width()-peak_width, 0, peak_width, rect.height(),
                             QColor(LIVE))


class MicOrb(QWidget):
    """The centre status disc: level ring, state text and a live waveform."""

    tapped = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self._state = 'Stopped'
        self._subtitle = 'マイク入力待ち'
        self._active = False
        self._phase = 0.0
        self._waveform = [0.0]*48
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._advance)
        self._timer.start()
        self.setMinimumSize(260, 260)
        # Expanding horizontally, fixed vertically: the disc keeps its natural size and
        # the labels have room above and below it.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(330)

    def set_state(self, state, subtitle=None, active=None):
        self._state = str(state)
        if subtitle is not None:
            self._subtitle = str(subtitle)
        if active is not None:
            self._active = bool(active)
        self.update()

    def set_level(self, value):
        self._level = max(0.0, min(1.0, float(value)))

    def _advance(self):
        if self._active:
            self._phase += 0.12
            # A synthetic sweep only when there is no real meter reading, so the orb
            # never invents signal the audio thread did not report.
            if self._level <= 0.001:
                index = int(self._phase) % len(self._waveform)
                self._waveform[index] = 0.25 + 0.2*((self._phase % 2.0) - 1.0)**2
        else:
            self._waveform = [value*0.90 for value in self._waveform]
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        centre = self.rect().center()
        # Cap the disc so a tall centre column does not leave the orb floating in empty
        # space, while a short window still shrinks it.
        radius = int(min(self.width(), self.height())*0.30)
        radius = max(88, min(radius, 150))

        glow = QLinearGradient(centre.x()-radius, centre.y()-radius,
                               centre.x()+radius, centre.y()+radius)
        glow.setColorAt(0.0, QColor('#1E2947'))
        glow.setColorAt(1.0, QColor('#121A33'))
        painter.setBrush(glow)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(centre, radius, radius)

        ring_width = 4
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor('#2A3454'), ring_width))
        painter.drawEllipse(centre, radius, radius)
        # The ring arc is the actual level, so the visual cannot claim signal that the
        # audio thread has not measured.
        sweep = int(-90 + 360*self._level)
        if sweep > 0:
            painter.setPen(QPen(QColor(LIVE), ring_width, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap))
            painter.drawArc(QRect(centre.x()-radius, centre.y()-radius, radius*2, radius*2),
                            int(-90*16), sweep*16)

        # Waveform bars radiating from the orb.
        painter.setPen(Qt.PenStyle.NoPen)
        for index, value in enumerate(self._waveform):
            angle = 2.0*pi*index/len(self._waveform)
            inner = radius + 14
            height = 6 + int(abs(value)*26)
            x = centre.x() + int((inner+height*0.5)*cos(angle))
            y = centre.y() + int((inner+height*0.5)*sin(angle))
            painter.setBrush(QColor(LIVE if self._active else '#3A4666'))
            painter.drawRoundedRect(x-2, y-2, 4, 4, 2, 2)

        # Glyph and labels live in three separate bands so nothing can overlap: the state
        # caption above the disc, the glyph inside it, the subtitle below.
        glyph_top = centre.y()-radius//2
        body_width = max(8, radius//5)
        body_height = max(14, radius//2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(LIVE if self._active else '#3A4666'))
        painter.drawRoundedRect(centre.x()-body_width//2, glyph_top,
                                body_width, body_height, body_width//2, body_width//2)
        painter.setPen(QPen(QColor(LIVE if self._active else '#3A4666'),
                            2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(centre.x()-body_width, glyph_top+body_height//5,
                        body_width*2, body_height, 0, 180*16)
        painter.drawLine(centre.x(), glyph_top+body_height+6,
                         centre.x(), glyph_top+body_height+16)

        font = painter.font()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(TEXT))
        state_bottom = glyph_top-6
        painter.drawText(QRect(0, state_bottom-24, self.width(), 24),
                         Qt.AlignmentFlag.AlignCenter, self._state)
        font.setBold(False)
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_MUTED))
        painter.drawText(QRect(0, centre.y()+radius+14, self.width(), 20),
                         Qt.AlignmentFlag.AlignCenter, self._subtitle)

    def mousePressEvent(self, event):
        self.tapped.emit()
        super().mousePressEvent(event)


class Sidebar(QFrame):
    """Navigation rail. Selecting a row switches the stacked page.

    Each row carries its own one-line purpose, because the rail is the only place a
    newcomer learns what the pages are for.
    """

    changed = Signal(str)
    help_clicked = Signal()

    PAGES = (('home', 'ボイスチェンジ', '◉', '選んで Start'),
             ('settings', '設定', '⚙', 'デバイス・微調整'))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('sidebar')
        self.setFixedWidth(196)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 18, 12, 18)
        layout.setSpacing(6)
        brand_box = QWidget()
        brand_row = QHBoxLayout(brand_box)
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_row.setSpacing(8)
        mark = QLabel()
        mark.setObjectName('brandIcon')
        from .splash import asset_path
        icon_file = asset_path('icon.png')
        if icon_file is not None:
            mark.setPixmap(QPixmap(str(icon_file)).scaled(
                26, 26, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        else:
            # No emoji fallback: a plain initial reads as a placeholder, not decoration.
            mark.setText('K')
        brand_row.addWidget(mark)
        brand = QLabel('Koeiro')
        brand.setObjectName('brand')
        brand_row.addWidget(brand)
        brand_row.addStretch(1)
        layout.addWidget(brand_box)
        layout.addSpacing(18)
        section = QLabel('メニュー')
        section.setObjectName('navSection')
        layout.addWidget(section)
        self.buttons = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for key, label, glyph, hint in self.PAGES:
            button = _NavButton(label, glyph, hint)
            button.setObjectName('navButton')
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(52)
            button.clicked.connect(lambda _checked=False, name=key: self.select(name))
            layout.addWidget(button)
            group.addButton(button)
            self.buttons[key] = button
        layout.addStretch(1)
        # The guide is reachable from the rail as well as the floating corner button,
        # so a skipped first-run dialog is never lost.
        self.help_button = QPushButton('？ 使い方ガイド')
        self.help_button.setObjectName('navButton')
        self.help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_button.setMinimumHeight(38)
        self.help_button.setToolTip('使い方ガイドを開きます（何度でも開けます）')
        self.help_button.clicked.connect(self.help_clicked)
        layout.addWidget(self.help_button)
        self.buttons['home'].setChecked(True)

    def select(self, key):
        if key not in self.buttons:
            return
        self.buttons[key].setChecked(True)
        self.changed.emit(key)


class _NavButton(QPushButton):
    """A rail row with a label and a purpose line, rather than a bare caption."""

    def __init__(self, label, glyph, hint, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 8, 12, 8)
        row.setSpacing(10)
        mark = QLabel(glyph)
        mark.setObjectName('navGlyph')
        mark.setFixedWidth(20)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(mark)
        text = QVBoxLayout()
        text.setSpacing(0)
        title = QLabel(label)
        title.setObjectName('navLabel')
        caption = QLabel(hint)
        caption.setObjectName('navHint')
        text.addWidget(title)
        text.addWidget(caption)
        row.addLayout(text, 1)
        self._parts = (mark, title, caption)


class PageHeader(QWidget):
    """Title plus purpose line shown at the top of every page.

    Without this the centre column looks identical on every tab and there is nothing
    telling the reader which page they are looking at. The badge mirrors the
    engine state the way a Draft pill mirrors an editor state.
    """

    BADGES = {
        'stopped': ('停止中', 'stopped'),
        'running': ('変換中', 'running'),
        'error': ('エラー', 'error'),
    }

    def __init__(self, title, purpose, parent=None):
        super().__init__(parent)
        self.setObjectName('pageHeader')
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 6)
        row.setSpacing(12)
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(2)
        self.title = QLabel(title)
        self.title.setObjectName('pageTitle')
        self.purpose = QLabel(purpose)
        self.purpose.setObjectName('pagePurpose')
        self.purpose.setWordWrap(True)
        text.addWidget(self.title)
        text.addWidget(self.purpose)
        row.addLayout(text, 1)
        self.badge = QLabel('停止中')
        self.badge.setObjectName('statusPill')
        self.badge.setProperty('badge', 'stopped')
        row.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignTop)

    def set_purpose(self, text):
        self.purpose.setText(str(text))

    def set_badge(self, kind):
        """kind: 'stopped', 'running' or 'error'. Unknown kinds keep the old pill."""
        label, _ = self.BADGES.get(kind, (None, None))
        if label is None:
            return
        self.badge.setText(label)
        self.badge.setProperty('badge', kind)
        self.style().unpolish(self.badge)
        self.style().polish(self.badge)


class VoicePanel(QFrame):
    """Right-hand voice list. Always visible, on every page.

    Picking a voice is step ② of every session, so it must not hide behind a tab:
    the list and its search stay one glance away while the centre column shows
    what is happening now.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('voicePanel')
        self.setFixedWidth(330)
        layout = QVBoxLayout(self)
        # The floating guide button overlaps the bottom-right corner, where the
        # output volume lives, so the panel keeps that corner clear.
        layout.setContentsMargins(16, 20, 16, 78)
        layout.setSpacing(10)
        title = QLabel('ボイスプリセット')
        title.setObjectName('railTitle')
        layout.addWidget(title)
        search_caption = QLabel('名前で絞り込む')
        search_caption.setObjectName('muted')
        layout.addWidget(search_caption)
        self.search = QLineEdit()
        self.search.setObjectName('voiceSearch')
        self.search.setClearButtonEnabled(True)
        layout.addWidget(self.search)
        tabs = QHBoxLayout()
        tabs.setSpacing(6)
        self.tab_group = QButtonGroup(self)
        self.tab_group.setExclusive(True)
        self.tabs = {}
        segment = QWidget()
        segment.setObjectName('segmentBox')
        segment_layout = QHBoxLayout(segment)
        segment_layout.setContentsMargins(3, 3, 3, 3)
        segment_layout.setSpacing(2)
        for key, label in (('all', 'すべて'), ('standard', '標準ボイス'),
                           ('user', '追加した声')):
            button = QPushButton(label)
            button.setObjectName('filterTab')
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.tab_group.addButton(button)
            segment_layout.addWidget(button)
            self.tabs[key] = button
        self.tabs['all'].setChecked(True)
        tabs.addWidget(segment, 1)
        layout.addLayout(tabs)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        self.cards_layout = QVBoxLayout(body)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.actions_layout = QVBoxLayout()
        self.actions_layout.setSpacing(8)
        layout.addLayout(self.actions_layout)
        output_label = QLabel('出力デバイス')
        output_label.setObjectName('muted')
        layout.addWidget(output_label)
        self.output_layout = QVBoxLayout()
        self.output_layout.setSpacing(8)
        layout.addLayout(self.output_layout)


class VoiceCard(QFrame):
    """One voice: avatar initial, name, kind and select."""

    selected = Signal(str)

    def __init__(self, key, name, sub, parent=None):
        super().__init__(parent)
        self.key = key
        self.setObjectName('voiceCard')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(10)
        avatar = QLabel((name or '?')[:1])
        avatar.setObjectName('voiceAvatar')
        avatar.setFixedSize(40, 40)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(avatar)
        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel(name)
        title.setObjectName('presetName')
        caption = QLabel(sub)
        caption.setObjectName('presetDetail')
        caption.setWordWrap(True)
        text.addWidget(title)
        text.addWidget(caption)
        row.addLayout(text, 1)
        choose = QPushButton('選択')
        choose.setObjectName('cardChoose')
        choose.setCursor(Qt.CursorShape.PointingHandCursor)
        choose.clicked.connect(lambda: self.selected.emit(self.key))
        row.addWidget(choose)
        self.set_selected(False)

    def set_selected(self, selected):
        self.setProperty('selected', bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event):
        self.selected.emit(self.key)
        super().mousePressEvent(event)


def stylesheet(asset_dir):
    """Shared dark-navy theme for the shell; widget-level tweaks stay in main_window."""
    import pathlib
    arrow = (pathlib.Path(asset_dir)/'chevron.svg').as_posix()
    return """
        QWidget { background: %(bg)s; color: %(text)s; font-family: 'Segoe UI'; font-size: 13px; }
        QFrame#sidebar { background: %(bg)s; border-right: 1px solid %(edge)s; }
        QLabel#brand { font-size: 15px; font-weight: 600; color: %(text)s; padding-bottom: 4px; }
        QLabel#brandIcon { font-size: 15px; font-weight: 600; color: %(text)s; }
        QLabel#navSection { color: %(muted)s; font-size: 11px; padding: 6px 0 0 14px; }
        QPushButton#navButton { background: transparent; border: none; border-radius: 8px;
                                color: %(muted)s; text-align: left; padding: 0px; }
        QPushButton#navButton:hover { background: %(pill)s; }
        QPushButton#navButton:checked { background: %(pill)s; }
        QLabel#navGlyph { color: %(muted)s; font-size: 15px; }
        QPushButton#navButton:checked QLabel#navGlyph { color: %(text)s; }
        QLabel#navLabel { color: %(muted)s; font-size: 14px; }
        QPushButton#navButton:checked QLabel#navLabel { color: %(text)s; font-weight: 600; }
        QLabel#navHint { color: %(muted)s; font-size: 11px; }
        QFrame#pageHeader { background: transparent; }
        QLabel#pageTitle { font-size: 19px; font-weight: 600; color: %(text)s; }
        QLabel#pagePurpose { color: %(muted)s; font-size: 12px; }
        QLabel#statusPill { font-size: 12px; font-weight: 600; padding: 5px 12px;
                            border-radius: 11px; }
        QLabel#statusPill[badge="stopped"] { background: %(pill)s; color: %(muted)s; }
        QLabel#statusPill[badge="running"] { background: %(live)s; color: %(bg)s; }
        QLabel#statusPill[badge="error"] { background: %(danger)s; color: #ffffff; }
        QFrame#voicePanel { background: %(bg)s; border-left: 1px solid %(edge)s; }
        QLabel#railTitle { font-size: 17px; font-weight: 600; }
        QLineEdit#voiceSearch { font-size: 13px; }
        QWidget#segmentBox { background: %(pill)s; border-radius: 10px; }
        QPushButton#filterTab { background: transparent; border: none;
                                border-radius: 7px; padding: 6px 4px; color: %(muted)s; font-size: 12px; }
        QPushButton#filterTab:hover { color: %(text)s; }
        QPushButton#filterTab:checked { background: #ffffff; color: %(bg)s; font-weight: 600; }
        QFrame#voiceCard { background: %(card)s; border: 1px solid %(border)s; border-radius: 12px; }
        QFrame#voiceCard:hover { border-color: %(border_dark)s; }
        QFrame#voiceCard[selected="true"] { border: 2px solid #ffffff; }
        QLabel#voiceAvatar { background: %(pill)s; border-radius: 20px; font-size: 18px;
                             font-weight: 600; color: %(muted)s; }
        QPushButton#cardChoose { padding: 8px 12px; }
        QLabel#presetName { font-size: 14px; font-weight: 600; }
        QLabel#presetDetail { color: %(muted)s; font-size: 11px; }
        QPushButton#routeButton { background: transparent; border: none;
                                  border-radius: 7px; padding: 10px 6px; font-weight: 600;
                                  color: %(muted)s; }
        QPushButton#routeButton:hover { color: %(text)s; }
        QPushButton#routeButton:checked { background: #ffffff; color: %(bg)s; }
        QLabel#muted { color: %(muted)s; font-size: 12px; }
        QLabel#title { font-size: 22px; font-weight: 600; }
        QLabel#sectionTitle { font-size: 16px; font-weight: 600; color: %(text)s; }
        QFrame#outputCard { background: %(card)s; border: 1px solid %(border)s; border-radius: 12px; }
        QFrame#pitchRow { background: transparent; }
        QFrame#transportBar { background: %(bg)s; border-top: 1px solid %(edge)s; }
        QLineEdit { background: %(bg)s; border: 1px solid %(border_dark)s; border-radius: 8px; padding: 8px; }
        QComboBox, QPushButton { background: %(pill)s; border: 1px solid %(border_dark)s;
                                 border-radius: 8px; padding: 7px; color: %(text)s; }
        QComboBox { padding-right: 26px; }
        QComboBox::drop-down { width: 22px; border: none; }
        QComboBox::down-arrow { image: url(__ARROW__); width: 12px; height: 8px; }
        QComboBox QAbstractItemView { background: %(card)s; border: 1px solid %(border_dark)s;
                                      selection-background-color: %(pill)s; selection-color: %(text)s; }
        QPushButton:hover { border-color: %(live)s; }
        QPushButton:focus, QComboBox:focus, QLineEdit:focus { border-color: %(live)s; }
        QCheckBox { spacing: 8px; color: %(text)s; border: 1px solid transparent; border-radius: 4px; }
        QCheckBox:focus { border-color: %(live)s; }
        QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid %(border_dark)s;
                               border-radius: 4px; background: %(bg)s; }
        QCheckBox::indicator:checked { background: %(live)s; border-color: %(live)s; }
        QSlider { border: 1px solid transparent; border-radius: 4px; }
        QSlider:focus { border-color: %(live)s; }
        QPushButton#start { background: %(primary)s; border-color: %(primary)s; color: %(primary_text)s;
                            font-weight: 600; padding: 10px 26px; }
        QPushButton#start:focus { border-color: %(live)s; }
        QPushButton#start:disabled { background: %(pill)s; color: #59617A; border-color: %(pill)s; }
        QPushButton#stop { padding: 10px 22px; }
        QPushButton#helpFab { background: %(primary)s; border: 2px solid %(primary)s;
                              border-radius: 20px; color: %(primary_text)s;
                              font-weight: 600; padding: 6px 18px; }
        QPushButton#helpFab:focus { border-color: %(live)s; }
        QPushButton#helpFab:hover { background: #D8DEEA; border-color: #D8DEEA; }
        QLabel#warningText { color: %(warn)s; font-size: 12px; }
        QLabel#stepNumber { color: %(text)s; font-size: 13px; font-weight: 700; }
        QLabel#stepTitle { color: %(text)s; font-size: 15px; font-weight: 600; }
        QFrame#stepCard { background: %(card)s; border: 1px solid %(border)s; border-radius: 12px; }
        *[guide="true"] { border: 2px solid %(live)s; border-radius: 8px; }
        QTabWidget::pane { border: 1px solid %(border)s; border-radius: 10px; background: %(card)s; }
        QTabBar::tab { padding: 8px 12px; color: %(muted)s; }
        QTabBar::tab:selected { color: %(text)s; background: %(card)s; }
        QSlider::groove:horizontal { background: %(track)s; height: 5px; border-radius: 2px; }
        QSlider::handle:horizontal { background: #ffffff; border: 1px solid %(border_dark)s;
                                     width: 16px; margin: -6px 0; border-radius: 8px; }
        QProgressBar { border: none; background: %(track)s; border-radius: 4px; height: 10px; }
        QProgressBar::chunk { background: #ffffff; border-radius: 4px; }
        QCheckBox { spacing: 8px; color: %(text)s; }
        QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
        QScrollBar::handle:vertical { background: %(border_dark)s; border-radius: 4px; min-height: 30px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
        QScrollBar::handle:horizontal { background: %(border_dark)s; border-radius: 4px; min-width: 30px; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        QToolTip { background: %(primary)s; color: %(primary_text)s; border: none; padding: 5px; }
    """.replace('__ARROW__', arrow) % {
        'bg': BG, 'edge': '#1C2544', 'text': TEXT, 'muted': TEXT_MUTED,
        'primary': PRIMARY, 'primary_text': PRIMARY_TEXT, 'live': LIVE,
        'pill': PILL, 'track': TRACK, 'card': CARD, 'border': BORDER,
        'border_dark': BORDER_DARK, 'danger': DANGER, 'warn': WARN}