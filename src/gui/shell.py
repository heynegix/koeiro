"""Sidebar navigation, the centre orb and the preset rail.

The window is reorganised around a fixed three-column layout: a navigation rail on the
left, a centred status/pitch/output column, and a preset rail on the right. Every widget
the rest of the application already drives is still built by MainWindow; this module
only owns the shell, so the voice logic and its tests stay where they are.
"""
from math import cos, pi, sin

from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QLinearGradient, QPen
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

# What each page is for, in the reader's words rather than the code's. The rail lists
# them in the order a session uses them: pick devices, pick a voice, then tune.
PAGE_PURPOSE = {
    'home': '① デバイス → ② モードと声 → ③ Start。この順で進めれば変換が始まります。',
    'library': '変換に使う声と変換方法を選びます。音声ファイルから自分の声も追加できます。',
    'presets': 'Female DSP（軽い加工）の調整ページです。AI Voiceでは使いません。',
    'settings': 'デバイス・音量・バッファと、モニター（自分の耳で確認）の設定です。',
    'advanced': 'AI処理の詳細と診断です。通常の使用では変更する必要はありません。',
}

ACCENT = '#3d8bfd'
ACCENT_SOFT = '#5aa9ff'
ACCENT_DIM = '#1d3a63'
SURFACE = '#1b2436'
SURFACE_RAISED = '#243049'
BORDER = '#2c3a55'
TEXT = '#e8eef9'
TEXT_MUTED = '#8ea0bd'


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
        painter.fillRect(rect, QColor(SURFACE))
        count = len(self._levels) or 1
        gap = 2
        span = (rect.height()-gap*(count-1))/count
        bar_height = max(1, int(span))
        for index, level in enumerate(self._levels):
            y = int(index*(span+gap))
            filled = int(max(0.0, min(1.0, level))*rect.width())
            painter.fillRect(0, y, rect.width(), bar_height, QColor('#253048'))
            if filled:
                painter.fillRect(0, y, filled, bar_height, QColor(ACCENT))
        # The newest peak is drawn across the strip so a single reading is visible even
        # though the per-bar history is not sampled at audio rate.
        peak_width = int(max(0.0, min(1.0, self._peak))*rect.width())
        if peak_width > 0:
            painter.fillRect(rect.width()-peak_width, 0, peak_width, rect.height(),
                             QColor(ACCENT_SOFT))


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
        glow.setColorAt(0.0, QColor(ACCENT_DIM))
        glow.setColorAt(1.0, QColor(SURFACE))
        painter.setBrush(glow)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(centre, radius, radius)

        ring_width = 4
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor('#2b3a55'), ring_width))
        painter.drawEllipse(centre, radius, radius)
        # The ring arc is the actual level, so the visual cannot claim signal that the
        # audio thread has not measured.
        sweep = int(-90 + 360*self._level)
        if sweep > 0:
            painter.setPen(QPen(QColor(ACCENT_SOFT), ring_width, Qt.PenStyle.SolidLine,
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
            painter.setBrush(QColor(ACCENT if self._active else ACCENT_DIM))
            painter.drawRoundedRect(x-2, y-2, 4, 4, 2, 2)

        # Glyph and labels live in three separate bands so nothing can overlap: the state
        # caption above the disc, the glyph inside it, the subtitle below.
        glyph_top = centre.y()-radius//2
        body_width = max(8, radius//5)
        body_height = max(14, radius//2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(ACCENT if self._active else ACCENT_DIM))
        painter.drawRoundedRect(centre.x()-body_width//2, glyph_top,
                                body_width, body_height, body_width//2, body_width//2)
        painter.setPen(QPen(QColor(ACCENT_SOFT if self._active else ACCENT_DIM),
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

    PAGES = (('home', 'ホーム', '⌂', '変換の開始と停止'),
             ('library', '声ライブラリ', '♫', '声の選択・追加・試聴'),
             ('presets', '声の加工', '◈', 'Female DSP の調整'),
             ('settings', '設定', '⚒', 'デバイス・音量・モニター'),
             ('advanced', '詳細設定', '⚙', 'AI処理の詳細と診断'))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('sidebar')
        self.setFixedWidth(196)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 18, 12, 18)
        layout.setSpacing(6)
        brand = QLabel('🎙  Anime Voice Changer')
        brand.setObjectName('brand')
        layout.addWidget(brand)
        layout.addSpacing(18)
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
    telling the reader which page they are looking at.
    """

    def __init__(self, title, purpose, parent=None):
        super().__init__(parent)
        self.setObjectName('pageHeader')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 6)
        layout.setSpacing(2)
        self.title = QLabel(title)
        self.title.setObjectName('pageTitle')
        self.purpose = QLabel(purpose)
        self.purpose.setObjectName('pagePurpose')
        self.purpose.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.purpose)

    def set_purpose(self, text):
        self.purpose.setText(str(text))


class PresetRail(QFrame):
    """Right-hand preset cards. One is selected at a time."""

    selected = Signal(str)

    def __init__(self, presets, parent=None):
        super().__init__(parent)
        self.setObjectName('presetRail')
        self.setFixedWidth(322)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(10)
        header = QHBoxLayout()
        title = QLabel('変換方法')
        title.setObjectName('railTitle')
        header.addWidget(title)
        header.addStretch(1)
        self.link = QPushButton('声ライブラリ ›')
        self.link.setObjectName('railLink')
        header.addWidget(self.link)
        layout.addLayout(header)
        # The rail switches the delivery route only; saying so here stops the cards from
        # reading like they also pick the voice.
        caption = QLabel('話し方のルートを切り替えます。声は「声ライブラリ」で選びます。')
        caption.setObjectName('muted')
        caption.setWordWrap(True)
        layout.addWidget(caption)
        self.cards = {}
        for key, label, detail, glyph in presets:
            card = _PresetCard(label, detail, glyph)
            card.clicked.connect(lambda name=key: self.choose(name))
            layout.addWidget(card)
            self.cards[key] = card
        layout.addStretch(1)

    def choose(self, key):
        for name, card in self.cards.items():
            card.setSelected(name == key)
        self.selected.emit(key)

    def select_silently(self, key):
        for name, card in self.cards.items():
            card.setSelected(name == key)


class _PresetCard(QFrame):
    clicked = Signal()

    def __init__(self, label, detail, glyph):
        super().__init__()
        self.setObjectName('presetCard')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(78)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 12, 14, 12)
        row.setSpacing(12)
        badge = QLabel(glyph)
        badge.setObjectName('presetGlyph')
        badge.setFixedSize(40, 40)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(badge)
        text = QVBoxLayout()
        text.setSpacing(2)
        name = QLabel(label)
        name.setObjectName('presetName')
        caption = QLabel(detail)
        caption.setObjectName('presetDetail')
        text.addWidget(name)
        text.addWidget(caption)
        row.addLayout(text, 1)
        self.dot = QLabel('●')
        self.dot.setObjectName('presetDot')
        self.dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.dot)
        self.setSelected(False)

    def setSelected(self, selected):
        self.setProperty('selected', bool(selected))
        self.style().unpolish(self)
        self.style().polish(self)
        self.dot.setText('●' if selected else '○')

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


def stylesheet(asset_dir):
    """Shared dark theme for the shell; widget-level tweaks stay in main_window."""
    import pathlib
    arrow = (pathlib.Path(asset_dir)/'chevron.svg').as_posix()
    return """
        QWidget { background: #131a28; color: %(text)s; font-family: 'Segoe UI'; font-size: 13px; }
        QFrame#sidebar { background: #151d2c; border-right: 1px solid %(border)s; }
        QLabel#brand { font-size: 15px; font-weight: 600; color: %(text)s; padding-bottom: 4px; }
        QPushButton#navButton { background: transparent; border: none; border-radius: 8px;
                                color: %(muted)s; text-align: left; padding: 0px; }
        QPushButton#navButton:hover { background: #1d2739; }
        QPushButton#navButton:checked { background: %(accent_dim)s;
                                        border-left: 3px solid %(accent)s; }
        QLabel#navGlyph { color: %(muted)s; font-size: 15px; }
        QPushButton#navButton:checked QLabel#navGlyph { color: %(accent)s; }
        QLabel#navLabel { color: %(text)s; font-size: 14px; }
        QPushButton#navButton:checked QLabel#navLabel { color: %(accent_soft)s; font-weight: 600; }
        QLabel#navHint { color: %(muted)s; font-size: 11px; }
        QFrame#pageHeader { background: transparent; }
        QLabel#pageTitle { font-size: 19px; font-weight: 600; color: %(text)s; }
        QLabel#pagePurpose { color: %(muted)s; font-size: 12px; }
        QFrame#presetRail { background: #151d2c; border-left: 1px solid %(border)s; }
        QLabel#railTitle { font-size: 17px; font-weight: 600; }
        QPushButton#railLink { background: transparent; border: none; color: %(accent)s; font-size: 12px; }
        QFrame#presetCard { background: %(surface)s; border: 1px solid %(border)s; border-radius: 12px; }
        QFrame#presetCard:hover { background: %(raised)s; }
        QFrame#presetCard[selected="true"] { background: %(accent_dim)s; border: 1px solid %(accent)s; }
        QLabel#presetGlyph { background: %(raised)s; border-radius: 20px; font-size: 18px; color: %(accent)s; }
        QLabel#presetName { font-size: 14px; font-weight: 600; }
        QLabel#presetDetail { color: %(muted)s; font-size: 11px; }
        QLabel#presetDot { color: %(muted)s; font-size: 13px; }
        QLabel#orbState { font-size: 16px; font-weight: 600; }
        QLabel#railHeading { font-size: 17px; font-weight: 600; color: %(text)s; }
        QLabel#muted { color: %(muted)s; font-size: 12px; }
        QLabel#title { font-size: 22px; font-weight: 600; }
        QLabel#sectionTitle { font-size: 16px; font-weight: 600; color: %(accent)s; }
        QFrame#outputCard { background: %(surface)s; border: 1px solid %(border)s; border-radius: 12px; }
        QFrame#pitchRow { background: transparent; }
        QLineEdit { background: %(surface)s; border: 1px solid %(border)s; border-radius: 8px; padding: 8px; }
        QComboBox, QPushButton { background: %(surface)s; border: 1px solid %(border)s;
                                 border-radius: 8px; padding: 7px; }
        QComboBox { padding-right: 26px; }
        QComboBox::drop-down { width: 22px; border: none; }
        QComboBox::down-arrow { image: url(__ARROW__); width: 12px; height: 8px; }
        QPushButton:hover { background: %(raised)s; }
        QPushButton#start { background: %(accent)s; border-color: %(accent)s; color: #ffffff;
                            font-weight: 600; padding: 10px 26px; }
        QPushButton#start:disabled { background: #223049; color: #64748b; border-color: %(border)s; }
        QPushButton#stop { padding: 10px 22px; }
        QPushButton#helpFab { background: %(accent_dim)s; border: 1px solid %(accent)s;
                              border-radius: 20px; color: %(accent_soft)s;
                              font-weight: 600; padding: 6px 18px; }
        QPushButton#helpFab:hover { background: %(accent)s; color: #ffffff; }
        QLabel#warningText { color: #ffcf7a; font-size: 12px; }
        QLabel#stepNumber { color: %(accent)s; font-size: 13px; font-weight: 700; }
        QLabel#stepTitle { color: %(text)s; font-size: 15px; font-weight: 600; }
        QFrame#stepCard { background: %(surface)s; border: 1px solid %(border)s; border-radius: 12px; }
        QFrame#stepCard[highlight="true"] { border: 1px solid %(accent)s; background: %(raised)s; }
        QTabWidget::pane { border: 1px solid %(border)s; border-radius: 10px; }
        QTabBar::tab { padding: 8px 12px; color: %(muted)s; }
        QTabBar::tab:selected { color: %(accent)s; background: %(surface)s; }
        QSlider::groove:horizontal { background: #33465f; height: 5px; border-radius: 2px; }
        QSlider::handle:horizontal { background: %(accent)s; width: 16px; margin: -6px 0;
                                      border-radius: 8px; }
        QProgressBar { border: none; background: #263348; border-radius: 4px; height: 10px; }
        QProgressBar::chunk { background: %(accent)s; border-radius: 4px; }
    """.replace('__ARROW__', arrow) % {
        'text': TEXT, 'muted': TEXT_MUTED, 'accent': ACCENT, 'accent_soft': ACCENT_SOFT,
        'accent_dim': ACCENT_DIM, 'surface': SURFACE, 'raised': SURFACE_RAISED,
        'border': BORDER}