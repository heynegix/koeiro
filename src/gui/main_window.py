from dataclasses import asdict, replace
from datetime import datetime
import json
import logging
import os
from src.vc.models import DEFAULT_DELIVERY, DELIVERY_MODES, is_meanvc2
from pathlib import Path
import sys
import time

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QHBoxLayout,
                               QCheckBox, QFrame, QLabel, QLayout, QMainWindow, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout,
                               QWidget, QDoubleSpinBox, QSpinBox, QFileDialog)

from src.audio.controller import AudioController
from src.audio.devices import choose_device
from src.audio.engine import AudioEngine, EngineConfig
from src.audio.meters import amplitude_to_db
from src.settings.manager import BUFFERS, SAMPLE_RATES
from src.processors.chain import ProcessorChain
from src.processors.dsp_parameters import DSPParameters
from src.processors.female_dsp import FemaleDSPProcessor
from src.processors.voice_router import VoiceRouter
from src.processors.ai_voice import AIVoiceProcessor
from src.vc.config import AIParameters
from src.vc.health import realtime_health
from src.vc.diagnostics import startup_diagnostics
from src.utils.background_files import BackgroundFiles
from src.presets.female_presets import PRESETS, load_preset
from src.prosody.parameters import ProsodyParameters, PRESETS as PROSODY_PRESETS, load_preset as load_prosody_preset

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, settings_manager, controller=None):
        super().__init__()
        self.manager = settings_manager
        self.settings = self.manager.load()
        self.files = BackgroundFiles()
        self.startup_diagnostic = None
        self._diagnostic_model = self.settings.ai_model
        def diagnose():
            self._diagnose_voice(self.settings.ai_model)
        self.files.submit('diagnostics', diagnose)
        if controller is None:
            self.dsp = FemaleDSPProcessor(self.settings.dsp_parameters())
            self.router = VoiceRouter(self.dsp, AIVoiceProcessor(self.settings.ai_parameters()), self.settings.voice_mode)
            self.controller = AudioController(AudioEngine(chain=ProcessorChain(main_processor=self.router)))
        else:
            self.controller = controller
            main = controller.engine.chain.main_processor
            self.router = main if isinstance(main, VoiceRouter) else None
            self.dsp = main.dsp if self.router else main if isinstance(main, FemaleDSPProcessor) else None
            if self.dsp:
                self.dsp.set_parameters(self.settings.dsp_parameters())
            # An injected processor was built with its own defaults, so adopt the saved
            # AI settings here instead of waiting for the first user change to apply them.
            if self.router:
                self.router.ai.bridge.parameters = self.settings.ai_parameters()
        self._revision = -1
        self._pending = False
        self._pending_request = 0
        self._closing = False
        self._can_close = False
        self._display_input = self._display_output = 0.0
        self._next_performance_update = 0.0
        self._logged_crossfades = 0
        self.voice_dialog = None
        self.reference_player = None
        self.preview_player = None
        self.preview_output = None
        self._preview_busy = False
        self._active_page = 'home'
        self.setWindowTitle("Anime Voice Changer")
        self.setMinimumSize(1040, 660)
        self.resize(max(self.settings.window_size[0], 1180),
                    max(self.settings.window_size[1], 780))
        self._build_ui()
        self._update_meanvc_controls()
        self._restore_position()
        self.controller.set_gain(self.settings.gain_db)
        self.controller.set_gate(self.settings.noise_gate_db)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # Preset rail contents. Only the two shipped delivery modes, because a
    # card that picks a voice or a route we removed would silently do nothing,
    # and experimental comparison modes stay out of the quick rail.
    RAIL_PRESETS = tuple(
        (key, spec['label'], spec['detail'], icon)
        for (key, spec), icon in zip(
            ((key, spec) for key, spec in DELIVERY_MODES.items()
             if spec.get('experiment','none')=='none'),
            ('⚡', '◍')))

    def _rail_targets(self):
        """Resolve each rail entry to (mode, delivery key) against the live profiles."""
        return {key: ('ai_voice', key) for key in DELIVERY_MODES}

    def _apply_rail_preset(self, key):
        """Apply a rail preset: a mode plus one of the two delivery routes."""
        targets = self._rail_targets()
        if key not in targets:
            return
        mode, delivery = targets[key]
        if self._pending or self._closing or self.controller.engine.running:
            self.rail_note.setText('変換中は設定を変更できません。Stopしてから変更してください。')
            return
        index = self.mode.findData(mode)
        if index >= 0:
            self.mode.setCurrentIndex(index)
        if self.ai_delivery.findData(delivery) >= 0:
            self.ai_delivery.setCurrentIndex(self.ai_delivery.findData(delivery))
        label = next((entry[1] for entry in self.RAIL_PRESETS if entry[0] == key), key)
        self.rail_note.setText('%s に切り替えます。声は「声ライブラリ」で選べます。' % label)
        self._capture_settings()
        self._save_settings()

    def _build_ui(self):
        """Three columns: navigation rail, centred status, preset rail.

        Page bodies are built by the existing builders and simply re-parented, so every
        control keeps its identity and the rest of the window code is untouched.
        """
        from . import shell
        from .shell import MicOrb, PageHeader, PresetRail, Sidebar, stylesheet
        container = QWidget()
        outer = QHBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(container)

        self.sidebar = Sidebar()
        self.sidebar.changed.connect(self._show_page)
        outer.addWidget(self.sidebar)

        centre = QWidget()
        centre_outer = QHBoxLayout(centre)
        centre_outer.setContentsMargins(0, 0, 0, 0)
        centre_outer.setSpacing(0)
        outer.addWidget(centre, 1)

        # A single scroll area holds the centre column, so the orb, the pitch row, the
        # output card and the transport scroll together on a short window while Start
        # and Stop stay reachable.
        centre_scroll = QScrollArea()
        centre_scroll.setWidgetResizable(True)
        centre_scroll.setFrameShape(QFrame.Shape.NoFrame)
        centre_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        centre_body = QWidget()
        centre_layout = QVBoxLayout(centre_body)
        centre_layout.setContentsMargins(24, 18, 24, 14)
        centre_layout.setSpacing(12)
        centre_scroll.setWidget(centre_body)
        centre_outer.addWidget(centre_scroll, 1)
        self.centre_scroll = centre_scroll

        self.rail_note = QLabel('変換方法（話し方のルート）を選ぶと、右のカードで切り替わります。'
                                '声は「声ライブラリ」で選べます。')
        self.rail_note.setObjectName('muted')
        self.rail_note.setWordWrap(True)
        centre_layout.addWidget(self.rail_note)

        self.pages = {}
        self.page_headers = {}
        for key, label, _glyph, _hint in Sidebar.PAGES:
            page = QWidget()
            self.pages[key] = (page, QVBoxLayout(page))
            header = PageHeader(label, shell.PAGE_PURPOSE.get(key, ''))
            self.pages[key][1].addWidget(header)
            self.page_headers[key] = header
            centre_layout.addWidget(page, 1)
        outer.addWidget(centre, 1)

        self.rail = PresetRail(self.RAIL_PRESETS)
        self.rail.selected.connect(self._apply_rail_preset)
        self.rail.link.setToolTip('変換方法の一覧と声の選択は「声ライブラリ」にあります。')
        self.rail.link.clicked.connect(lambda: self._show_page('library'))
        outer.addWidget(self.rail)
        self.rail.select_silently('default')
        self.setStyleSheet(stylesheet(Path(__file__).parent/'assets'))

        layout = self.pages['home'][1]
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        # The home page is a three-step flow: choose devices, choose the mode and voice,
        # then press Start. Each step is numbered and its own card, so the order is
        # visible without reading any help text.
        device_card, device_column = self._step_card(
            '①', 'マイクと出力先を選ぶ',
            '入力と出力は同じ方式にそろえます（Windowsは両方 [Windows WASAPI] がおすすめ）。')
        form = QFormLayout()
        form.setVerticalSpacing(12)
        self.input_device = QComboBox()
        self.output_device = QComboBox()
        self.input_device.setMinimumWidth(240)
        self.output_device.setMinimumWidth(240)
        for combo in (self.input_device, self.output_device):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        form.addRow("マイク入力", self.input_device)
        form.addRow("音声の出力先", self.output_device)
        device_column.addLayout(form)
        self.route_hint = QLabel("デバイスを読み込んでいます…")
        self.route_hint.setWordWrap(True)
        self.route_hint.setObjectName("muted")
        device_column.addWidget(self.route_hint)
        self.refresh = QPushButton("デバイス一覧を更新")
        self.refresh.setToolTip('マイクや仮想ケーブルを接続したあとに押してください。')
        self.refresh.clicked.connect(self._refresh_devices)
        device_column.addWidget(self.refresh)
        layout.addWidget(device_card)

        sound_card, sound_column = self._step_card(
            '②', '変換モードと声を選ぶ',
            'AI Voice は「声ライブラリ」で選んだ声に変換します。まずは標準ボイスで試してください。')
        mode_row = QFormLayout()
        mode_row.setVerticalSpacing(12)
        self.mode = QComboBox()
        self.mode.addItem('Original / 変換なし', 'original')
        self.mode.addItem('Female DSP / 軽い加工', 'female_dsp')
        self.mode.addItem('AI Voice / 声の変換', 'ai_voice')
        self.mode.setToolTip('声の変換方法を選びます。\n'
                          'Original: 変換しません\n'
                          'Female DSP: 軽い加工（CPUが軽い）\n'
                          'AI Voice: 選んだ声で完全に別の声に変換')
        self.mode.setMinimumHeight(34)
        mode_row.addRow('変換モード', self.mode)
        self.voice_quick = QComboBox()
        self.voice_quick.setMinimumHeight(34)
        self.voice_quick.setToolTip('変換先の声です。「声ライブラリ」と同じ選択を共有します。')
        self.voice_quick.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.voice_quick.setMinimumContentsLength(16)
        mode_row.addRow('変換したい声', self.voice_quick)
        sound_column.addLayout(mode_row)
        self.voice_quick_note = QLabel('')
        self.voice_quick_note.setObjectName('muted')
        self.voice_quick_note.setWordWrap(True)
        sound_column.addWidget(self.voice_quick_note)
        library_link = QPushButton('声ライブラリを開く（声の追加・試聴）')
        library_link.clicked.connect(lambda: self._show_page('library'))
        sound_column.addWidget(library_link)
        layout.addWidget(sound_card)
        voice_tab, ai_tab = QWidget(), QWidget()
        self._build_dsp_ui(QVBoxLayout(voice_tab))
        self._build_ai_ui(QVBoxLayout(ai_tab))
        prosody_tab = QWidget()
        self.prosody_tab = prosody_tab
        self._build_prosody_ui(QVBoxLayout(prosody_tab))
        advanced_tab = QWidget()
        self.pages['library'][1].addWidget(ai_tab)
        self.pages['presets'][1].addWidget(voice_tab)
        self.pages['advanced'][1].addWidget(prosody_tab)
        advanced = self.pages['advanced'][1]
        # Shown instead of the prosody controls for the standard voice, whose pitch and
        # intonation are fixed by the shipped reference: the tabs are hidden, not dead.
        self.prosody_fixed_note = QLabel(
            'この声では抑揚補正（Prosody）は使いません。標準ボイスのピッチと抑揚は'
            '同梱のReferenceに固定されているためです。')
        self.prosody_fixed_note.setWordWrap(True)
        self.prosody_fixed_note.setObjectName('muted')
        self.prosody_fixed_note.setVisible(False)
        advanced.addWidget(self.prosody_fixed_note)
        advanced.addWidget(self.ai_details)
        self._build_prosody_advanced(advanced)
        self.ai_pitch, self.ai_pitch_label = self._slider(advanced,'Beatrice native Pitch',-96,96,round(self.settings.ai_pitch*8))
        self.ai_pitch_label.setText(f'{self.ai_pitch.value()/8:+.3f} st')
        self.ai_pitch.valueChanged.connect(self._ai_fx_changed)
        self.ai_wait, self.ai_wait_label = self._slider(advanced,'AI pre-roll wait · Experimental',20,156,round(self.settings.ai_output_wait_ms))
        self.ai_fade, self.ai_fade_label = self._slider(advanced,'Mode crossfade',20,100,round(self.settings.ai_crossfade_ms))
        self.ai_wait_label.setText(f'{self.ai_wait.value()} ms · pre-roll（実測遅延ではありません）')
        self.ai_fade_label.setText(f'{self.ai_fade.value()} ms')
        self.ai_wait.valueChanged.connect(self._ai_structure_changed)
        self.ai_fade.valueChanged.connect(self._ai_structure_changed)
        self.ai_performance.setParent(advanced_tab)
        advanced.addWidget(self.ai_performance)
        self.debug_snapshot = QPushButton('Debug Snapshotを保存')
        self.debug_snapshot.clicked.connect(self._debug_snapshot)
        advanced.addWidget(self.debug_snapshot)
        self.file_status = QLabel('')
        self.file_status.setWordWrap(True)
        advanced.addWidget(self.file_status)
        self.mode.setCurrentIndex(self.mode.findData(self.settings.voice_mode))
        self._publish_dsp()
        layout = self.pages['settings'][1]
        self.gain, self.gain_label = self._slider(layout, "Gain", -200, 200,
                                                 round(self.settings.gain_db * 10))
        self.gate, self.gate_label = self._slider(layout, "Noise Gate", -800, -100,
                                                 round(self.settings.noise_gate_db * 10))
        self.gain.valueChanged.connect(self._gain_changed)
        self.gate.valueChanged.connect(self._gate_changed)
        self._gain_changed(self.gain.value())
        self._gate_changed(self.gate.value())
        options = QFormLayout()
        self.rate = QComboBox()
        for rate in SAMPLE_RATES:
            self.rate.addItem(f"{rate} Hz", rate)
        self.rate.setCurrentIndex(self.rate.findData(self.settings.sample_rate))
        options.addRow("Sample Rate", self.rate)
        self.buffer = QComboBox()
        for frames in BUFFERS:
            self.buffer.addItem(f"{frames} samples", frames)
        self.buffer.setCurrentIndex(self.buffer.findData(self.settings.buffer_size))
        options.addRow("Buffer", self.buffer)
        self.recommended = QPushButton('推奨設定を適用（48 kHz / 256 / 標準ボイス）')
        self.recommended.setToolTip('このPCで実測した推奨値にそろえます。デバイスの選択は変わりません。')
        self.recommended.clicked.connect(self._apply_recommended)
        self.monitor_device = QComboBox()
        self.monitor_device.setMinimumWidth(240)
        self.monitor_device.setToolTip("モニター音声を鳴らすデバイス。マイクや出力先とは別のもの"
                                       "（ヘッドホン等）を選んでください。")
        self.monitor_toggle = QPushButton("Monitor OFF")
        self.monitor_toggle.setCheckable(True)
        self.monitor_toggle.blockSignals(True)
        self.monitor_toggle.setChecked(self.settings.monitor)
        self.monitor_toggle.blockSignals(False)
        if self.settings.monitor:
            self.monitor_toggle.setText("Monitor ON")
        self.monitor_toggle.setMinimumHeight(34)
        self.monitor_toggle.setToolTip("変換後の音声を別デバイス（ヘッドホン等）で聴きます。\n"
                                       "停止中でも変換中でも切り替えられます。\n"
                                       "出力先と同じにすると二重に聞こえるので別々にしてください。")
        self.monitor_toggle.toggled.connect(self._monitor_toggled)
        monitor_row = QHBoxLayout()
        monitor_row.addWidget(self.monitor_device, 1)
        monitor_row.addWidget(self.monitor_toggle)
        options.addRow("Monitor", monitor_row)
        layout.addLayout(options)
        layout.addWidget(self.recommended)

        # Monitor lives in its own card: level, a device check by ear, and the live
        # status. It is applied without stopping the stream, so the section stays
        # enabled while the engine runs.
        monitor_card = QFrame()
        monitor_card.setObjectName('outputCard')
        monitor_column = QVBoxLayout(monitor_card)
        monitor_column.setContentsMargins(16, 12, 16, 12)
        monitor_column.setSpacing(6)
        monitor_heading = QLabel('モニター（変換後の音を自分の耳で確認）')
        monitor_heading.setObjectName('sectionTitle')
        monitor_column.addWidget(monitor_heading)
        self.monitor_volume, self.monitor_volume_label = self._slider(
            monitor_column, 'モニター音量', -300, 60, round(self.settings.monitor_volume_db*10))
        self.monitor_test = QPushButton('このデバイスでテスト音を鳴らす')
        self.monitor_test.setToolTip('選んだデバイスから短いチャイムを1回だけ鳴らします。\n'
                                     '変換中でも使えます。')
        self.monitor_test.clicked.connect(self._test_monitor)
        monitor_column.addWidget(self.monitor_test)
        self.monitor_warning = QLabel('')
        self.monitor_warning.setWordWrap(True)
        self.monitor_warning.setObjectName('warningText')
        monitor_column.addWidget(self.monitor_warning)
        self.monitor_note = QLabel('モニターは変換後の音声を別デバイスで聴く機能です。多少の途切れは仕様です。')
        self.monitor_note.setWordWrap(True)
        self.monitor_note.setObjectName('muted')
        monitor_column.addWidget(self.monitor_note)
        layout.addWidget(monitor_card)
        self.monitor_volume.valueChanged.connect(self._monitor_volume_changed)
        self.monitor_device.currentIndexChanged.connect(self._monitor_changed)
        self._monitor_test_note = None
        self._monitor_test_busy = False
        self._monitor_apply = QTimer(self)
        self._monitor_apply.setSingleShot(True)
        self._monitor_apply.setInterval(120)
        self._monitor_apply.timeout.connect(self._apply_monitor)
        self._update_monitor_ui()

        # --- centre column: orb, meters, pitch, output card, transport ---
        from .shell import LevelBar
        stage = QWidget()
        stage.setObjectName('stage')
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(10)
        # The orb takes the room that is left over; every other row keeps its own height.
        self.orb = MicOrb()
        self.orb.tapped.connect(self._toggle_transport)
        stage_layout.addWidget(self.orb, 0)
        meters = QHBoxLayout()
        meters.setSpacing(18)
        self.input_meter, self.input_db = self._meter(meters, "入力レベル")
        self.output_meter, self.output_db = self._meter(meters, "出力レベル")
        stage_layout.addLayout(meters)
        # The textual meter already names the level, so the strips below carry only the
        # bar itself. Two captions per meter read as two different measurements.
        self.input_bar = LevelBar(bars=18)
        self.output_bar = LevelBar(bars=18)
        bars = QHBoxLayout()
        bars.setSpacing(18)
        bars.addWidget(self.input_bar, 1)
        bars.addWidget(self.output_bar, 1)
        stage_layout.addLayout(bars)
        self._build_output_card(stage_layout)
        # `layout` now points at the settings page, so the stage goes to the home page
        # explicitly and sits above the routing form: the orb is the first thing seen.
        self.pages['home'][1].insertWidget(1, stage, 1)
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(0, 4, 0, 0)
        footer_layout.setSpacing(6)
        # Step ③ closes the home flow: the transport and its status line live in the same
        # numbered card as ①②, so the page states the whole sequence devices → voice → Start.
        start_card, start_column = self._step_card(
            '③', 'Start して話す',
            'Startで変換開始、Stopで停止します。中央のオーブをクリックしても同じです。')
        actions = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        actions.addStretch(1)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        start_column.addLayout(actions)
        self.status = QLabel("Status: Stopped")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        start_column.addWidget(self.status)
        footer_layout.addWidget(start_card)
        self.latency = QLabel()
        self.latency.setWordWrap(True)
        self.latency.setObjectName("muted")
        self.xruns = QLabel("Underflow: 0 · Overflow: 0")
        self.xruns.setObjectName("muted")
        self.performance = QLabel()
        self.performance.setWordWrap(True)
        self.performance.setObjectName("muted")
        footer_layout.addWidget(self.status)
        footer_layout.addWidget(self.latency)
        self.health = QLabel('Realtime Health: Waiting')
        self.health.setObjectName('muted')
        self.health.setWordWrap(True)
        footer_layout.addWidget(self.health)
        self.pages['home'][1].addWidget(footer)
        advanced.addWidget(self.performance)
        self.input_device.currentIndexChanged.connect(self._route_changed)
        self.output_device.currentIndexChanged.connect(self._route_changed)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.rate.currentIndexChanged.connect(self._update_latency)
        self.buffer.currentIndexChanged.connect(self._update_latency)
        self._update_latency()
        for combo in (self.input_device, self.output_device, self.rate, self.buffer, self.monitor_device):
            combo.setMinimumHeight(34)
        self.monitor_toggle.setMinimumHeight(34)
        # A persistent bottom-right entry point: the guide is one click away on every
        # page, and a skipped first-run guide can always be reopened from here.
        self.help_button = QPushButton('？ 使い方ガイド', container)
        self.help_button.setObjectName('helpFab')
        self.help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_button.setToolTip('使い方ガイドを開きます。初回起動時にも自動で開きます。')
        self.help_button.setMinimumHeight(40)
        self.help_button.clicked.connect(self._show_tutorial)
        self.sidebar.help_clicked.connect(self._show_tutorial)
        self.help_button.show()
        self._guide_widget = None
        self._guide_open = False
        self._guide_dialog = None
        self._guide_highlight_timer = QTimer(self)
        self._guide_highlight_timer.setSingleShot(True)
        self._guide_highlight_timer.setInterval(2800)
        self._guide_highlight_timer.timeout.connect(self._clear_highlight)
        self._show_page('home')
        QTimer.singleShot(400, self._maybe_show_tutorial)

    def resizeEvent(self, event):
        """Keep the guide button pinned to the bottom-right corner of the window."""
        super().resizeEvent(event)
        self._place_help_button()

    def showEvent(self, event):
        super().showEvent(event)
        self._place_help_button()

    def _place_help_button(self):
        button = getattr(self, 'help_button', None)
        container = self.centralWidget()
        if button is None or container is None:
            return
        margin = 22
        button.adjustSize()
        size = button.size()
        button.move(max(0, container.width()-size.width()-margin),
                    max(0, container.height()-size.height()-margin))
        button.raise_()

    def _step_card(self, number, title, hint=''):
        """A numbered card used by the home page's three-step flow."""
        if not hasattr(self, 'home_step_cards'):
            self.home_step_cards = []
        card = QFrame()
        card.setObjectName('stepCard')
        column = QVBoxLayout(card)
        column.setContentsMargins(16, 12, 16, 14)
        column.setSpacing(8)
        header = QHBoxLayout()
        badge = QLabel(number)
        badge.setObjectName('stepNumber')
        heading = QLabel(title)
        heading.setObjectName('stepTitle')
        header.addWidget(badge)
        header.addWidget(heading)
        header.addStretch(1)
        column.addLayout(header)
        if hint:
            note = QLabel(hint)
            note.setObjectName('muted')
            note.setWordWrap(True)
            column.addWidget(note)
        self.home_step_cards.append((number, card))
        return card, column

    def _build_pitch_row(self, layout):
        """The pitch control, placed with the Female DSP controls it belongs to.

        MeanVC2 fixes the pitch shift per profile, so this slider drives the Female DSP
        path only. The reason is stated inline rather than left to a tooltip, because a
        disabled control with no explanation reads as a broken app.
        """
        row = QFrame()
        row.setObjectName('pitchRow')
        self.pitch_row = row
        column = QVBoxLayout(row)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(12)
        label = QLabel('ピッチ')
        label.setObjectName('muted')
        row_layout.addWidget(label)
        self.pitch_display = QSlider(Qt.Orientation.Horizontal)
        self.pitch_display.setMinimum(-12)
        self.pitch_display.setMaximum(12)
        self.pitch_display.setValue(0)
        self.pitch_display.setEnabled(False)
        self.pitch_value = QLabel('+0.0 st')
        self.pitch_value.setMinimumWidth(64)
        self.pitch_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        row_layout.addWidget(self.pitch_display, 1)
        row_layout.addWidget(self.pitch_value)
        column.addLayout(row_layout)
        self.pitch_note = QLabel('')
        self.pitch_note.setObjectName('muted')
        column.addWidget(self.pitch_note)
        self.pitch_display.valueChanged.connect(self._rail_pitch_changed)
        layout.addWidget(row)
        self._update_pitch_note()

    def _update_pitch_note(self):
        """Say why the pitch control is unavailable, in the current mode."""
        mode = self.mode.currentData() if hasattr(self, 'mode') else None
        if mode == 'female_dsp':
            self.pitch_note.setText('Female DSP のピッチです。声の高さを上下します。')
            self.pitch_note.setVisible(True)
            return
        if mode == 'ai_voice':
            self.pitch_note.setText('AI Voice ではピッチは声ごとに決まっています。'
                                    '「声ライブラリ」で別の声を選ぶと変わります。')
        else:
            self.pitch_note.setText('ピッチ変更は Female DSP のときだけ使えます。')
        self.pitch_note.setVisible(True)

    def _build_output_card(self, layout):
        card = QFrame()
        card.setObjectName('outputCard')
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(18, 14, 18, 14)
        card_layout.setSpacing(14)
        glyph = QLabel('🔊')
        glyph.setObjectName('presetGlyph')
        glyph.setFixedSize(44, 44)
        card_layout.addWidget(glyph)
        text = QVBoxLayout()
        text.setSpacing(3)
        title = QLabel('出力音声')
        title.setObjectName('presetName')
        self.output_card_text = QLabel('停止中')
        self.output_card_text.setObjectName('presetDetail')
        self.output_card_text.setWordWrap(True)
        text.addWidget(title)
        text.addWidget(self.output_card_text)
        card_layout.addLayout(text, 1)
        self.preview_button = QPushButton('変換を試聴')
        self.preview_button.clicked.connect(self._preview_converted)
        self.preview_button.setEnabled(False)
        self.preview_button.setToolTip('入力した音声ファイルを、この声で実際に変換して再生します。'
                                       '停止中にしか使えません。')
        card_layout.addWidget(self.preview_button)
        self.preview_note = QLabel('')
        self.preview_note.setObjectName('muted')
        self.preview_note.setWordWrap(True)
        self.preview_note.setVisible(False)
        card_layout.addWidget(self.preview_note, 1)
        layout.addWidget(card)

    def _rail_pitch_changed(self, value):
        self.pitch_value.setText('%+.1f st' % (value/4.0))
        if self.pitch_display.isEnabled():
            self.pitch.setValue(round(value*2.5))
            self._publish_dsp()

    def _show_tutorial(self):
        """Open the guide; it walks the real pages and highlights each control.

        The guide is a small tool window rather than a modal dialog, so the reader can
        try the highlighted control while the explanation is still on screen, and a
        headless render can never block on it.
        """
        if getattr(self, '_guide_open', False):
            return
        from PySide6.QtCore import Qt as _Qt
        from .tutorial import TutorialDialog
        self._guide_open = True
        dialog = TutorialDialog(self, on_page=self._guide_page)
        dialog.setWindowFlag(_Qt.WindowType.Tool)
        dialog.setModal(False)
        dialog.finished.connect(self._guide_finished)
        self._guide_dialog = dialog
        dialog.show()

    def _guide_finished(self, _result=0):
        """Read the reader's choice once the guide closes, then clean up."""
        dialog, self._guide_dialog = getattr(self, '_guide_dialog', None), None
        self._guide_open = False
        self._clear_highlight()
        if dialog is None:
            return
        hide = dialog.hide_next_time
        dialog.deleteLater()
        if hide and not self.settings.tutorial_seen:
            self.settings = replace(self.settings, tutorial_seen=True)
            self._save_settings()

    def _guide_page(self, spec):
        """Follow the guide: show the page it describes and mark the control."""
        page = spec.get('page')
        if page:
            self._show_page(page)
        target = spec.get('target')
        widget = getattr(self, target, None) if target else None
        self._highlight_widget(widget)

    def _highlight_widget(self, widget):
        """Outline one control for as long as the guide is talking about it."""
        self._clear_highlight()
        if widget is None:
            return
        self._guide_widget = widget
        widget.setProperty('guide', True)
        self._repolish(widget)
        area = getattr(self, 'centre_scroll', None)
        if area is not None:
            area.ensureWidgetVisible(widget)
        self._guide_highlight_timer.start()

    def _clear_highlight(self, *_):
        widget = getattr(self, '_guide_widget', None)
        self._guide_widget = None
        if widget is None:
            return
        widget.setProperty('guide', False)
        self._repolish(widget)

    @staticmethod
    def _repolish(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _maybe_show_tutorial(self):
        # Automated runs must never block on the modal guide.
        if self._closing or self.settings.tutorial_seen or "PYTEST_CURRENT_TEST" in os.environ:
            return
        self._show_tutorial()

    def _toggle_transport(self):
        if self.controller.engine.running:
            self._stop()
        else:
            self._start()

    def _show_page(self, key):
        page = self.pages.get(key)
        if page is None:
            return
        self._active_page = key
        for name, (widget, _layout) in self.pages.items():
            widget.setVisible(name == key)
        if self.sidebar.buttons.get(key) and not self.sidebar.buttons[key].isChecked():
            self.sidebar.select(key)

    def _preview_converted(self):
        """Convert the selected voice on a local WAV and play the result.

        Conversion runs in the worker process; this only starts it and plays the file
        it returns. Nothing is sent anywhere and no capture device is opened.
        """
        if self._preview_busy or self._closing:
            return
        if self.controller.engine.running or self._pending:
            self.preview_note.setText('変換中は試聴できません。Stopしてから実行してください。')
            self.preview_note.setVisible(True)
            return
        if self.mode.currentData() != 'ai_voice':
            self.preview_note.setText('AI Voiceを選んでから試聴できます。')
            self.preview_note.setVisible(True)
            return
        try:
            preview_params = self._ai_parameters()
        except ValueError as error:
            self.preview_note.setText(str(error))
            self.preview_note.setVisible(True)
            return
        source, _ = QFileDialog.getOpenFileName(self, '試聴する音声を選ぶ', '',
                                                 '音声ファイル (*.wav *.mp3 *.m4a *.flac *.ogg)')
        if not source:
            return
        self._preview_busy = True
        self.preview_button.setEnabled(False)
        self.preview_button.setText('変換しています…')
        self.preview_note.setVisible(True)
        self.preview_note.setText('変換ワーカーで処理しています。数秒〜数十秒かかります…')
        self.preview_source = source
        model = preview_params.model
        params = preview_params
        enhancer = params.enhancer or 'none'
        self.files.submit('preview', lambda: self._run_preview(
            source, model, enhancer, params.lavasr_denoise, params.experiment,
            params.tune_sib_db, params.tune_cons_db, params.tune_caps,
            params.tune_floor_db, params.tune_excess_db, params.tune_mid,
            params.tune_match, params.tune_ptrans, params.tune_pcap,
            params.tune_combined, params.tune_level_db))

    def _run_preview(self, source, model, enhancer, denoise=False, experiment='none',
                     tune_sib_db=3.0, tune_cons_db=3.0, tune_caps=1.0,
                     tune_floor_db=3.0, tune_excess_db=9.0, tune_mid=0.8,
                     tune_match=0.25, tune_ptrans=0.20, tune_pcap=1.0,
                     tune_combined=True, tune_level_db=-20.0):
        from . import preview as preview_module
        try:
            path, stats = preview_module.convert(source, model, enhancer=enhancer,
                                                  denoise=denoise, experiment=experiment,
                                                  tune_sib_db=tune_sib_db, tune_cons_db=tune_cons_db,
                                                  tune_caps=tune_caps, tune_floor_db=tune_floor_db,
                                                  tune_excess_db=tune_excess_db, tune_mid=tune_mid,
                                                  tune_match=tune_match, tune_ptrans=tune_ptrans,
                                                  tune_pcap=tune_pcap, tune_combined=tune_combined,
                                                  tune_level_db=tune_level_db)
            summary = preview_module.describe(stats)
        except Exception as error:
            self.files.submit('preview_done', lambda: self._preview_failed(error))
            return
        self.files.submit('preview_done', lambda: self._preview_ready(path, summary))

    def _preview_failed(self, error):
        self._preview_busy = False
        self.preview_button.setText('変換を試聴')
        self.preview_button.setEnabled(self.router is not None
                                       and self.mode.currentData() == 'ai_voice'
                                       and not self.controller.engine.running)
        self.preview_note.setText('試聴に失敗しました: ' + str(error))
        self.preview_note.setVisible(True)

    def _preview_ready(self, path, summary):
        self._preview_busy = False
        self.preview_button.setText('再生を停止')
        self.preview_button.setEnabled(True)
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        if self.preview_player is None:
            self.preview_output = QAudioOutput(self)
            self.preview_output.setVolume(0.9)
            self.preview_player = QMediaPlayer(self)
            self.preview_player.setAudioOutput(self.preview_output)
            self.preview_player.playbackStateChanged.connect(self._preview_state_changed)
            self.preview_player.errorOccurred.connect(
                lambda _error, message: self._set_preview_note('再生エラー: '+message))
        self.preview_player.setSource(QUrl.fromLocalFile(str(path)))
        self.preview_player.play()
        self._set_preview_note(('変換完了 · '+summary) if summary else '変換完了')

    def _preview_state_changed(self, _state):
        from PySide6.QtMultimedia import QMediaPlayer
        if self.preview_player is None:
            return
        playing = self.preview_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self.preview_button.setText('再生を停止' if playing else '変換を試聴')

    def _set_preview_note(self, text):
        self.preview_note.setText(text)
        self.preview_note.setVisible(True)

    def _fit_page(self, selected):
        # Hidden pages must not take space from the visible one; only the active page
        # keeps its own preferred height, and the column scrolls when it does not fit.
        for name, (widget, _layout) in self.pages.items():
            policy = QSizePolicy.Policy.Preferred if name == selected else QSizePolicy.Policy.Ignored
            widget.setSizePolicy(policy, policy)
        page = self.pages.get(selected)
        if page is not None:
            page[0].updateGeometry()

    def _build_prosody_ui(self,layout):
        p=self.settings.prosody_parameters()
        self.prosody_on=QCheckBox('Auto Intonation · AI Voiceのみ')
        self.prosody_on.setChecked(p.enabled)
        layout.addWidget(self.prosody_on)
        self.prosody_engine=QComboBox()
        for label,value in [('Off','off'),('Rule v2','rule_v2'),('Text-Aware v3 · β','text_v3')]:
            self.prosody_engine.addItem(label,value)
        self.prosody_engine.setCurrentIndex(self.prosody_engine.findData(p.engine))
        layout.addWidget(self.prosody_engine)
        self.prosody_preset=QComboBox()
        self.prosody_preset.addItems([*PROSODY_PRESETS,'Custom'])
        self.prosody_preset.setCurrentText(p.preset)
        form=QFormLayout(); form.addRow('Preset',self.prosody_preset); layout.addLayout(form)
        self.anime_amount,self.anime_amount_label=self._slider(layout,'Anime Amount',0,100,round(p.amount))
        self.prosody_range,self.prosody_range_label=self._slider(layout,'Pitch Range',0,100,round(p.pitch_range))
        self.prosody_energy,self.prosody_energy_label=self._slider(layout,'Energy',0,100,round(p.energy))
        self.ending_lift=QCheckBox('Ending Emphasis · 強調設定（方向は元の声から）')
        self.ending_lift.setChecked(p.ending_emphasis); layout.addWidget(self.ending_lift)
        self.prosody_status=QLabel('Prosody Disabled')
        self.prosody_status.setWordWrap(True)
        self.prosody_status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.prosody_status)
        note=QLabel('元の抑揚に補正を加えます。Text-Awareはローカル日本語ASRと文字列ルールを使うβ機能です。\n認識は発話途中から反映されます。音声は解析を待ちません。')
        note.setWordWrap(True); layout.addWidget(note)

    def _build_prosody_advanced(self,layout):
        """Prosody detail controls, wrapped so the whole block hides as one unit."""
        outer = layout
        box = QWidget()
        self.prosody_advanced_box = box
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        p=self.settings.prosody_parameters()
        form=QFormLayout()
        self.prosody_advanced={}
        labels={'range_expansion':'Range expansion','rise_boost':'Rise boost','fall_boost':'Fall boost',
            'ending_strength':'Ending emphasis','history_ms':'Trajectory history (ms)','onset_lift':'Onset lift (st)',
            'energy_dynamics':'Energy dynamics','min_pitch':'ΔPitch minimum (st)','max_pitch':'ΔPitch maximum (st)',
            'max_gain_db':'Dynamic gain limit (dB)','attack_ms':'Pitch attack (ms)','release_ms':'Pitch release (ms)',
            'baseline_ms':'Baseline history (ms)','confidence':'YIN periodicity threshold','min_silence_ms':'Onset minimum silence (ms)'}
        for name,low,high in ProsodyParameters.bounds():
            if name not in labels:
                continue
            widget=QDoubleSpinBox(); widget.setRange(low,high)
            widget.setDecimals(2 if high<=2 else 0); widget.setSingleStep(.05 if high<=2 else 5)
            widget.setValue(getattr(p,name)); form.addRow(labels[name],widget)
            self.prosody_advanced[name]=widget
            widget.valueChanged.connect(self._prosody_advanced_changed)
        self.prosody_update=QComboBox()
        for ms in (20,30,40,50):
            self.prosody_update.addItem(f'{ms} ms',ms)
        self.prosody_update.setCurrentIndex(self.prosody_update.findData(int(p.update_ms)))
        form.addRow('Pitch/control update',self.prosody_update); layout.addLayout(form)
        self.prosody_details=QLabel('F0 analysis: FFT YIN / trailing 80 ms · FCPEはOfflineのみ')
        self.prosody_details.setWordWrap(True); layout.addWidget(self.prosody_details)
        self.prosody_update.currentIndexChanged.connect(self._prosody_changed)
        self.prosody_engine.currentIndexChanged.connect(self._prosody_changed)
        self.text_influence=QDoubleSpinBox(); self.text_influence.setRange(0,100); self.text_influence.setValue(p.text_influence)
        text_form=QFormLayout(); text_form.addRow('Text influence (%)',self.text_influence)
        self.asr_threads=QSpinBox(); self.asr_threads.setRange(1,4); self.asr_threads.setValue(p.asr_threads)
        text_form.addRow('ASR CPU threads',self.asr_threads)
        self.asr_update=QComboBox(); self.asr_window=QComboBox()
        for ms in (300,600,900): self.asr_update.addItem(f'{ms} ms',ms)
        for ms in (250,400,600,800,1000,1500,2000): self.asr_window.addItem(f'{ms} ms',ms)
        self.asr_update.setCurrentIndex(self.asr_update.findData(p.asr_update_ms))
        self.asr_window.setCurrentIndex(self.asr_window.findData(p.asr_window_ms))
        text_form.addRow('ASR update (β)',self.asr_update); text_form.addRow('ASR window (β)',self.asr_window)
        self.text_strategy=QComboBox()
        for label,value in [('v0.9 baseline','legacy'),('Direct bias','direct'),('Rule modulation','modulation'),('Calibrated Hybrid','hybrid'),('Text Event shaping β','events')]:
            self.text_strategy.addItem(label,value)
        self.text_strategy.setCurrentIndex(self.text_strategy.findData(p.text_strategy))
        text_form.addRow('Text calibration',self.text_strategy)
        layout.addLayout(text_form)
        outer.addWidget(box)
        self.log_transcripts=QCheckBox('Log ASR Transcript · 発話内容を保存（通常OFF）')
        self.log_transcripts.setChecked(p.transcript_logging); layout.addWidget(self.log_transcripts)
        self.asr_restart=QPushButton('ASR Workerを再起動')
        self.asr_restart.clicked.connect(self._restart_asr); layout.addWidget(self.asr_restart)
        self.asr_details=QLabel('Text Prosody Disabled'); self.asr_details.setWordWrap(True)
        self.asr_details.setTextFormat(Qt.TextFormat.PlainText); layout.addWidget(self.asr_details)
        self.text_influence.valueChanged.connect(self._prosody_changed)
        self.asr_threads.valueChanged.connect(self._prosody_changed)
        self.asr_update.currentIndexChanged.connect(self._prosody_changed)
        self.asr_window.currentIndexChanged.connect(self._prosody_changed)
        self.text_strategy.currentIndexChanged.connect(self._prosody_changed)
        self.log_transcripts.toggled.connect(self._prosody_changed)
        self.prosody_preset.currentTextChanged.connect(self._prosody_preset_changed)
        for widget in (self.prosody_on,self.ending_lift):
            widget.toggled.connect(self._prosody_changed)
        for widget in (self.anime_amount,self.prosody_range,self.prosody_energy):
            widget.valueChanged.connect(self._prosody_changed)
        self._prosody_changed()

    def _prosody_parameters(self):
        values={name:widget.value() for name,widget in self.prosody_advanced.items()}
        return ProsodyParameters(enabled=self.prosody_on.isChecked() and not is_meanvc2(self.ai_model.currentData()),preset=self.prosody_preset.currentText(),
            amount=self.anime_amount.value(),pitch_range=self.prosody_range.value(),energy=self.prosody_energy.value(),
            ending_emphasis=self.ending_lift.isChecked(),update_ms=self.prosody_update.currentData(),
            engine=self.prosody_engine.currentData(),text_influence=self.text_influence.value(),
            transcript_logging=self.log_transcripts.isChecked(),asr_threads=self.asr_threads.value(),
            asr_update_ms=self.asr_update.currentData(),asr_window_ms=self.asr_window.currentData(),
            text_strategy=self.text_strategy.currentData(),**values)

    def _restart_asr(self):
        if self.router:
            asr=self.router.ai.bridge.prosody.asr
            asr.restart()

    def _prosody_changed(self,*_):
        if not hasattr(self,'prosody_update'):
            return
        p=self._prosody_parameters()
        for widget,value in ((self.anime_amount_label,p.amount),(self.prosody_range_label,p.pitch_range),
                             (self.prosody_energy_label,p.energy)):
            widget.setText(f'{value:.0f}%')
        if self.router:
            self.router.ai.bridge.prosody.configure(p)

    def _prosody_advanced_changed(self,*_):
        self.prosody_preset.blockSignals(True); self.prosody_preset.setCurrentText('Custom'); self.prosody_preset.blockSignals(False)
        self._prosody_changed()

    def _prosody_preset_changed(self,name):
        if name in PROSODY_PRESETS:
            p=load_prosody_preset(name,self._prosody_parameters())
            for key,widget in self.prosody_advanced.items():
                widget.blockSignals(True); widget.setValue(getattr(p,key)); widget.blockSignals(False)
            log.info('Prosody preset change: %s',name)
            self._prosody_changed()

    def _build_dsp_ui(self, layout):
        """Female DSP controls; the page states when it is not the active path."""
        self.dsp_notice = QLabel('')
        self.dsp_notice.setWordWrap(True)
        self.dsp_notice.setObjectName('warningText')
        layout.addWidget(self.dsp_notice)
        self.dsp_switch = QPushButton('Female DSP に切り替えて使う')
        self.dsp_switch.setToolTip('軽い加工（CPUが軽い）に切り替えます。ピッチ・フォルマント・'
                                   '明るさ・乾湿をここで調整できます。')
        self.dsp_switch.clicked.connect(
            lambda: self.mode.setCurrentIndex(self.mode.findData('female_dsp')))
        layout.addWidget(self.dsp_switch)
        self.preset = QComboBox()
        for name in (*PRESETS, "Custom"):
            self.preset.addItem(name)
        self.quality = QComboBox()
        self.quality.addItem("Low Latency · 2048", "low_latency")
        self.quality.addItem("Balanced · 4096", "balanced")
        self.quality.setToolTip("解析窓の選択はStop後に変更。Pitch / Formant / Presetは実行中も変更可能。")
        for name, combo in (("Preset", self.preset), ("DSP Quality", self.quality)):
            combo.setMinimumHeight(34)
            layout.addWidget(self._labelled_row(name, combo))
        self.pitch, self.pitch_label = self._slider(layout, "Pitch", -120, 120, 30)
        self.formant, self.formant_label = self._slider(layout, "Formant", -60, 60, 20)
        self.brightness, self.brightness_label = self._slider(layout, "Brightness", 0, 100, 60)
        self.wet, self.wet_label = self._slider(layout, "Effect · Dry / Wet", 0, 100, 100)
        row = QHBoxLayout()
        self.low_cut = QCheckBox("Low Cut · 90 Hz")
        self.limiter = QCheckBox("Limiter · ON")
        row.addWidget(self.low_cut)
        row.addWidget(self.limiter)
        layout.addLayout(row)
        self.dsp_note = QLabel("PitchとFormantは独立。Anime Testも抑揚は変更しません。")
        self.dsp_note.setWordWrap(True)
        self.dsp_note.setObjectName("muted")
        layout.addWidget(self.dsp_note)
        self._build_pitch_row(layout)
        self._apply_dsp_widgets(self.settings.dsp_parameters(), self.settings.preset)
        for slider in (self.pitch, self.formant, self.brightness, self.wet):
            slider.valueChanged.connect(self._dsp_changed)
        for checkbox in (self.low_cut, self.limiter):
            checkbox.toggled.connect(self._dsp_changed)
        self.mode.currentIndexChanged.connect(self._dsp_changed)
        self.quality.currentIndexChanged.connect(self._dsp_changed)
        self.preset.currentTextChanged.connect(self._preset_changed)
        if self.dsp is None:
            for widget in (self.mode, self.preset, self.quality, self.pitch, self.formant,
                           self.brightness, self.wet, self.low_cut, self.limiter):
                widget.setEnabled(False)

    def _dsp_widgets(self):
        return (self.mode, self.preset, self.quality, self.pitch, self.formant,
                self.brightness, self.wet, self.low_cut, self.limiter)

    def _apply_dsp_widgets(self, parameters, preset_name):
        for widget in self._dsp_widgets():
            widget.blockSignals(True)
        self.mode.setCurrentIndex(self.mode.findData(parameters.mode))
        self.quality.setCurrentIndex(self.quality.findData(parameters.quality))
        self.preset.setCurrentText(preset_name)
        self.pitch.setValue(round(parameters.pitch * 10))
        self.formant.setValue(round(parameters.formant * 10))
        self.brightness.setValue(round(parameters.brightness))
        self.wet.setValue(round(parameters.wet * 100))
        self.low_cut.setChecked(parameters.low_cut)
        self.limiter.setChecked(parameters.limiter)
        for widget in self._dsp_widgets():
            widget.blockSignals(False)
        self._publish_dsp()

    def _dsp_parameters(self):
        return DSPParameters(mode='female_dsp' if self.mode.currentData() == 'ai_voice' else self.mode.currentData(), pitch=self.pitch.value()/10,
            formant=self.formant.value()/10, brightness=self.brightness.value(),
            low_cut=self.low_cut.isChecked(), limiter=self.limiter.isChecked(),
            wet=self.wet.value()/100, quality=self.quality.currentData())

    def _publish_dsp(self):
        parameters = self._dsp_parameters()
        self.pitch_label.setText(f"{parameters.pitch:+.1f} st")
        self.formant_label.setText(f"{parameters.formant:+.1f} st")
        self.brightness_label.setText(f"{parameters.brightness:.0f}% · 50% Neutral")
        self.wet_label.setText(f"{parameters.wet * 100:.0f}%")
        self.limiter.setText("Limiter · ON" if parameters.limiter else "Limiter · OFF（安全clampは有効）")
        ai_mode = self.mode.currentData() == 'ai_voice'
        if self.dsp:
            self.dsp.set_parameters(replace(parameters,mode='female_dsp') if self.router else parameters)
            self.controller.engine.diagnostics.note_gui_action()
        if self.router:
            self.router.select(self.mode.currentData())
        for widget in (self.pitch, self.formant, self.brightness, self.low_cut,
                       self.limiter, self.wet, self.preset):
            widget.setEnabled(self.dsp is not None and not ai_mode)

    def _build_ai_ui(self, layout):
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        heading = QLabel('変換したい声')
        heading.setObjectName('sectionTitle')
        layout.addWidget(heading)
        self.voice_hint = QLabel('声を選び、マイクと出力先を確認して Start。')
        self.voice_hint.setWordWrap(True)
        self.voice_hint.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.voice_hint)
        form = QFormLayout()
        self.ai_model = QComboBox()
        self.ai_model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.ai_model.setMinimumContentsLength(16)
        from src.vc.models import VOICE_PROFILES, DEFAULT_VOICE_ID
        ordered = ([DEFAULT_VOICE_ID] if DEFAULT_VOICE_ID in VOICE_PROFILES else [])
        ordered += [key for key in VOICE_PROFILES if key != DEFAULT_VOICE_ID]
        for key in ordered:
            self.ai_model.addItem(VOICE_PROFILES[key]['name'], key)
        self.ai_model.setCurrentIndex(self.ai_model.findData(self.settings.ai_model))
        self.ai_model.setMinimumHeight(42)
        layout.addWidget(self.ai_model)
        # The home page mirrors this list, so the voice can be changed where the flow
        # starts; both combos always show the same selection.
        self._sync_voice_combos()
        self.voice_quick.currentIndexChanged.connect(self._voice_quick_changed)
        self.ai_delivery=QComboBox()
        from src.runtime_paths import bundle_deliveries
        from src.vc.models import DEFAULT_DELIVERY, DELIVERY_MODES
        allowed_deliveries=bundle_deliveries()
        for key, spec in DELIVERY_MODES.items():
            if allowed_deliveries is not None and key not in allowed_deliveries:
                continue
            self.ai_delivery.addItem('%s（%s）' % (spec['label'], spec['detail']), key)
        # The saved value is the worker's delivery half, so match it back to the combo key.
        # Experimental variants share the utterance+LavaSR halves, so the experiment
        # name disambiguates them; a stale experiment falls back to the shipped route.
        saved=self.settings.ai_delivery
        wanted_experiment=getattr(self.settings,'ai_experiment','none')
        delivery=next((key for key,spec in DELIVERY_MODES.items()
                       if spec['delivery']==saved and spec.get('experiment','none')==wanted_experiment), None)
        if delivery is None:
            delivery=next((key for key,spec in DELIVERY_MODES.items()
                           if spec['delivery']==saved), None)
        if delivery is None:
            # A saved mode this build does not ship must not leave the combo empty.
            delivery=(DEFAULT_DELIVERY if allowed_deliveries is None
                       or DEFAULT_DELIVERY in allowed_deliveries else 'streaming')
        self.ai_delivery.setCurrentIndex(self.ai_delivery.findData(delivery))
        layout.addWidget(self.ai_delivery)
        self.delivery_note=QLabel('')
        self.delivery_note.setWordWrap(True)
        self.delivery_note.setMinimumHeight(1)
        layout.addWidget(self.delivery_note)
        self._update_delivery_note()
        self.finish_phrase=QPushButton('今の発話を変換')
        self.finish_phrase.clicked.connect(self._finish_current_utterance)
        layout.addWidget(self.finish_phrase)
        actions = QHBoxLayout()
        self.add_voice = QPushButton('＋ 音声ファイルから声を追加')
        self.add_voice.setObjectName('start')
        self.add_voice.clicked.connect(self._add_voice)
        self.preview_voice = QPushButton('Referenceを試聴')
        self.preview_voice.clicked.connect(self._preview_reference)
        actions.addWidget(self.add_voice, 1)
        actions.addWidget(self.preview_voice)
        layout.addLayout(actions)
        self.voice_note = QLabel('登録はローカルで完結。学習不要・元の音声ファイルはそのまま。')
        self.voice_note.setObjectName('muted')
        self.voice_note.setWordWrap(True)
        self.voice_note.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.voice_note)
        self.use_voice = QPushButton('選択した声を使う（AI Voice）')
        self.use_voice.clicked.connect(self._use_selected_voice)
        layout.addWidget(self.use_voice)
        self.ai_details = QWidget()
        details = QVBoxLayout(self.ai_details)
        self.ai_quality = QComboBox()
        for label, key in [('Low Latency · chunk 13 ms', 'low_latency'), ('Balanced · chunk 26 ms', 'balanced'), ('Stable · chunk 52 ms', 'stable')]:
            self.ai_quality.addItem(label, key)
        self.ai_quality.setToolTip('数値はAIの処理単位です。出力待ちやモデル遅延、Discordまでの実測遅延とは異なります。')
        self.ai_quality.setCurrentIndex(self.ai_quality.findData(self.settings.ai_quality))
        self.ai_threads = QComboBox()
        for count in range(1,5):
            self.ai_threads.addItem(str(count), count)
        self.ai_threads.setCurrentIndex(self.ai_threads.findData(self.settings.ai_threads))
        self.ai_threads.setToolTip('AIの並列数。2が速いPCが多く、4が速い場合もあります。「変換を試聴」で比べてください。')
        for name, combo in [('処理構成', self.ai_quality),('CPU threads',self.ai_threads)]:
            details.addWidget(self._labelled_row(name, combo))
        self.ai_brightness, self.ai_brightness_label = self._slider(details, 'AI Brightness', 0, 100, round(self.settings.ai_brightness))
        self.ai_low_cut = QCheckBox('Low Cut · 90 Hz')
        self.ai_limiter = QCheckBox('Limiter')
        self.ai_post_fx = QCheckBox('Light Post FX（Pitch/Formantなし）')
        for widget, value in [(self.ai_low_cut,self.settings.ai_low_cut), (self.ai_limiter,self.settings.ai_limiter), (self.ai_post_fx,self.settings.ai_post_fx)]:
            widget.setChecked(value)
            details.addWidget(widget)
        self.tune_header = QLabel('声の調整（比較用の自然寄せのみ）')
        self.tune_header.setWordWrap(True)
        details.addWidget(self.tune_header)
        self.tune_sib, self.tune_sib_label = self._slider(details, 'サ行抑え', 0, 60, round(self.settings.ai_tune_sib_db*10))
        self.tune_cons, self.tune_cons_label = self._slider(details, '子音明瞭', 0, 60, round(self.settings.ai_tune_cons_db*10))
        self.tune_caps, self.tune_caps_label = self._slider(details, '抑揚', 10, 30, round(self.settings.ai_tune_caps*20))
        self.tune_floor, self.tune_floor_label = self._slider(details, '床抑え', 0, 60, round(self.settings.ai_tune_floor_db*10))
        self.tune_excess, self.tune_excess_label = self._slider(details, 'ツヤ', 6, 48, round(self.settings.ai_tune_excess_db*2))
        self.tune_mid, self.tune_mid_label = self._slider(details, '中域ツヤ', 0, 100, round(self.settings.ai_tune_mid*100))
        self.tune_match, self.tune_match_label = self._slider(details, '抑揚追従', 0, 50, round(self.settings.ai_tune_match*100))
        self.tune_ptrans, self.tune_ptrans_label = self._slider(details, '抑揚ピッチ追従', 0, 50, round(self.settings.ai_tune_ptrans*100))
        self.tune_pcap, self.tune_pcap_label = self._slider(details, '抑揚ピッチ上限', 0, 40, round(self.settings.ai_tune_pcap*20))
        self.tune_level, self.tune_level_label = self._slider(details, '入力レベリング', -52, -28, round(self.settings.ai_tune_level_db*2))
        self.tune_combined = QCheckBox('抑揚ピッチ補完（combined）')
        self.tune_combined.setChecked(self.settings.ai_tune_combined)
        details.addWidget(self.tune_combined)
        self.tune_reset = QPushButton('声調整を初期値に戻す')
        self.tune_reset.clicked.connect(self._reset_tune)
        details.addWidget(self.tune_reset)
        self._tune_widgets = (self.tune_sib, self.tune_cons, self.tune_caps,
                               self.tune_floor, self.tune_excess, self.tune_mid,
                               self.tune_match, self.tune_ptrans, self.tune_pcap,
                               self.tune_level, self.tune_combined, self.tune_reset)
        self.ai_status = QLabel('AI Status: Not Loaded')
        self.ai_status.setWordWrap(True)
        self.ai_status.setTextFormat(Qt.TextFormat.PlainText)
        self.ai_natural = QPushButton('標準ボイスで推奨設定を適用')
        self.ai_natural.clicked.connect(self._apply_natural_voice)
        details.addWidget(self.ai_natural)
        self.ai_restart = QPushButton('AI Workerを再起動')
        self.ai_restart.clicked.connect(self._restart_ai)
        details.addWidget(self.ai_restart)
        self.ai_performance = QLabel('推論は別プロセス。Voice ModeでAI Voiceを選ぶとロードします。')
        self.ai_performance.setWordWrap(True)
        layout.addWidget(self.ai_status)
        layout.addWidget(self.ai_performance)
        note = QLabel('声の追加・切り替えは停止中に行います。\n新しい声の似具合は録音内容によって変わります。')
        note.setWordWrap(True)
        layout.addWidget(note)
        for combo in (self.ai_quality, self.ai_threads):
            combo.currentIndexChanged.connect(self._ai_structure_changed)
        self.ai_model.currentIndexChanged.connect(self._ai_model_changed)
        self.ai_delivery.currentIndexChanged.connect(self._ai_structure_changed)
        self.ai_brightness.valueChanged.connect(self._ai_fx_changed)
        for checkbox in (self.ai_low_cut,self.ai_limiter,self.ai_post_fx):
            checkbox.toggled.connect(self._ai_fx_changed)
        for slider in (self.tune_sib,self.tune_cons,self.tune_caps,self.tune_floor,self.tune_excess,
                       self.tune_mid,self.tune_match,self.tune_ptrans,self.tune_pcap,self.tune_level):
            slider.valueChanged.connect(self._tune_changed)
        self.tune_combined.toggled.connect(self._tune_changed)
        self.tune_sib_label.setText(f'{self.tune_sib.value()/10:.1f} dB')
        self.tune_cons_label.setText(f'{self.tune_cons.value()/10:.1f} dB')
        self.tune_caps_label.setText(f'{self.tune_caps.value()/20:.2f}倍')
        self.tune_floor_label.setText(f'{self.tune_floor.value()/10:.1f} dB')
        self.tune_excess_label.setText(f'{self.tune_excess.value()/2:.1f} dB')
        self.tune_mid_label.setText(f'{self.tune_mid.value()/100:.2f}')
        self.tune_match_label.setText(f'{self.tune_match.value()/100:.2f}')
        self.tune_ptrans_label.setText(f'{self.tune_ptrans.value()/100:.2f}')
        self.tune_pcap_label.setText(f'{self.tune_pcap.value()/20:.2f} st')
        self.tune_level_label.setText(f'{self.tune_level.value()/2:.1f} dB')
        self._ai_fx_changed()
        self._update_voice_info()
        if not self.router:
            for widget in (self.ai_model,self.ai_quality,self.ai_threads,self.ai_brightness,self.ai_low_cut,self.ai_limiter,self.ai_post_fx):
                widget.setEnabled(False)

    def _ai_parameters(self):
        # The delivery combo owns both halves of the route, so the worker never receives
        # a delivery and an enhancer that disagree.
        delivery=DELIVERY_MODES.get(self.ai_delivery.currentData(),DELIVERY_MODES[DEFAULT_DELIVERY])
        from src.vc.models import default_voice_id, VOICE_PROFILES
        model = self.ai_model.currentData()
        if model not in VOICE_PROFILES:
            model = default_voice_id() or next(iter(VOICE_PROFILES), None)
        if model is None:
            raise ValueError('利用可能なAI音声がありません。声ライブラリから声を登録してください。')
        return AIParameters(model=model,quality=self.ai_quality.currentData(), threads=self.ai_threads.currentData(),
                            brightness=self.ai_brightness.value(), low_cut=self.ai_low_cut.isChecked(),
                            limiter=self.ai_limiter.isChecked(), post_fx=self.ai_post_fx.isChecked(),
                            output_wait_ms=self.ai_wait.value() if hasattr(self,'ai_wait') else self.settings.ai_output_wait_ms,
                            crossfade_ms=self.ai_fade.value() if hasattr(self,'ai_fade') else self.settings.ai_crossfade_ms,
                            pitch=self.ai_pitch.value()/8 if hasattr(self,'ai_pitch') else self.settings.ai_pitch,
                            delivery=delivery['delivery'], enhancer=delivery['enhancer'],
                            lavasr_denoise=delivery['enhancer']=='lavasr',
                            experiment=delivery.get('experiment','none'),
                            tune_sib_db=self.tune_sib.value()/10, tune_cons_db=self.tune_cons.value()/10,
                            tune_caps=self.tune_caps.value()/20, tune_floor_db=self.tune_floor.value()/10,
                            tune_excess_db=self.tune_excess.value()/2, tune_mid=self.tune_mid.value()/100,
                            tune_match=self.tune_match.value()/100, tune_ptrans=self.tune_ptrans.value()/100,
                            tune_pcap=self.tune_pcap.value()/20, tune_combined=self.tune_combined.isChecked(),
                            tune_level_db=self.tune_level.value()/2)

    def _utterance_active(self):
        try:
            return self._ai_parameters().delivery == 'utterance'
        except ValueError:
            return False

    def _finish_current_utterance(self):
        if self.router and self.controller.engine.running:
            self.router.ai.bridge.finish_utterance.set()
            self.router.ai.bridge.input_ready.signal()

    def _ai_fx_changed(self, *_):
        try:
            parameters = self._ai_parameters()
        except ValueError:
            return
        self.ai_brightness_label.setText(f'{parameters.brightness:.0f}% · 50% Neutral')
        if hasattr(self,'ai_pitch_label'):
            from src.vc.models import profile
            self.ai_pitch_label.setText(f'{parameters.pitch:+.3f} st · 推奨 {profile(parameters.model)["pitch"]:+g}')
        if self.router:
            # Only lightweight immutable FX parameters change here.
            from dataclasses import replace
            current = self.router.ai.bridge.parameters
            self.router.ai.bridge.parameters = replace(current, brightness=parameters.brightness,
                low_cut=parameters.low_cut, limiter=parameters.limiter, post_fx=parameters.post_fx,pitch=parameters.pitch)

    def _tune_changed(self, *_):
        self.tune_sib_label.setText(f'{self.tune_sib.value()/10:.1f} dB')
        self.tune_cons_label.setText(f'{self.tune_cons.value()/10:.1f} dB')
        self.tune_caps_label.setText(f'{self.tune_caps.value()/20:.2f}倍')
        self.tune_floor_label.setText(f'{self.tune_floor.value()/10:.1f} dB')
        self.tune_excess_label.setText(f'{self.tune_excess.value()/2:.1f} dB')
        self.tune_mid_label.setText(f'{self.tune_mid.value()/100:.2f}')
        self.tune_match_label.setText(f'{self.tune_match.value()/100:.2f}')
        self.tune_ptrans_label.setText(f'{self.tune_ptrans.value()/100:.2f}')
        self.tune_pcap_label.setText(f'{self.tune_pcap.value()/20:.2f} st')
        self.tune_level_label.setText(f'{self.tune_level.value()/2:.1f} dB')
        self._ai_structure_changed()

    def _reset_tune(self):
        if self._pending or self.controller.engine.running:
            return
        for slider, default in ((self.tune_sib, 30), (self.tune_cons, 30), (self.tune_caps, 20),
                                (self.tune_floor, 30), (self.tune_excess, 18), (self.tune_mid, 80),
                                (self.tune_match, 25), (self.tune_ptrans, 20), (self.tune_pcap, 20),
                                (self.tune_level, -40)):
            slider.blockSignals(True)
            slider.setValue(default)
            slider.blockSignals(False)
        self.tune_combined.blockSignals(True)
        self.tune_combined.setChecked(True)
        self.tune_combined.blockSignals(False)
        self._tune_changed()

    def _ai_structure_changed(self, *_):
        if self.router:
            self.ai_wait_label.setText(f'{self.ai_wait.value()} ms · pre-roll（実測遅延ではありません）')
            self.ai_fade_label.setText(f'{self.ai_fade.value()} ms')
            if is_meanvc2(self.ai_model.currentData()):
                self.ai_wait_label.setText(self._meanvc_preroll_text())
                self._update_meanvc_controls()
            try:
                parameters = self._ai_parameters()
            except ValueError:
                self._set_controls(False)
                return
            self._pending = True
            self._pending_request = self.controller.configure_ai(parameters)
            self._set_controls(False)

    def _restart_ai(self):
        if self.router and not self._pending:
            self._pending = True
            self._pending_request = self.controller.restart_ai()
            self._set_controls(False)

    def _mode_changed(self, *_):
        """Keep the pitch explanation in step with the selected mode."""
        if hasattr(self, 'pitch_note'):
            self._update_pitch_note()
        self._update_orb(self.controller.snapshot)
        if hasattr(self, 'output_card_text'):
            self.output_card_text.setText('現在の設定: ' + self._describe_output()
                                           if self.controller.engine.running
                                           else '停止中 · Startで変換を開始します')

    def _sync_voice_combos(self, *_):
        """Mirror the voice list (order included) into the home-page quick picker."""
        if not hasattr(self, 'voice_quick'):
            return
        self.voice_quick.blockSignals(True)
        self.voice_quick.clear()
        for index in range(self.ai_model.count()):
            self.voice_quick.addItem(self.ai_model.itemText(index), self.ai_model.itemData(index))
        self.voice_quick.setCurrentIndex(self.ai_model.currentIndex())
        self.voice_quick.blockSignals(False)

    def _voice_quick_changed(self, index):
        """The quick picker drives the library combo rather than duplicating state."""
        if index < 0:
            return
        key = self.voice_quick.itemData(index)
        target = self.ai_model.findData(key)
        if target >= 0 and target != self.ai_model.currentIndex():
            self.ai_model.setCurrentIndex(target)

    def _ai_model_changed(self, *_):
        self._sync_voice_combos()
        if not self._pending and not self.controller.engine.running:
            self._stop_preview()
            self._apply_recommended()
            self._update_meanvc_controls()
            self._update_voice_info()
            self.startup_diagnostic = None
            model = self.ai_model.currentData()
            self._diagnostic_model = model
            self.files.submit('diagnostics', lambda: self._diagnose_voice(model))

    def _diagnose_voice(self, model):
        result = startup_diagnostics(Path(__file__).resolve().parents[2], model)
        if self._diagnostic_model == model:
            self.startup_diagnostic = result

    def _update_voice_info(self):
        try:
            from src.vc.models import profile
            selected = profile(self.ai_model.currentData())
        except ValueError:
            self.preview_voice.setEnabled(False)
            self.voice_hint.setText('声がありません。声ライブラリから声を登録してください。')
            return
        reference = Path(__file__).resolve().parents[2]/'models'/selected['folder']/'reference.wav'
        self.preview_voice.setEnabled(reference.is_file())
        from src.vc.voice_library import metadata
        if selected.get('standard'):
            info = metadata(self.ai_model.currentData())
            self.voice_hint.setText(
                f"アプリ標準の声「{selected['name']}」を選択中\n"
                f"Reference {info['reference_seconds']:.1f}秒 · 最初からこの声で使えます。"
                "自分の声に変えるには「＋ 音声ファイルから声を追加」から。")
            self.voice_note.setText('この声はアプリに同梱されています。変更・削除はできません。')
        elif selected.get('user_voice'):
            info = metadata(self.ai_model.currentData())
            self.voice_hint.setText(
                f"登録した声「{info['name']}」を選択中\n"
                f"Reference {info['reference_seconds']:.1f}秒 · 似具合は録音内容で変わります。")
            self.voice_note.setText('登録はローカルで完結。学習不要・元の音声ファイルはそのまま。')
        else:
            self.voice_hint.setText(selected['name']+'\n今までの声はそのまま選べます。')

    def _add_voice(self):
        if self._pending or self._closing or self.controller.engine.running or self.voice_dialog:
            return
        from .voice_registration import VoiceRegistrationDialog
        self._stop_preview()
        self._pending = True
        self._pending_request = self.controller.stop()
        self._set_controls(False)
        # Existing registered voices are offered as destinations for adding more audio.
        # The worker resolves the folder id, so the display label travels beside it.
        # The shipped standard voice is not offered: it is the app's own voice and is
        # not the listener's registration to extend.
        from src.vc.voice_library import is_standard
        voices = [(self.ai_model.itemText(index), str(self.ai_model.itemData(index)))
                  for index in range(self.ai_model.count())
                  if str(self.ai_model.itemData(index)).startswith('user_')
                  and not is_standard(str(self.ai_model.itemData(index)))]
        self.voice_dialog = VoiceRegistrationDialog(self,
            ready=lambda: not self._pending and not self.controller.engine.running
                and (self.router is None or not self.router.ai.bridge.alive),
            voices=voices)
        self.voice_dialog.finished.connect(self._voice_registration_finished)
        self.voice_dialog.open()

    def _use_selected_voice(self):
        if self._pending or self._closing or self.controller.engine.running or self.voice_dialog:
            return
        self._stop_preview()
        self.mode.setCurrentIndex(self.mode.findData('ai_voice'))

    def _voice_registration_finished(self, _result):
        dialog = self.voice_dialog
        identifier = dialog.registered_id
        self.voice_dialog = None
        dialog.deleteLater()
        if identifier and not self._closing:
            from src.vc.models import VOICE_PROFILES, refresh_user_profiles
            refresh_user_profiles()
            self.ai_model.blockSignals(True)
            self.ai_model.clear()
            for key, value in VOICE_PROFILES.items():
                self.ai_model.addItem(value['name'], key)
            # The freshly registered voice is offered, so select it; fall back
            # to the default voice if it is somehow not listed.
            index = self.ai_model.findData(identifier)
            if index < 0:
                from src.vc.models import DEFAULT_VOICE_ID
                index = self.ai_model.findData(DEFAULT_VOICE_ID)
            self.ai_model.setCurrentIndex(index)
            self.ai_model.blockSignals(False)
            self._sync_voice_combos()
            self._ai_model_changed()
            self._capture_settings()
            self._save_settings()
        if not self._pending:
            self._set_controls(not self.controller.engine.running)

    def _preview_reference(self):
        if self.controller.engine.running or self._pending or self._closing:
            return
        if self.reference_player and self.reference_player.playbackState().name == 'PlayingState':
            self._stop_preview()
            return
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        from src.vc.models import profile
        if self.reference_player is None:
            self.reference_output = QAudioOutput(self)
            self.reference_output.setVolume(.5)
            self.reference_player = QMediaPlayer(self)
            self.reference_player.setAudioOutput(self.reference_output)
            self.reference_player.playbackStateChanged.connect(lambda _state: self.preview_voice.setText(
                '試聴を停止' if self.reference_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState else 'Referenceを試聴'))
            self.reference_player.errorOccurred.connect(lambda _error, message: self.voice_note.setText('試聴エラー: '+message))
        path = Path(__file__).resolve().parents[2]/'models'/profile(self.ai_model.currentData())['folder']/'reference.wav'
        self.reference_player.setSource(QUrl.fromLocalFile(str(path)))
        self.reference_player.play()

    def _stop_preview(self):
        if self.reference_player:
            self.reference_player.stop()
        if self.preview_player:
            self.preview_player.stop()
        self.preview_button.setText('変換を試聴')

    def _meanvc_preroll_text(self):
        return f'{self._ai_parameters().startup_frames/48:g} ms · MeanVC2固定pre-roll（実測遅延ではありません）'

    def _update_meanvc_controls(self):
        mean=is_meanvc2(self.ai_model.currentData())
        self.ai_delivery.setEnabled(mean and not self.controller.engine.running)
        self._update_delivery_note()
        self.finish_phrase.setVisible(mean)
        self.ai_threads.setVisible(mean)
        if mean:self.ai_wait_label.setText(self._meanvc_preroll_text())
        for i,label in enumerate(['Low Latency · chunk 13 ms','Balanced · chunk 26 ms','Stable · chunk 52 ms']):
            self.ai_quality.setItemText(i,'MeanVC2 · 160ms入力 / 120+40ms VC' if mean else label)
        for widget in (self.ai_brightness,self.ai_low_cut,self.ai_limiter,self.ai_post_fx,self.ai_pitch):
            widget.setEnabled(not mean)
        if hasattr(self,'prosody_on'):
            if mean:self.prosody_on.setChecked(False)
            self.prosody_on.setEnabled(not mean)
            if mean:self.prosody_engine.setCurrentIndex(self.prosody_engine.findData('off'))
            for widget in (self.prosody_engine,self.text_influence,self.text_strategy,self.log_transcripts):
                widget.setEnabled(not mean)
        if mean:
            from src.vc.models import profile
            selected=profile(self.ai_model.currentData())
            repair=selected.get('phrase_repair',False)
            description=('強弱のみ補完 β。ピッチの再編集はOFF。' if selected.get('repair_mode') == 'energy' else
                         '強弱・母音の微小な抑揚補完 β。Text-Aware・全体Pitch・EQはOFF。' if repair else 'Prosody・Text-Aware・Pitch・EQはOFF。')
            self.ai_performance.setText(f"Reference / Speaker Embedding固定。{description}\nモデルbuffer {selected['model_buffer_ms']}ms＋pre-roll {selected.get('startup_chunks',6)*160}ms＋補完待ち {selected.get('extra_delay_ms',0)}ms、特徴補間遅延 {selected['grid_delay_ms']}ms（実測遅延ではありません）。")

    def _update_delivery_note(self):
        """Explain the selected delivery route, with its measured cost and its limits."""
        from src.vc.models import DELIVERY_MODES
        mean=is_meanvc2(self.ai_model.currentData())
        self.delivery_note.setVisible(mean)
        if not mean:
            self.delivery_note.setText('')
            return
        mode=self.ai_delivery.currentData()
        spec=DELIVERY_MODES.get(mode)
        if spec is None:
            self.delivery_note.setText('')
            return
        if spec['delivery']=='streaming':
            text=('逐次変換：話しながら120msごとに処理します。\n'
                  '遅延は小さいですが、带域復元は行いません。')
        elif spec.get('experiment','none')=='none':
            text=('一括変換：0.8秒の無音で発話を区切り、変換後にLavaSRで帯域を復元します。最大60秒。\n'
                  'N150実測（オフラインWAV）：25秒分に約8.5秒、60秒分に約21秒。'
                  '変換待ち・録音中は無音です。人間の試聴評価は未実施。')
        else:
            # Listening-comparison variant: same voice, one behavioural difference.
            text=('【比較用】%s\n%s\n'
                  '通常の一括変換とは別モードです。聞き比べ用であり、速度・声質は未評価です。'
                  % (spec['label'], spec['detail']))
        self.delivery_note.setText(text)

    def _apply_natural_voice(self):
        """Select the default voice and apply the recommended settings."""
        if self._pending or self.controller.engine.running:return
        from src.vc.models import DEFAULT_VOICE_ID
        self.ai_model.blockSignals(True)
        self.ai_model.setCurrentIndex(self.ai_model.findData(DEFAULT_VOICE_ID))
        self.ai_model.blockSignals(False)
        self._apply_recommended()

    def _apply_recommended(self):
        if self._pending or self.controller.engine.running:
            return
        self.rate.setCurrentIndex(self.rate.findData(48000))
        self.buffer.setCurrentIndex(self.buffer.findData(256))
        for widget in (self.ai_quality,self.ai_wait,self.ai_fade):
            widget.blockSignals(True)
        mean=is_meanvc2(self.ai_model.currentData())
        # Every shipped voice is a MeanVC2 registration, so there is a single
        # recommended setting and no per-character variation to branch on.
        self.ai_quality.setCurrentIndex(self.ai_quality.findData('low_latency'))
        self.ai_wait.setValue(156 if mean else 78)
        self.ai_fade.setValue(20)
        for widget in (self.ai_quality,self.ai_wait,self.ai_fade):
            widget.blockSignals(False)
        if mean:
            from src.vc.models import profile
            threads=2 if profile(self.ai_model.currentData()).get('phrase_repair') else 4
            self.ai_threads.blockSignals(True);self.ai_threads.setCurrentIndex(self.ai_threads.findData(threads));self.ai_threads.blockSignals(False)
        self.ai_brightness.setValue(50 if mean else 47)
        self.ai_pitch.setValue(0 if mean else 32)
        for widget in (self.ai_low_cut,self.ai_limiter,self.ai_post_fx):
            widget.setChecked(not mean)
        self._ai_structure_changed()
        self._route_changed()  # device choice is preserved

    def _debug_snapshot(self):
        self._capture_settings()
        settings = asdict(self.settings)
        path = self.manager.path.parent/'debug_snapshots'/f'debug_snapshot_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
        def save():
            engine = self.controller.engine
            data = dict(settings=settings,engine=asdict(self.controller.snapshot),
                callback=asdict(engine.performance.snapshot()),main=asdict(engine.chain.performance.snapshot()),
                ai=self.router.ai.bridge.snapshot() if self.router else None,
                diagnostics=engine.diagnostics.snapshot(),startup=self.startup_diagnostic,
                measured_app_latency_ms=None)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            return str(path)
        self.files.submit('snapshot',save)

    def _dsp_changed(self, *_):
        self.preset.blockSignals(True)
        self.preset.setCurrentText("Custom")
        self.preset.blockSignals(False)
        self._publish_dsp()

    def _preset_changed(self, name):
        if name in PRESETS:
            self._apply_dsp_widgets(load_preset(name, self.quality.currentData()), name)

    def _meter(self, layout, name):
        row = QHBoxLayout()
        row.addWidget(QLabel(name))
        text = QLabel("−80.0 dBFS")
        row.addStretch()
        row.addWidget(text)
        layout.addLayout(row)
        meter = QProgressBar()
        meter.setRange(0, 800)
        meter.setValue(0)
        meter.setTextVisible(False)
        layout.addWidget(meter)
        return meter, text

    def _slider(self, layout, name, minimum, maximum, value):
        """A labelled slider in its own widget, so a whole control hides at once.

        The caption row and the track travel together: hiding only the track would leave
        a label promising a control that is not there.
        """
        container = QWidget()
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        row = QHBoxLayout()
        row.addWidget(QLabel(name))
        row.addStretch()
        label = QLabel()
        row.addWidget(label)
        column.addLayout(row)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        column.addWidget(slider)
        layout.addWidget(container)
        slider.container = container
        return slider, label

    @staticmethod
    def _container(widget):
        """The wrapper `_slider` created, or the widget itself for plain controls."""
        return getattr(widget, 'container', widget)

    def _labelled_row(self, name, widget):
        """A caption and a control in one widget, so both hide together."""
        container = QWidget()
        column = QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        column.addWidget(QLabel(name))
        column.addWidget(widget)
        widget.container = container
        return container

    def _voice_summary_line(self):
        """One plain sentence about the voice and route that will be used."""
        name = self.ai_model.currentText().strip()
        if not name:
            return '声がありません。「＋ 音声ファイルから声を追加」から登録してください。'
        spec = DELIVERY_MODES.get(self.ai_delivery.currentData(), {})
        route = spec.get('label', '')
        return f'いまの設定：{name}' + (f' · {route}' if route else '')

    def _update_control_visibility(self):
        """Show only the controls that can do something right now.

        A control that can never apply reads as a broken app, so anything the current
        mode, voice or delivery cannot use is hidden as well as disabled. The shipped
        standard voice is a MeanVC2 registration whose pitch, EQ and prosody are fixed,
        so those sections are not shown while it is selected.
        """
        if not hasattr(self, 'ai_delivery') or not hasattr(self, 'voice_quick'):
            return
        mode = self.mode.currentData()
        ai_mode = mode == 'ai_voice'
        dsp_mode = mode == 'female_dsp'
        mean = is_meanvc2(self.ai_model.currentData())
        spec = DELIVERY_MODES.get(self.ai_delivery.currentData(), {})
        utterance = spec.get('delivery') == 'utterance'
        natural = spec.get('experiment', 'none') == 'natural'
        # Home ②: the quick voice picker and the file preview belong to the AI route.
        self.voice_quick.setVisible(ai_mode)
        self.voice_quick_note.setVisible(ai_mode)
        if ai_mode:
            self.voice_quick_note.setText(self._voice_summary_line())
        self.preview_button.setVisible(ai_mode)
        self.use_voice.setVisible(not ai_mode)
        # Library: committing an unspoken utterance only exists on that route.
        self.finish_phrase.setVisible(mean and utterance)
        # Voice tuning exists for the listening-comparison route only.
        self.tune_header.setVisible(natural)
        for widget in self._tune_widgets:
            self._container(widget).setVisible(natural)
        # The Female DSP page only drives the DSP path. Whole rows hide, captions
        # included, so the page never shows a label without its control.
        self.pitch_row.setVisible(dsp_mode)
        self.dsp_note.setVisible(dsp_mode)
        for widget in (self.preset, self.quality, self.pitch, self.formant, self.brightness,
                       self.wet, self.low_cut, self.limiter):
            self._container(widget).setVisible(dsp_mode)
        self.dsp_switch.setVisible(self.dsp is not None and not dsp_mode)
        if self.dsp is None:
            self.dsp_notice.setText('このビルドではFemale DSPは使えません。')
        elif dsp_mode:
            self.dsp_notice.setText('')
        elif ai_mode:
            self.dsp_notice.setText('いまは「AI Voice」です。このページの加工（Pitch / Formant / '
                                    '明るさ / EQ）はAI Voiceでは使いません。声の選択は「声ライブラリ」で行います。')
        else:
            self.dsp_notice.setText('いまは「Original（変換なし）」です。軽い加工を使うには'
                                    'Female DSPに切り替えてください。')
        # Advanced: MeanVC2 fixes the processing chunk, pitch and brightness, and the
        # prosody path stays off for it, so those sections are hidden rather than dead.
        for widget in (self.ai_quality, self.ai_brightness, self.ai_low_cut, self.ai_limiter,
                       self.ai_post_fx, self.ai_pitch, self.ai_wait):
            self._container(widget).setVisible(not mean)
        self.prosody_tab.setVisible(not mean)
        self.prosody_advanced_box.setVisible(not mean)
        self.prosody_fixed_note.setVisible(mean)

    def _monitor_toggled(self, active):
        """Remember the switch and apply it live, without a Stop/Start cycle."""
        self._update_monitor_ui()
        self._monitor_changed()

    def _monitor_volume_changed(self, _value):
        self._monitor_changed()

    def _monitor_changed(self, *_):
        """Coalesce monitor edits into one live update."""
        self._update_monitor_ui()
        if not self._closing:
            self._monitor_apply.start()

    def _update_monitor_ui(self):
        """Keep the switch label, level readout and device warning in step."""
        if not hasattr(self, 'monitor_toggle'):
            return
        active = self.monitor_toggle.isChecked()
        self.monitor_toggle.setText('Monitor ON' if active else 'Monitor OFF')
        self.monitor_volume_label.setText(f'{self.monitor_volume.value()/10:+.1f} dB')
        device = self.monitor_device.currentData()
        output = self.output_device.currentData() if hasattr(self, 'output_device') else None
        if device is None:
            self.monitor_warning.setText('モニターに使える出力デバイスがありません。')
        elif output is not None and device.identity == output.identity:
            self.monitor_warning.setText('⚠ メインの出力先と同じデバイスです。同じ音が二重に'
                                         '聞こえるので、ヘッドホンなど別のデバイスを選んでください。')
        else:
            self.monitor_warning.setText('')
        self.monitor_volume.setEnabled(device is not None)
        self.monitor_test.setEnabled(device is not None and not self._monitor_test_busy)

    def _apply_monitor(self):
        """Push the monitor state to the audio thread (live while running)."""
        if self._closing:
            return
        self._capture_settings()
        self._save_settings()
        if not self.controller.engine.running:
            return
        if self._pending:
            # One request id is in flight; retry shortly instead of clobbering it.
            self._monitor_apply.start()
            return
        self._pending = True
        self._pending_request = self.controller.configure_monitor(
            self.monitor_toggle.isChecked(), self.monitor_device.currentData(),
            self.monitor_volume.value()/10)

    def _test_monitor(self):
        """Play one short chime on the monitor device; nothing is recorded."""
        if self._closing or self._monitor_test_busy:
            return
        device = self.monitor_device.currentData()
        if device is None:
            self.monitor_note.setText('テストできる出力デバイスがありません。')
            return
        self._monitor_test_busy = True
        self.monitor_test.setEnabled(False)
        self.monitor_note.setText(f'{device.label} でテスト音を再生しています…')
        gain = self.monitor_volume.value()/10 - 12.0
        backend = self.controller.engine.backend
        from src.audio.monitor import play_test_tone
        self.files.submit('monitor_test', lambda: 'テスト音を再生しました：' +
                          str(play_test_tone(backend, device, 48000, 1.2, gain)))

    def _drain_monitor_test_note(self):
        """Publish the worker's test-tone outcome once it finishes."""
        message = self.files.results.pop('monitor_test', None)
        if message:
            self._monitor_test_busy = False
            self._monitor_test_note = (message, time.monotonic()+10.0)
            self._update_monitor_ui()
        note = self._monitor_test_note
        if note and time.monotonic() < note[1]:
            self.monitor_note.setText(note[0])
        elif note:
            self._monitor_test_note = None

    def _gain_changed(self, value):
        self.gain_label.setText(f"{value / 10:+.1f} dB")
        self.controller.set_gain(value / 10)

    def _gate_changed(self, value):
        self.gate_label.setText(f"{value / 10:.1f} dB")
        self.controller.set_gate(value / 10)

    def _refresh_devices(self):
        self._capture_settings()
        self._pending = True
        self.status.setText("Status: Loading devices")
        self._set_controls(False)
        self._pending_request = self.controller.refresh_devices()

    def _populate_devices(self):
        devices = self.controller.devices
        selected_input = choose_device(devices, "input", self.settings.input_device)
        selected_output = choose_device(devices, "output", self.settings.output_device,
                                         selected_input.host_api if selected_input else None)
        for combo, direction, selected in ((self.input_device, "input", selected_input),
                                             (self.output_device, "output", selected_output)):
            combo.blockSignals(True)
            combo.clear()
            for device in sorted(devices,key=lambda d: 'wasapi' not in d.host_api.lower()):
                if getattr(device, f"max_{direction}_channels") > 0:
                    combo.addItem(device.label, device)
            if selected:
                combo.setCurrentIndex(next(i for i in range(combo.count())
                                           if combo.itemData(i) == selected))
            combo.blockSignals(False)
        selected_monitor = choose_device(devices, "output", self.settings.monitor_device)
        self.monitor_device.blockSignals(True)
        self.monitor_device.clear()
        for device in sorted(devices,key=lambda d: 'wasapi' not in d.host_api.lower()):
            if device.max_output_channels > 0:
                self.monitor_device.addItem(device.label, device)
        if selected_monitor:
            self.monitor_device.setCurrentIndex(next((i for i in range(self.monitor_device.count())
                                                      if self.monitor_device.itemData(i) == selected_monitor),
                                                     self.monitor_device.currentIndex()))
        self.monitor_device.blockSignals(False)
        self._update_monitor_ui()
        self._route_changed()

    def _route_changed(self):
        input_device = self.input_device.currentData()
        output_device = self.output_device.currentData()
        if not input_device or not output_device:
            hint = ("音声デバイスがありません。接続・Windowsのマイク権限を確認してください。"
                    if sys.platform == "win32" else
                    "音声デバイスがありません。接続・OSのマイク権限を確認してください。")
        elif input_device.host_api != output_device.host_api:
            hint = ("InputとOutputを同じ方式にしてください。両方 [Windows WASAPI] を推奨。"
                    if sys.platform == "win32" else
                    "InputとOutputを同じ方式にしてください（PulseAudio / PipeWire）。")
        elif 'mme' in input_device.host_api.lower() or 'mme' in output_device.host_api.lower():
            hint = 'WASAPI recommended — MMEでは音切れを記録しています。Input / OutputともWindows WASAPIを推奨。'
        elif "cable input" in output_device.name.lower():
            hint = "App → CABLE Input ／ Discordのマイク → CABLE Output"
        elif sys.platform != "win32":
            hint = "仮想ケーブル未選択。スピーカーへの出力時はハウリングに注意し、ヘッドホンを使用してください。"
        else:
            hint = "VB-CABLE未選択。スピーカーへの出力時はハウリングに注意し、ヘッドホンを使用してください。"
        self.route_hint.setText(hint)

    def _start(self):
        input_device = self.input_device.currentData()
        output_device = self.output_device.currentData()
        if self._pending or self.voice_dialog is not None or not input_device or not output_device:
            return
        self._stop_preview()
        self._capture_settings()
        self._save_settings()
        self._pending = True
        self._set_controls(False)
        self.status.setText("Status: Starting")
        self._pending_request = self.controller.start(EngineConfig(
            input_device, output_device, self.rate.currentData(), self.buffer.currentData(),
            monitor=self.monitor_toggle.isChecked(),
            monitor_device=self.monitor_device.currentData(),
            monitor_gain_db=self.monitor_volume.value()/10))

    def _stop(self):
        if self._pending:
            return
        self._pending = True
        self._set_controls(False)
        self.status.setText("Status: Stopping")
        self._pending_request = self.controller.stop()

    def _set_controls(self, stopped):
        for widget in (self.input_device, self.output_device, self.rate, self.buffer, self.refresh):
            widget.setEnabled(stopped and not self._closing)
        # Monitor is applied live, so it stays available while the engine runs.
        for widget in (self.monitor_device, self.monitor_toggle, self.monitor_volume,
                       self.monitor_test):
            widget.setEnabled(not self._closing and not self._pending)
        self.quality.setEnabled(stopped and not self._closing and self.dsp is not None
                                and self.mode.currentData() != 'ai_voice')
        for widget in (self.ai_model,self.ai_delivery,self.ai_quality,self.ai_threads,self.ai_wait,self.ai_fade,self.recommended,self.ai_natural):
            widget.setEnabled(stopped and not self._closing and self.router is not None)
        from src.vc.models import DELIVERY_MODES as _DELIVERY_MODES  # local alias: module also imports it at top
        _tune_active = (_DELIVERY_MODES.get(self.ai_delivery.currentData(), {}).get('experiment','none') == 'natural')
        for widget in self._tune_widgets:
            widget.setEnabled(stopped and not self._closing and self.router is not None and _tune_active)
        # Switching to/from the natural comparison refreshes tune availability.
        if hasattr(self, 'delivery_note'):
            self._update_delivery_note()
        mean=is_meanvc2(self.ai_model.currentData())
        self.ai_delivery.setEnabled(stopped and not self._closing and mean and self.router is not None)
        self.finish_phrase.setEnabled(not stopped and mean and self.router is not None and self._utterance_active() and not self._closing)
        self._update_control_visibility()
        self.ai_threads.setEnabled(stopped and not self._closing and mean)
        self.ai_wait.setEnabled(stopped and not self._closing and not mean)
        self.ai_quality.setEnabled(stopped and not self._closing and not mean)
        self.ai_restart.setEnabled(self.router is not None and not self._pending and not self._closing)
        self.mode.setEnabled(self.router is not None and not self._pending and not self._closing)
        self._update_monitor_ui()
        available = self.input_device.currentData() is not None and self.output_device.currentData() is not None
        self.start_button.setEnabled(stopped and available and not self._closing)
        self.stop_button.setEnabled(not stopped and not self._pending and not self._closing)
        self.add_voice.setEnabled(stopped and not self._pending and not self._closing and self.voice_dialog is None and self.router is not None)
        self.preview_voice.setEnabled(stopped and not self._pending and not self._closing and self.voice_dialog is None and is_meanvc2(self.ai_model.currentData()))
        # The converted-output preview needs a voice, a stopped engine and no pending work.
        self.preview_button.setEnabled(stopped and not self._pending and not self._closing
                                       and self.voice_dialog is None and not self._preview_busy
                                       and self.router is not None
                                       and self.mode.currentData() == 'ai_voice')
        self.use_voice.setEnabled(stopped and not self._pending and not self._closing and self.voice_dialog is None and self.router is not None)
        # The rail pitch slider only means something on the DSP path.
        self.pitch_display.setEnabled(stopped and not self._pending and not self._closing
                                      and self.dsp is not None
                                      and self.mode.currentData() == 'female_dsp')
        if self.voice_dialog is not None:
            self.start_button.setEnabled(False)
            self.ai_restart.setEnabled(False)
            self.mode.setEnabled(False)
            self.ai_model.setEnabled(False)

    def _tick(self):
        if self._closing and not self.controller.alive and not self.files.alive:
            self._can_close = True
            self.close()
            return
        snapshot = self.controller.snapshot
        revision = self.controller.devices_revision
        if revision != self._revision:
            self._revision = revision
            self._populate_devices()
        # A queued request hasn't necessarily reached the worker yet. A request
        # ID prevents a previous Stopped/Running snapshot from acknowledging it.
        if self._pending:
            if (snapshot.request_id < self._pending_request
                    or snapshot.state in ("Starting", "Stopping", "Loading devices")):
                return
            self._pending = False
            if self.router:
                # A quality change is serialized on the control thread. Replay
                # the latest slider values after its acknowledgement so edits
                # made while loading are not overwritten by the queued snapshot.
                self._ai_fx_changed()
        if not self._closing:
            suffix = f" — {snapshot.error}" if snapshot.error else ""
            if self.router and self.mode.currentData() == 'ai_voice':
                bridge = self.router.ai.bridge
                suffix += f" · AI: {bridge.status}"
                if bridge.status != 'Ready' or self.rate.currentData() != 48000:
                    suffix += ' · Fallback: Original'
            self.status.setText(f"Status: {snapshot.state}{suffix}")
            stopped = snapshot.state in ("Stopped", "Error")
            busy = snapshot.state in ("Starting", "Stopping", "Loading devices", "Restart required")
            self._set_controls(stopped)
            if busy:
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(False)
        input_peak, output_peak = self.controller.engine.take_peaks()
        # Visual-only decay at 30 Hz; callback just publishes current peaks.
        self._display_input = max(input_peak, self._display_input * 0.8) if snapshot.state == "Running" else 0
        self._display_output = max(output_peak, self._display_output * 0.8) if snapshot.state == "Running" else 0
        for meter, label, value in ((self.input_meter, self.input_db, self._display_input),
                                      (self.output_meter, self.output_db, self._display_output)):
            db = amplitude_to_db(value)
            meter.setValue(round(min(0, db) * 10 + 800))
            label.setText(f"{db:.1f} dBFS")
        self.input_bar.set_level(self._display_input)
        self.output_bar.set_level(self._display_output)
        self._update_orb(snapshot)
        self.xruns.setText(f"Underflow: {snapshot.underflows} · Overflow: {snapshot.overflows}"
                          + (" · 音切れ時はBufferを増やしてください" if snapshot.underflows or snapshot.overflows else ""))
        self._update_latency()
        now = time.monotonic()
        if now < self._next_performance_update:
            return
        self._next_performance_update = now + 0.25
        engine = self.controller.engine
        player = engine.monitor_player
        if player is not None and player.running:
            tap = engine.monitor_tap
            self.monitor_note.setText(
                f"Monitor再生中：{player.device_label} · 途切れ {player.zeroed} / 溢れ {tap.dropped}"
                "（多少は仕様です。ヘッドホン推奨）")
        elif engine.monitor_error:
            self.monitor_note.setText(f"Monitorを開始できませんでした：{engine.monitor_error}")
        elif self.monitor_toggle.isChecked() and self.controller.engine.running:
            self.monitor_note.setText('Monitorを開始しています…')
        else:
            self.monitor_note.setText("Monitorは変換後の音声を別デバイスで聴く機能です。多少の途切れは仕様です。")
        self._drain_monitor_test_note()
        dsp_stats = self.controller.engine.chain.performance.snapshot()
        callback_stats = self.controller.engine.performance.snapshot()
        if self.router:
            ai = self.router.ai.bridge.snapshot()
            fallback = (' · Fallback: Original' if self.mode.currentData() == 'ai_voice' and
                        (ai['status'] != 'Ready' or self.rate.currentData() != 48000) else '')
            if ai.get('delivery')=='utterance' and fallback:fallback=' · 発話出力は無音'
            self.ai_status.setText(f"AI Status: {ai['status']}{fallback}\n{ai['error']}")
            health = (realtime_health(ai,callback_stats,1000*self.buffer.currentData()/self.rate.currentData())
                      if self.mode.currentData()=='ai_voice' else self.mode.currentText())
            prosody=ai['prosody']
            self.health.setText('Realtime Health: '+health+' · '+prosody['health'])
            # Progress belongs to a live run: while stopped this line would claim a
            # recording that is not happening.
            if ai.get('delivery')=='utterance' and self.controller.engine.running:
                utterance=ai.get('utterance',{})
                self.health.setText(f"発話単位 · {utterance.get('state','準備中')} · 録音 {utterance.get('recording_seconds',0):.1f}秒 · 待ち {utterance.get('queued',0)} · 完了 {utterance.get('completed',0)}\n{utterance.get('message','')}")
                self.finish_phrase.setEnabled(ai['status']=='Ready' and self.controller.engine.running and not self._pending)
            self.prosody_status.setText(('Context Prosody Active · ' if prosody['health']=='Prosody Healthy' else '')+prosody['health']+' · '+prosody['status']+'\n'+prosody['error'])
            if prosody['status']=='Error' and self.prosody_on.isChecked():
                self.prosody_on.setChecked(False)
            control=prosody.get('control',{})
            timing=prosody.get('analysis',{})
            asr=prosody.get('asr',{})
            if self.prosody_engine.currentData()=='text_v3':
                self.prosody_status.setText(self.prosody_status.text()+f"\nText Prosody {asr.get('status','Stopped')} · "+('Text Context Active' if asr.get('text_active') else 'Rule v2 fallback'))
            text=self.router.ai.bridge.prosody.asr.current()
            performance=asr.get('processing',{})
            self.asr_details.setText(
                f"Text Prosody: {asr.get('status','Stopped')} · {asr.get('error','')}\n"
                f"Partial: {text.get('partial_text','')}\nStable: {text.get('stable_prefix','')}\n"
                f"Context: {text.get('phrase_type','UNKNOWN')} · confidence {text.get('context_confidence',0):.2f} · age {text.get('age_ms',0):.0f} ms\n"
                f"ASR avg / p50 / p95 / p99 / max: "+' / '.join(f'{performance.get(k,0):.2f}' for k in ('average_ms','p50_ms','p95_ms','p99_ms','maximum_ms'))+
                f" ms · RTF {asr.get('rtf',0):.2f}\nQueue {asr.get('queue',{}).get('current',0):.2f} · drops {asr.get('dropped_chunks',0):.2f}\n"
                f"Audio / Text / Final ΔPitch: {control.get('audio_pitch_delta',0):+.3f} / {control.get('text_pitch_delta',0):+.3f} / {prosody['applied_pitch_delta']:+.3f} st\n"
                f"Audio / Text / Final ΔGain: {control.get('audio_gain_db',0):+.2f} / {control.get('text_gain_db',0):+.2f} / {prosody['applied_gain_db']:+.2f} dB\n"
                f"Window {asr.get('window_ms',0):.0f} ms · lag {asr.get('lag',{}).get('last_ms',0):.0f} ms · Phrase {control.get('phrase_id',0)}\n"
                f"Text requested / effective: {control.get('text_pitch_delta',0):+.3f} / {control.get('effective_text_pitch',0):+.3f} st · survival {control.get('text_effect_survival_rate',0):.0%} · duty {control.get('text_effect_active_ratio',0):.0%}\n"
                f"Rule range / rise / fall: {control.get('text_range_multiplier',1):.2f} / {control.get('text_rise_multiplier',1):.2f} / {control.get('text_fall_multiplier',1):.2f}\n"
                f"Events: "+', '.join(f"{e['event_type']} {e['state']} confidence {e['confidence']:.2f} · envelope {e['envelope']:.2f} / effective {e['effective_duration_ms']:.0f} ms" for e in control.get('text_events',{}).get('active_events',[]))+
                '\nEvent waterfall: '+str(control.get('text_events',{}).get('waterfall',{})))
            self.prosody_details.setText(
                f"Phrase: {control.get('phrase_state','SILENCE')} · ending: {control.get('ending_probability',0):.2f}\n"
                f"Slope: pitch {control.get('pitch_slope',0):+.2f} st/s · energy {control.get('energy_slope',0):+.2f} dB/s\n"
                f"Raw / smoothed / quantized ΔPitch: {control.get('raw_pitch_delta',0):+.3f} / {control.get('pitch_delta',0):+.3f} / {control.get('quantized_pitch_delta',0):+.3f} st (0.125 st)\n"
                f"F0 / baseline: {control.get('f0',0):.1f} / {control.get('baseline_f0',0):.1f} Hz · periodicity: {control.get('voiced',0):.2f}\n"
                f"Relative pitch: {control.get('relative_pitch_st',0):+.2f} st · ΔPitch: {prosody['applied_pitch_delta']:+.2f} st · gain: {prosody['applied_gain_db']:+.2f} dB\n"
                f"FFT YIN avg / p50 / p95 / p99 / max: "+' / '.join(f'{timing.get(k,0):.3f}' for k in ('average_ms','p50_ms','p95_ms','p99_ms','maximum_ms'))+
                f" ms\nWorker: {prosody['status']} · Queue: {prosody['queue']['current']:.2f} · drops: {prosody['queue_drop_chunks']:.2f} · invalid F0: {prosody.get('invalid_f0_count',0)} · errors: {prosody['errors']}")
            lines = []
            for key,label in [('inference','Inference'),('process_total','AI process total'),('resample','Resample'),('postprocess','Post FX'),('rpc','Worker / IPC'),('queue_wait','Input queue wait'),('output_wait','Output queue wait')]:
                stats = ai.get(key, {})
                values = [stats.get(k,0) for k in ('average_ms','p50_ms','p95_ms','p99_ms','maximum_ms')]
                lines.append(f"{label} avg / p50 / p95 / p99 / max: "+' / '.join(f'{v:.2f}' for v in values)+' ms')
            for key in ('input_queue','output_queue'):
                q = ai[key]
                lines.append(f"{key} current / avg / p95 / max: {q['current']:.2f} / {q['average']:.2f} / {q['p95']:.2f} / {q['maximum']:.2f}; high-water: {q['high_water_events']}")
            lines.append(f"Worker: {ai['worker_state']} · Crossfades: {ai['mode_crossfades']}")
            if ai['mode_crossfades'] != self._logged_crossfades:
                log.info('Mode crossfade: %s',ai['mode_crossfades'])
                self._logged_crossfades = ai['mode_crossfades']
            lines += [f"RTF: {ai.get('rtf',0):.3f} · Queue: {ai['queue_current']:.2f} / {ai['queue_capacity']} · Max: {ai['queue_max']:.2f}",
                      f"AI underrun: {ai['ai_underrun']} · AI overrun: {ai['ai_overrun']} · Dropped: {ai['dropped_chunks']:.2f}",
                      f"Pre-roll target / observed: {ai['buffering_target_ms']:.1f} / {ai['preroll_observed_ms']:.1f} ms · 外部Resample: {ai['resampler_delay_ms']} ms · Model delay: 未実測",
                      '処理時間・buffering・モデル遅延・I/Oは別です。End-to-Endは未実測。']
            if is_meanvc2(self.ai_model.currentData()):
                lines.append(f"Reference固定 · {ai['model'].get('steps','?')} steps · Prosody/Text-Aware OFF · Model buffer {ai['model'].get('algorithmic_buffer_ms',480)} ms · 特徴補間遅延 {ai['model'].get('interpolation_grid_delay_ms',0)} ms · 実機の声質は要確認")
                if ai['model'].get('phrase_repair'):
                    repair=ai.get('phrase',{})
                    state='補完停止・元のVC音声' if repair.get('disabled') else '抑揚補完 β'
                    lines.append(f"{state} · 追加待ち1600ms · 適用 {repair.get('applied',0)} / 遅延時の無補正 {repair.get('late_bypass',0)} · エラー {repair.get('failures',0)}")
            memory = ai.get('ram_bytes', ai['model'].get('ram_bytes',0))
            if memory:
                lines.append(f'AI Worker RAM: {memory/1024**2:.1f} MiB')
            inference_max = ai.get('inference', {}).get('maximum_ms',0)
            if self.mode.currentData() == 'ai_voice' and inference_max >= .8*self.router.ai.bridge.chunk_frames/48:
                lines.append('WARNING: AI processing is close to chunk deadline（累積max）')
            self.ai_performance.setText('\n'.join(lines))
            if ai.get('delivery')=='utterance':
                params=self._ai_parameters()
                post='LavaSR' if params.enhancer=='lavasr' else '後処理なし'
                if params.experiment!='none':
                    post+='（比較用: %s）' % params.experiment
                self.ai_performance.setText(f"発話を録音 → MeanVC2を発話ごとにリセット・末尾まで変換 → {post} → 全体を出力\n"
                    f"直近VC {ai.get('utterance_vc_seconds',0):.1f}秒 / 後処理 {ai.get('utterance_post_seconds',0):.1f}秒 · 全体RTF {ai.get('rtf',0):.2f}\n"
                    f"受け付け不可 {ai.get('utterance',{}).get('rejected',0)}発話 · 30秒分割 {ai.get('utterance',{}).get('limit_splits',0)}\n"
                    f"{('FlashSR失敗・MeanVC2のみで出力: '+ai.get('enhancer_error','')) if ai.get('enhancer_error') else '話し終えるまで出力しません。ライブ遅延・声質は要試聴。'}"
                    + (f"\n入力目安: メーター緑〜黄（-20dB前後）。現在 {amplitude_to_db(max(self._display_input,1e-6)):.0f}dB"
                       if params.experiment != 'none' else ''))
        self.file_status.setText(self.files.error or self.files.result)
        if self.startup_diagnostic and not self.startup_diagnostic['ai_available']:
            self.ai_status.setText('AI Voice unavailable · Original / Female DSP使用可能\n'+'; '.join(self.startup_diagnostic['issues']))
        deadline = 1000 * self.buffer.currentData() / self.rate.currentData()
        self.performance.setText(
            f"Main callback avg / p95 / max: {dsp_stats.average_ms:.3f} / {dsp_stats.p95_ms:.3f} / {dsp_stats.maximum_ms:.3f} ms\n"
            f"Callback avg / p50 / p95 / p99 / max: {callback_stats.average_ms:.3f} / {callback_stats.p50_ms:.3f} / {callback_stats.p95_ms:.3f} / {callback_stats.p99_ms:.3f} / {callback_stats.maximum_ms:.3f} ms\n"
            f"Deadline: {deadline:.2f} ms · PortAudio CPU load: {self.controller.engine.cpu_load * 100:.1f}%"
            f" · 超過: {callback_stats.deadline_exceeded}"
            + ("\nWARNING: Realtime processing is close to audio deadline"
               if snapshot.state == "Running" and callback_stats.maximum_ms >= deadline * 0.8 else ""))
        self._fit_page(self._active_page)

    def _update_orb(self, snapshot):
        """Feed the centre orb from the same numbers the meters show.

        The orb never invents activity: when the engine is stopped the level is zero and
        the ring is empty, so the display cannot suggest signal that was never measured.
        """
        running = snapshot.state == 'Running' and not snapshot.error
        meanvc = self.mode.currentData() == 'ai_voice'
        if running:
            if meanvc and self.router:
                bridge = self.router.ai.bridge
                if bridge.status != 'Ready':
                    subtitle = 'AI準備中 · 現在はOriginalを通します'
                else:
                    delivery = getattr(self.router.ai.bridge.parameters, 'delivery', 'streaming')
                    subtitle = ('話終わるまで待ってから出力' if delivery == 'utterance'
                                else '声が変換されて出力されます')
            else:
                subtitle = '声が変換されて出力されます'
            state = 'マイク入力中'
        elif snapshot.error:
            state = 'エラー'
            subtitle = snapshot.error
        else:
            state = '停止中'
            subtitle = 'マイク入力待ち'
        level = max(self._display_input, self._display_output) if running else 0.0
        self.orb.set_state(state, subtitle, active=running)
        self.orb.set_level(level)
        if running:
            self.output_card_text.setText('現在の設定: ' + self._describe_output())
        else:
            self.output_card_text.setText('停止中 · Startで変換を開始します')

    def _describe_output(self):
        mode = self.mode.currentData()
        if mode == 'original':
            return 'Original（無変換）'
        if mode == 'female_dsp':
            return 'Female DSP'
        if mode != 'ai_voice':
            return '停止中'
        model = self.ai_model.currentText()
        try:
            parameters = self._ai_parameters()
        except ValueError:
            return '声がありません'
        if parameters.delivery == 'utterance' and parameters.enhancer != 'none':
            return f'{model} · {parameters.enhancer}'
        return model

    def _update_latency(self):
        rate, frames = self.rate.currentData(), self.buffer.currentData()
        if not rate or not frames:
            return
        one = 1000 * frames / rate
        reported = self.controller.snapshot.reported_latency_ms
        text = f"{rate} Hz · Mono DSP · Buffer: {frames}\n"
        text += f"概算Buffer latency: {one:.1f} ms / block · 入出力2ブロック仮定: {2 * one:.1f} ms"
        if reported is not None and self.controller.snapshot.state == "Running":
            text += f"\nPortAudio報告 I/O推定: {reported:.1f} ms"
            delay = (self.dsp.algorithmic_latency_samples / rate * 1000 if self.dsp and self.mode.currentData() == 'female_dsp' else 0)
            text += f" · DSP内部遅延: {delay:.1f} ms"
            text += "\n処理時間とDSP内部遅延は別の値です。"
        self.latency.setText(text + "\nいずれもエンドツーエンドの実測値ではありません。")

    def _restore_position(self):
        if self.settings.window_position:
            self.move(*self.settings.window_position)
        # Restore onto an available screen, also after disconnecting a monitor.
        if not any(screen.availableGeometry().intersects(self.frameGeometry()) for screen in QApplication.screens()):
            self.move(QApplication.primaryScreen().availableGeometry().topLeft())

    def _capture_settings(self):
        input_device, output_device = self.input_device.currentData(), self.output_device.currentData()
        monitor_device = self.monitor_device.currentData() if hasattr(self, 'monitor_device') else None
        monitor_on = self.monitor_toggle.isChecked() if hasattr(self, 'monitor_toggle') else False
        monitor_db = self.monitor_volume.value()/10 if hasattr(self, 'monitor_volume') else self.settings.monitor_volume_db
        self.settings = replace(self.settings,
            input_device=input_device.identity if input_device else None,
            output_device=output_device.identity if output_device else None,
            monitor_device=monitor_device.identity if monitor_device else None,
            monitor=monitor_on, monitor_volume_db=monitor_db,
            sample_rate=self.rate.currentData(), buffer_size=self.buffer.currentData(),
            gain_db=self.gain.value() / 10, noise_gate_db=self.gate.value() / 10,
            window_size=(self.width(), self.height()),
            window_position=(self.x(), self.y()))
        parameters = self._dsp_parameters()
        self.settings = replace(self.settings, dsp_mode=parameters.mode,
            preset=self.preset.currentText(), pitch_semitones=parameters.pitch,
            formant_semitones=parameters.formant, brightness=parameters.brightness,
            low_cut=parameters.low_cut, limiter=parameters.limiter, wet=parameters.wet,
            dsp_quality=parameters.quality)
        ai = None
        try:
            ai = self._ai_parameters()
        except ValueError:
            pass
        if ai is not None:
            self.settings = replace(self.settings, voice_mode=self.mode.currentData(), ai_model=ai.model,
            ai_quality=ai.quality, ai_threads=ai.threads, ai_brightness=ai.brightness,
            ai_low_cut=ai.low_cut, ai_limiter=ai.limiter, ai_post_fx=ai.post_fx,
            ai_output_wait_ms=ai.output_wait_ms,ai_crossfade_ms=ai.crossfade_ms,ai_pitch=ai.pitch,
            ai_delivery=ai.delivery,ai_enhancer=ai.enhancer,ai_lavasr_denoise=ai.lavasr_denoise,
            ai_experiment=ai.experiment, ai_tune_sib_db=ai.tune_sib_db,
            ai_tune_cons_db=ai.tune_cons_db, ai_tune_caps=ai.tune_caps,
            ai_tune_floor_db=ai.tune_floor_db, ai_tune_excess_db=ai.tune_excess_db,
            ai_tune_mid=ai.tune_mid, ai_tune_match=ai.tune_match,
            ai_tune_ptrans=ai.tune_ptrans, ai_tune_pcap=ai.tune_pcap,
            ai_tune_combined=ai.tune_combined, ai_tune_level_db=ai.tune_level_db)
        self.settings=replace(self.settings,prosody=asdict(self._prosody_parameters()))

    def _save_settings(self):
        settings = replace(self.settings)
        self.files.submit('settings',lambda: self.manager.save(settings))

    def closeEvent(self, event: QCloseEvent):
        if self._can_close:
            self.timer.stop()
            event.accept()
            return
        event.ignore()
        if not self._closing:
            self._stop_preview()
            if self.voice_dialog:
                self.voice_dialog.reject()
            self._capture_settings()
            self._save_settings()
            self.files.close()  # drain final save without blocking Qt
            self._closing = True
            self.status.setText("Status: Closing — 音声デバイスを解放しています")
            self._set_controls(False)
            self.controller.shutdown(self.files.wait_closed)
