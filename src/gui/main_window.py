from dataclasses import asdict, replace
from datetime import datetime
import json
import logging
import os
from src.vc.models import DEFAULT_DELIVERY, DELIVERY_MODES, is_meanvc2
from pathlib import Path
import sys
import time

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (QApplication, QButtonGroup, QComboBox, QFormLayout, QHBoxLayout,
                               QCheckBox, QFrame, QLabel, QLayout, QLineEdit, QMainWindow, QMessageBox,
                               QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QVBoxLayout, QWidget, QDoubleSpinBox, QSpinBox)

from src.audio.controller import AudioController
from src.audio.devices import choose_device, is_virtual
from src.audio.engine import AudioEngine, EngineConfig
from src.audio.governor import LatencyGovernor
from src.lab import actions as lab_actions
from src.lab.parameters import LabParameters
from src.processors.whisper import WhisperVoicer
from src.audio.meters import amplitude_to_db
from src.settings.manager import BUFFERS, SAMPLE_RATES
from src.settings.scenes import SceneStore
from src.presence import PresenceClient
from src.processors.chain import ProcessorChain
from src.processors.dsp_parameters import DSPParameters, MODES
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
    def __init__(self, settings_manager, controller=None, presence=None, presence_factory=None):
        super().__init__()
        self.manager = settings_manager
        self.settings = self.manager.load()
        # Discord activity (Rich Presence). Injected by tests; otherwise created on
        # the first show, so importing the GUI never opens a connection.
        self.presence = presence
        self._presence_factory = presence_factory or PresenceClient
        self._run_started_at = None
        self.files = BackgroundFiles()
        self.startup_diagnostic = None
        self._diagnostic_model = self.settings.ai_model
        def diagnose():
            self._diagnose_voice(self.settings.ai_model)
        self.files.submit('diagnostics', diagnose)
        # Lab features. The whisper stage is built here so it is part of the chain from
        # the first block; the governor only ever reads counters.
        self.lab_parameters = self.settings.lab_parameters()
        self.governor = LatencyGovernor()
        self.whisper = WhisperVoicer(self.lab_parameters.whisper_amount,
                                     self.lab_parameters.whisper_pitch)
        self.whisper.amount = self.lab_parameters.whisper_amount if self.lab_parameters.whisper else 0.0
        if controller is None:
            self.dsp = FemaleDSPProcessor(self.settings.dsp_parameters())
            self.router = VoiceRouter(self.dsp, AIVoiceProcessor(self.settings.ai_parameters()), self.settings.voice_mode)
            self.controller = AudioController(AudioEngine(chain=ProcessorChain(
                main_processor=self.router, whisper=self.whisper)))
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
            # An injected chain owns its pre-processors; use its stage when it has one.
            self.whisper = getattr(controller.engine.chain, 'whisper', None)
        # Live translation voice: mic JA -> EN speech to the output device. The
        # driver owns its ASR worker and thread; the router plays finished
        # utterances. Armed only from the translate route (see
        # _update_translation_driver), so it never spends RAM unasked.
        from src.translate.live import TranslationDriver
        playback = self.router.translate_playback if self.router is not None else None
        self.translate_driver = TranslationDriver(self.controller.engine, playback)
        self._translate_join_pending = False
        self._revision = -1
        self._pending = False
        self._pending_request = 0
        self._closing = False
        self._can_close = False
        self._display_input = self._display_output = 0.0
        self._next_performance_update = 0.0
        self._logged_crossfades = 0
        self.voice_dialog = None
        self._active_drawer = None
        self.setWindowTitle("Koeiro（声彩）")
        from .splash import window_icon
        icon = window_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)
        self.setMinimumSize(1040, 660)
        self.resize(max(self.settings.window_size[0], 1180),
                    max(self.settings.window_size[1], 780))
        # The frame is drawn by the app, not the OS: one surface from the top edge
        # down, with the window controls in the header row (DESIGN.md section 4).
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self._build_ui()
        self._update_meanvc_controls()
        self._restore_position()
        self.controller.set_gain(self.settings.gain_db)
        self.controller.set_gate(self.settings.noise_gate_db)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _select_route(self, key):
        """The segmented route buttons drive the delivery combo, which owns the route.

        The combo stays the single source of truth (worker parameters, settings and
        tests all read it); the buttons are only the visible face of that decision,
        so there is never a second state to disagree.
        """
        if self._pending or self._closing or self.controller.engine.running:
            self._sync_route_buttons()
            return
        index = self.ai_delivery.findData(key)
        if index >= 0 and index != self.ai_delivery.currentIndex():
            self.ai_delivery.setCurrentIndex(index)

    def _sync_route_buttons(self):
        """Mirror the delivery combo back onto the segmented buttons."""
        if not hasattr(self, 'route_buttons'):
            return
        current = self.ai_delivery.currentData()
        for key, button in self.route_buttons.items():
            button.blockSignals(True)
            button.setChecked(key == current)
            button.blockSignals(False)

    def _build_ui(self):
        """One header, one stage, two drawers.

        DESIGN.md section 4: the stage is the only permanent surface. Everything that
        is not the live state -- microphone, way of talking, volume, processing,
        diagnostics -- moves into the 調整 drawer, and the optional features into the
        実験 drawer. Both overlap from the right, so a setting can be changed while the
        live numbers stay on screen. The voice list stays on the right because choosing
        a voice happens in every session, and it folds to zero width on request.
        """
        from . import shell
        from .shell import (Drawer, EdgeResize, HeaderBar, LevelBar, Nameplate,
                            Relay, VoicePanel, stylesheet)
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(container)

        self.header = HeaderBar()
        self.header.tune_clicked.connect(lambda: self._open_drawer('tune'))
        self.header.lab_clicked.connect(lambda: self._open_drawer('lab'))
        self.header.voice_clicked.connect(self._toggle_voice_panel)
        self.header.help_clicked.connect(self._show_tutorial)
        self.header.minimize_clicked.connect(self.showMinimized)
        self.header.maximize_clicked.connect(self._toggle_maximize)
        self.header.close_clicked.connect(self.close)
        outer.addWidget(self.header)
        # The guide lives in the header, so there is one place to look for it.
        self.help_button = self.header.help_button

        body = QWidget()
        body_row = QHBoxLayout(body)
        body_row.setContentsMargins(0, 0, 0, 0)
        body_row.setSpacing(0)
        outer.addWidget(body, 1)

        # The stage scrolls as one piece on a short window; Start sits at its foot.
        stage_scroll = QScrollArea()
        stage_scroll.setWidgetResizable(True)
        stage_scroll.setFrameShape(QFrame.Shape.NoFrame)
        stage_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        stage_body = QWidget()
        stage_layout = QVBoxLayout(stage_body)
        stage_layout.setContentsMargins(28, 18, 28, 18)
        stage_layout.setSpacing(12)
        stage_scroll.setWidget(stage_body)
        body_row.addWidget(stage_scroll, 1)
        self.centre_scroll = stage_scroll

        # Stage state: an error line at the top, hidden unless the engine reported one.
        self.stage_error = QLabel('')
        self.stage_error.setObjectName('stageError')
        self.stage_error.setWordWrap(True)
        self.stage_error.setTextFormat(Qt.TextFormat.PlainText)
        self.stage_error.setVisible(False)
        stage_layout.addWidget(self.stage_error)

        # Stage state: no microphone at all. One line and the one button that helps.
        self.empty_state = QFrame()
        self.empty_state.setObjectName('emptyState')
        empty_column = QVBoxLayout(self.empty_state)
        empty_column.setContentsMargins(18, 14, 18, 14)
        empty_column.setSpacing(8)
        empty_title = QLabel('マイクが見つかりません')
        empty_title.setObjectName('sectionTitle')
        empty_column.addWidget(empty_title)
        empty_hint = QLabel('マイクを接続し、OSのマイク権限を確認してから更新してください。')
        empty_hint.setObjectName('muted')
        empty_hint.setWordWrap(True)
        empty_column.addWidget(empty_hint)
        self.empty_refresh = QPushButton('マイク一覧を更新')
        self.empty_refresh.clicked.connect(self._refresh_devices)
        empty_column.addWidget(self.empty_refresh)
        self.empty_state.setVisible(False)
        stage_layout.addWidget(self.empty_state)

        # Input level: one horizontal strip, filled by the measured peak only.
        level_row = QHBoxLayout()
        level_row.setSpacing(12)
        level_caption = QLabel('入力レベル')
        level_caption.setObjectName('muted')
        level_row.addWidget(level_caption)
        self.input_bar = LevelBar(bars=1)
        level_row.addWidget(self.input_bar, 1)
        self.input_db = QLabel('−∞ dBFS')
        self.input_db.setObjectName('muted')
        self.input_db.setMinimumWidth(96)
        self.input_db.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        level_row.addWidget(self.input_db)
        # The output level is a number, not a second strip: the stage keeps one meter.
        self.output_db = QLabel('出力 −∞ dBFS')
        self.output_db.setObjectName('muted')
        self.output_db.setMinimumWidth(132)
        self.output_db.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        level_row.addWidget(self.output_db)
        stage_layout.addLayout(level_row)

        # The three-point relay answers "where is it stuck" and toggles the transport.
        self.relay = Relay()
        self.relay.tapped.connect(self._toggle_transport)
        stage_layout.addWidget(self.relay)

        # One latency number, with the honest caveat underneath it.
        latency_row = QHBoxLayout()
        latency_row.setSpacing(12)
        self.latency_value = QLabel('-- ms')
        self.latency_value.setObjectName('latencyValue')
        latency_row.addWidget(self.latency_value)
        latency_caption = QLabel('概算のバッファ遅延')
        latency_caption.setObjectName('muted')
        latency_row.addWidget(latency_caption, 1)
        stage_layout.addLayout(latency_row)
        self.latency = QLabel()
        self.latency.setWordWrap(True)
        self.latency.setObjectName('muted')
        self.latency.setTextFormat(Qt.TextFormat.PlainText)
        stage_layout.addWidget(self.latency)

        # Start / Stop at the foot of the stage, never scrolled away from.
        transport = QFrame()
        transport.setObjectName('transportBar')
        transport_layout = QHBoxLayout(transport)
        transport_layout.setContentsMargins(0, 10, 0, 0)
        transport_layout.setSpacing(12)
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        self.start_button.setMinimumHeight(44)
        self.stop_button.setMinimumHeight(44)
        transport_layout.addWidget(self.start_button)
        transport_layout.addWidget(self.stop_button)
        self.status = QLabel("Status: Stopped")
        self.status.setWordWrap(True)
        self.status.setObjectName("muted")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        transport_layout.addWidget(self.status, 1)
        stage_layout.addWidget(transport)
        self.transport_bar = transport
        # What Start will actually use, stated where Start is. This is the stage's
        # own copy: the 調整 drawer's 出力音声 card reuses the same attribute name
        # for its label, so this one keeps a distinct name and is fed beside it.
        self.stage_voice = QLabel('停止中')
        self.stage_voice.setObjectName('stageVoice')
        self.stage_voice.setWordWrap(True)
        self.stage_voice.setTextFormat(Qt.TextFormat.PlainText)
        stage_layout.addWidget(self.stage_voice)
        # The AI state, repeated large: after Start the foot's small readout is
        # easy to miss, so the same state gets one big line in the flow, above
        # the stretch. Same source as the foot (set together in
        # _update_ai_readiness), same palette tokens, no new surface.
        self.ai_banner = QLabel('AI · 未起動')
        self.ai_banner.setObjectName('aiBanner')
        self.ai_banner.setWordWrap(True)
        self.ai_banner.setTextFormat(Qt.TextFormat.PlainText)
        stage_layout.addWidget(self.ai_banner)
        self.ai_banner_note = QLabel('Startを押すとモデルを読み込みます。')
        self.ai_banner_note.setObjectName('muted')
        self.ai_banner_note.setWordWrap(True)
        self.ai_banner_note.setTextFormat(Qt.TextFormat.PlainText)
        stage_layout.addWidget(self.ai_banner_note)
        # The stage foot: the product mark in the room left over at the bottom
        # right. The AI state used to sit on the left here, but it now has the
        # big line above, and two wordings of the same fact read as disagreement.
        stage_layout.addStretch(1)
        foot = QFrame()
        foot.setObjectName('stageFoot')
        foot_row = QHBoxLayout(foot)
        foot_row.setContentsMargins(0, 0, 0, 0)
        foot_row.setSpacing(16)
        foot_row.addStretch(1)
        self.nameplate = Nameplate()
        foot_row.addWidget(self.nameplate, 0, Qt.AlignmentFlag.AlignBottom)
        stage_layout.addWidget(foot)
        self.stage_foot = foot

        # The voice list: right-hand, and foldable to zero width.
        self.voice_panel = VoicePanel()
        body_row.addWidget(self.voice_panel)
        self.setStyleSheet(stylesheet(Path(__file__).parent/'assets'))

        # The two drawers overlap the stage from the right.
        self.drawers = {}
        self.drawer_tune = Drawer(
            '調整', 'マイク、話し方、音量、加工、診断。変更はこの面の中で完結します。',
            container, surface='tune')
        self.drawer_lab = Drawer(
            '実験', '追加機能です。すべて既定OFFで、切っても本体の変換ルート'
                    '（Original / Female DSP / AI Voice）はそのまま動きます。',
            container, surface='lab')
        for key, drawer in (('tune', self.drawer_tune), ('lab', self.drawer_lab)):
            self.drawers[key] = drawer
            drawer.closed.connect(self._drawer_closed)
            drawer.closed.connect(self._sync_header_surfaces)
        tune_layout = self.drawer_tune.body
        lab_layout = self.drawer_lab.body

        # --- 調整 ① 配線: where the sound comes in and where it goes -------------
        self.band_wire = self._band(tune_layout, '🔌 ', '①', '配線',
                                    '音の入り口と出口。マイク、出力先、モニター、音の規格。')
        device_column = self._section(self.band_wire.body, 'マイクを選ぶ')
        device_note = QLabel('あなたの声が入る場所です。')
        device_note.setObjectName('muted')
        device_note.setWordWrap(True)
        device_column.addWidget(device_note)
        form = QFormLayout()
        form.setVerticalSpacing(12)
        self.input_device = QComboBox()
        self.input_device.setMinimumWidth(240)
        self.input_device.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.input_device.setMinimumContentsLength(12)
        form.addRow("マイク入力", self.input_device)
        device_column.addLayout(form)
        self.route_hint = QLabel("デバイスを読み込んでいます…")
        self.route_hint.setWordWrap(True)
        self.route_hint.setObjectName("muted")
        device_column.addWidget(self.route_hint)
        self.refresh = QPushButton("デバイス一覧を更新")
        self.refresh.setToolTip('マイクや仮想ケーブルを接続したあとに押してください。')
        self.refresh.clicked.connect(self._refresh_devices)
        device_column.addWidget(self.refresh)

        output_column = self._section(self.band_wire.body, '出力先を選ぶ')
        output_note = QLabel('変換後の音が出る場所です。Discordに流すときは仮想ケーブル'
                             '（CABLE Input）を選びます。')
        output_note.setObjectName('muted')
        output_note.setWordWrap(True)
        output_column.addWidget(output_note)
        self.output_device = QComboBox()
        self.output_device.setMinimumWidth(240)
        self.output_device.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.output_device.setMinimumContentsLength(12)
        self.output_device.setToolTip('変換後の音が出る場所。Discordに流すときは仮想ケーブル（CABLE Input）を選びます。')
        output_column.addWidget(self.output_device)
        self.output_device.currentIndexChanged.connect(self._route_changed)

        # --- 調整 ② 変換: what the sound becomes ----------------------------------
        self.band_convert = self._band(tune_layout, '🎛 ', '②', '変換',
                                       'その音を何に変えるか。入力の整え、モードと声、'
                                       'モードごとの調整。')

        # --- 調整 ②: volume, gate and the optional denoiser ----------------------
        level_section = self._section(self.band_convert.body, '入力の整え（音量とノイズ）')
        level_note = QLabel('変換の手前で入力を整えます。モードを変えても共通です。')
        level_note.setObjectName('muted')
        level_note.setWordWrap(True)
        level_section.addWidget(level_note)
        # The track stops where GainProcessor still answers: it accepts -20..+20 dB,
        # and a settings file is clamped to the same, so a slider that travelled past
        # -20 dB showed a number the engine refused and silently kept the old level.
        self.gain, self.gain_label = self._slider(level_section, "音量 (Gain)", -200, 120,
                                                  round(self.settings.gain_db * 10))
        self.gate, self.gate_label = self._slider(level_section, "Noise Gate", -800, -100,
                                                 round(self.settings.noise_gate_db * 10))
        self.gain.valueChanged.connect(self._gain_changed)
        self.gate.valueChanged.connect(self._gate_changed)
        self._gain_changed(self.gain.value())
        self._gate_changed(self.gate.value())
        # Optional spectral noise reduction, applied live like the gate. It is off by
        # default because it reshapes the voice; the strength is a user choice.
        self.denoise_toggle = QCheckBox('ノイズ除去（定常ノイズを抑える・実験的）')
        self.denoise_toggle.setChecked(self.settings.denoise)
        self.denoise_toggle.setToolTip('ファン・サー音などの定常ノイズを抑えます。\n'
                                       '声質が変わり得るため既定はOFF。停止中でも切替できます。')
        self.denoise_toggle.toggled.connect(self._denoise_changed)
        level_section.addWidget(self.denoise_toggle)
        self.denoise_strength, self.denoise_strength_label = self._slider(
            level_section, 'ノイズ除去の強さ', 0, 100, round(self.settings.denoise_strength * 100))
        self.denoise_strength.valueChanged.connect(self._denoise_changed)
        self.denoise_strength_label.setText(f'{self.denoise_strength.value()/100:.2f}')
        self._denoise_changed()
        # Optional input levelling, applied live like the gate. Small voices
        # come up to the level the model grips best; silence is never boosted.
        self.auto_level_toggle = QCheckBox('音量の自動均一化（小声を持ち上げる）')
        self.auto_level_toggle.setChecked(self.settings.auto_level)
        self.auto_level_toggle.setToolTip('入力を一定の音量にそろえます（無音は上げません）。\n'
                                          '既定はOFF。停止中でも変換中でも切替できます。')
        self.auto_level_toggle.toggled.connect(self._auto_level_changed)
        level_section.addWidget(self.auto_level_toggle)
        self._auto_level_changed()
        # Environmental sound routing for the AI route: broadband transients
        # (claps, keyboard, knocks) reach the listener raw instead of being
        # re-synthesised. Only the streaming route can align a raw copy against
        # converted output, so the toggle has no effect elsewhere.
        self.env_passthrough_toggle = QCheckBox('環境音をそのまま通す（拍手・キーボードなど）')
        self.env_passthrough_toggle.setChecked(self.settings.env_passthrough)
        self.env_passthrough_toggle.setToolTip('拍手・キーボード・物音などの一過性の環境音を、変換せずそのまま出力します。\n'
                                               '咳・鼻すすり・くしゃみなど声まわりの音は環境音とみなさず変換します。\n'
                                               '声が戻るとすぐに通常の変換へ戻ります。既定はON。\n'
                                               'AI Voice・逐次変換のときだけ働きます。')
        self.env_passthrough_toggle.toggled.connect(self._env_passthrough_changed)
        level_section.addWidget(self.env_passthrough_toggle)
        self._env_passthrough_changed()

        # --- 調整 ②: the voice and the way of talking -----------------------------
        sound_column = self._section(self.band_convert.body, '声と話し方を選ぶ')
        sound_note = QLabel('誰の声にするか（右の声パネル）と、待つか待たないかを決めます。')
        sound_note.setObjectName('muted')
        sound_note.setWordWrap(True)
        sound_column.addWidget(sound_note)
        mode_row = QFormLayout()
        mode_row.setVerticalSpacing(12)
        self.mode = QComboBox()
        self.mode.addItem('Original / 変換なし', 'original')
        self.mode.addItem('Female DSP / 軽い加工', 'female_dsp')
        self.mode.addItem('AI Voice / 声の変換', 'ai_voice')
        self.mode.addItem('同時通訳 / English voice', 'translate')
        self.mode.setToolTip('声の変換方法を選びます。\n'
                          'Original: 変換しません\n'
                          'Female DSP: 軽い加工（CPUが軽い）\n'
                          'AI Voice: 選んだ声で完全に別の声に変換\n'
                          '同時通訳: 日本語の発話を英語音声に変えて出力します（実験室②が必要）')
        self.mode.setMinimumHeight(34)
        mode_row.addRow('変換モード', self.mode)
        self.voice_quick = QComboBox()
        self.voice_quick.setMinimumHeight(34)
        self.voice_quick.setToolTip('変換先の声です。右の声パネルと同じ選択を共有します。')
        self.voice_quick.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.voice_quick.setMinimumContentsLength(16)
        mode_row.addRow('変換したい声', self.voice_quick)
        sound_column.addLayout(mode_row)
        self.voice_quick_note = QLabel('')
        self.voice_quick_note.setObjectName('muted')
        self.voice_quick_note.setWordWrap(True)
        sound_column.addWidget(self.voice_quick_note)
        panel_jump = QPushButton('右の声パネルで選ぶ →')
        panel_jump.setObjectName('panelJump')
        panel_jump.setToolTip('右の声パネルに移動し、検索欄にカーソルを置きます。')
        panel_jump.clicked.connect(self._focus_voice_panel)
        sound_column.addWidget(panel_jump)
        self.panel_jump = panel_jump
        self._build_voice_data()
        self._build_route_control(sound_column)
        self._build_voice_panel()

        # --- 調整 ①: monitor (the by-ear check) ----------------------------------
        # Device, switch, level and test answer one question, so they share one card
        # instead of being split between this and the audio-format card.
        monitor_card = QFrame()
        monitor_card.setObjectName('outputCard')
        monitor_column = QVBoxLayout(monitor_card)
        monitor_column.setContentsMargins(16, 12, 16, 12)
        monitor_column.setSpacing(6)
        monitor_heading = QLabel('モニター（変換後の音を自分の耳で確認）')
        monitor_heading.setObjectName('sectionTitle')
        monitor_column.addWidget(monitor_heading)
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
        monitor_column.addLayout(monitor_row)
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
        # Same band as the microphone and the output device: the three questions are
        # "what comes in", "where it goes" and "what does it sound like there".
        self.band_wire.body.addWidget(monitor_card)
        self.monitor_volume.valueChanged.connect(self._monitor_volume_changed)
        self.monitor_device.currentIndexChanged.connect(self._monitor_changed)
        self._monitor_test_note = None
        self._monitor_test_busy = False
        self._monitor_apply = QTimer(self)
        self._monitor_apply.setSingleShot(True)
        self._monitor_apply.setInterval(120)
        self._monitor_apply.timeout.connect(self._apply_monitor)
        self._update_monitor_ui()

        # --- 調整 ①: sample rate, buffer and the recommended values --------------
        device_details = self._section(self.band_wire.body, '音の規格（Sample Rate / Buffer）')
        details_note = QLabel('48 kHz / 256 が標準です。変えると変換の品質とCPU負荷が変わります。')
        details_note.setObjectName('muted')
        details_note.setWordWrap(True)
        device_details.addWidget(details_note)
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
        device_details.addLayout(options)
        self.recommended = QPushButton('推奨設定を適用（48 kHz / 256 / 標準ボイス）')
        self.recommended.setToolTip('このPCで実測した推奨値にそろえます。デバイスの選択は変わりません。')
        self.recommended.clicked.connect(self._apply_recommended)
        device_details.addWidget(self.recommended)

        # --- 調整 ②: the mode-specific parts of the conversion -------------------
        self._build_dsp_section(self.band_convert.body)
        self._build_ai_details(self.band_convert.body)
        self._build_prosody_section(self.band_convert.body)
        self._build_finish_section(self.band_convert.body)
        self._build_ai_formant_section(self.band_convert.body)

        # --- 調整 ③ アプリ: saving and managing ----------------------------------
        self.band_app = self._band(tune_layout, '⚙️ ', '③', 'アプリ',
                                   'このセッションの音とは別の管理。シーン、更新、Discord、診断。')
        # Order asked in: what profile am I in, is the app current, do I broadcast,
        # and last the measured numbers nobody reads mid-session.
        self._build_scenes_section(self.band_app.body)
        self._build_update_section(self.band_app.body)
        self._build_discord_section(self.band_app.body)
        self._build_diagnostics_section(self.band_app.body)

        # --- 実験: every optional feature behind one switch -------------------------
        self._build_lab_drawer(lab_layout)

        # Each drawer indexes itself: the search box and the jump row read the cards
        # that ended up in it, so neither can list something that is not there.
        self.drawer_tune.build_index()
        self.drawer_lab.build_index()

        self.mode.setCurrentIndex(self.mode.findData(self.settings.voice_mode))
        self._publish_dsp()
        self.input_device.currentIndexChanged.connect(self._route_changed)
        self.output_device.currentIndexChanged.connect(self._route_changed)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.rate.currentIndexChanged.connect(self._update_latency)
        self.buffer.currentIndexChanged.connect(self._update_latency)
        self._update_latency()
        for combo in (self.input_device, self.output_device, self.rate, self.buffer, self.monitor_device):
            combo.setMinimumHeight(34)
        self.monitor_toggle.setMinimumHeight(34)

        self._guide_open = False
        self._guide_overlay = None
        self._active_drawer = None
        self._drawer_place = QTimer(self)
        self._drawer_place.setSingleShot(True)
        self._drawer_place.timeout.connect(self._place_drawers)
        self._build_hotkeys()
        # A frameless window gets no border from the OS to drag either: the app
        # supplies the pointer and the resize start. Installed once the widgets
        # exist, so their pointer tracking is switched on with it.
        self.edge_resize = EdgeResize(self)
        self._ai_readiness_key = None
        self._schedule_first_run_tutorial(splash_played=False)
        QTimer.singleShot(8000, self._auto_update_check)

    def _drawer_closed(self):
        """A drawer closed itself (its 閉じる button); drop the reference to it."""
        self._active_drawer = None
        self._sync_header_surfaces()

    def _sync_header_surfaces(self):
        """Tint the one bar button whose surface is on screen."""
        self.header.set_surface_active('tune', self._active_drawer == 'tune')
        self.header.set_surface_active('lab', self._active_drawer == 'lab')
        self.header.set_surface_active('voice', self.voice_panel.isVisible())

    def _focus_drawer_search(self):
        """Ctrl+F: the search box of the drawer that is open, if one is."""
        drawer = self.drawers.get(self._active_drawer)
        if drawer is not None and drawer.isVisible():
            drawer.focus_search()

    def _place_drawers(self):
        """Keep every open drawer against the right edge of the window."""
        container = self.centralWidget()
        if container is None:
            return
        for drawer in self.drawers.values():
            if drawer.isVisible():
                drawer.place(container)

    def _open_drawer(self, name):
        """Open one drawer and close the other, or close both when name is None."""
        for key, drawer in self.drawers.items():
            if key == name:
                drawer.open()
                drawer.place(self.centralWidget())
            elif drawer.isVisible():
                drawer.dismiss()
        self._active_drawer = name
        self._sync_header_surfaces()

    def _toggle_voice_panel(self, collapsed=None):
        """Fold the voice list to zero width, or open it again."""
        panel = self.voice_panel
        if collapsed is None:
            collapsed = panel.isVisible()
        panel.set_collapsed(collapsed)
        self.header.set_voice_checked(not collapsed)
        self.header.set_surface_active('voice', not collapsed)
        self._place_drawers()

    def _toggle_maximize(self):
        """Maximise or restore, from the header button or a double click on the bar."""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
        self.header.set_maximized(self.isMaximized())

    def changeEvent(self, event):
        """Keep the maximise glyph right however the state changed.

        Snapping, a keyboard toggle and the platform all change the state without
        going through the button, so the button follows the window, not the click.
        """
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange and hasattr(self, 'header'):
            self.header.set_maximized(self.isMaximized())

    def _release_edge_resize(self):
        """Take the app-wide filter off before this window goes away."""
        filter_ = getattr(self, 'edge_resize', None)
        application = QApplication.instance()
        if filter_ is not None and application is not None:
            application.removeEventFilter(filter_)
            self.edge_resize = None

    def _update_ai_readiness(self):
        """Say whether the selected route is ready, from its own live state.

        The stage states the fact before a Start needs it: not loaded, loading, ready,
        error, or the two states where it will not be used at all (a sample rate other
        than 48 kHz, or a machine where the model cannot load). Nothing here estimates
        progress; the route reports states, so states are all that is shown. The one
        place it is stated is the big line above the stage foot. The translate route
        reports the translation driver instead of the voice bridge, which stays
        parked there by design.
        """
        from .shell import AiReadiness
        if not hasattr(self, 'ai_banner'):
            return
        if self.router is None:
            state, detail = 'unavailable', None
        elif self.mode.currentData() == 'translate':
            state, detail = self._translate_readiness()
        else:
            bridge = self.router.ai.bridge
            supported = bool(getattr(self.router.ai, 'supported', True))
            if self.startup_diagnostic and not self.startup_diagnostic.get('ai_available', True):
                state, detail = 'unavailable', None
            elif not supported:
                state, detail = 'unavailable', None
            elif bridge.status == 'Error':
                state, detail = 'error', (bridge.error or 'モデルを読み込めませんでした。')
            elif bridge.status in ('Loading', 'Starting'):
                state, detail = 'loading', None
            elif self.rate.currentData() != 48000:
                state, detail = 'fallback', None
            elif bridge.status == 'Ready':
                state, detail = 'ready', None
            else:
                state, detail = 'stopped', None
        key = (state, detail)
        if key != self._ai_readiness_key:
            self._ai_readiness_key = key
            label, note = AiReadiness.TEXT.get(state, AiReadiness.TEXT['stopped'])
            self.ai_banner.setText(f'AI · {label}')
            self.ai_banner_note.setText(note if detail is None else detail)
            self.ai_banner.setProperty('state', state)
            self.ai_banner.style().unpolish(self.ai_banner)
            self.ai_banner.style().polish(self.ai_banner)

    def _translate_readiness(self):
        """The translate route's readiness, from the translation driver.

        The voice bridge stays parked on this route by design, so its Stopped
        state would wrongly claim nothing is ready. Every branch names the next
        concrete step instead.
        """
        if self.rate.currentData() != 48000:
            return ('fallback', '同時通訳は48 kHz専用です。Sample Rateを48000 Hzにしてください。')
        if not self.lab_enabled.isChecked():
            return ('stopped', '「実験」の「実験機能有効」をONにすると同時通訳が動きます。')
        if self._capture_lab().translation_provider == 'off':
            return ('stopped', '「実験」の②で合成（Kokoro）を選ぶと同時通訳が動きます。')
        driver = getattr(self, 'translate_driver', None)
        status = driver.status if driver is not None else {}
        if status.get('error'):
            return ('error', f'同時通訳エラー: {status["error"]}')
        if not self.controller.engine.running:
            return ('stopped', 'Startを押すと同時通訳が始まります（文単位・遅延大）。')
        if not driver.alive:
            return ('loading', '同時通訳を準備しています…')
        if status.get('asr') in ('Loading', 'Starting', 'preparing'):
            return ('loading', '音声認識を読込中…しばらくお待ちください。')
        heard, said = status.get('heard', ''), status.get('said', '')
        if said:
            return ('ready', f'「{heard}」→「{said}」')
        if heard:
            return ('ready', f'「{heard}」を翻訳・合成しています…')
        if status.get('mic_silent'):
            return ('ready', 'マイクの音が来ていません。入力デバイスと音量を確認してください。')
        return ('ready', '聞き取っています…文を言い切ってください。')

    def resizeEvent(self, event):
        """Open drawers follow the right edge; no widget is positioned by hand."""
        super().resizeEvent(event)
        self._place_drawers()

    def showEvent(self, event):
        super().showEvent(event)
        self._place_drawers()
        self._ensure_presence()

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
                                    '右の声パネルで別の声を選ぶと変わります。')
        else:
            self.pitch_note.setText('ピッチ変更は Female DSP のときだけ使えます。')
        self.pitch_note.setVisible(True)

    def _build_output_card(self, layout):
        # No glyph: the title already says what this is, and an emoji would
        # decorate rather than inform.
        card = QFrame()
        card.setObjectName('outputCard')
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(18, 14, 18, 14)
        card_layout.setSpacing(14)
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
        layout.addWidget(card)

    def _rail_pitch_changed(self, value):
        self.pitch_value.setText('%+.1f st' % (value/4.0))
        if self.pitch_display.isEnabled():
            self.pitch.setValue(round(value*2.5))
            self._publish_dsp()

    def _show_tutorial(self):
        """Open the guide: the window dims and one control at a time stays lit.

        The guide is part of this window (a child of the central widget), so the splash
        owns the startup moment and hands over to it, and a headless render can never
        block on it. It covers the window while it runs, so a first run cannot switch a
        setting by accident while the explanation is on screen.
        """
        if getattr(self, '_guide_open', False):
            return
        from .tutorial import TutorialOverlay
        self._guide_open = True
        overlay = TutorialOverlay(self.centralWidget(), on_page=self._guide_page)
        overlay.finished.connect(self._guide_finished)
        self._guide_overlay = overlay
        self.header.set_surface_active('guide', True)
        overlay.start()

    def _guide_finished(self, _completed=False):
        """Read the reader's choice once the guide closes, then clean up."""
        overlay, self._guide_overlay = getattr(self, '_guide_overlay', None), None
        self._guide_open = False
        self.header.set_surface_active('guide', False)
        if overlay is None:
            return
        hide = overlay.hide_next_time
        overlay.deleteLater()
        if hide and not self.settings.tutorial_seen:
            self.settings = replace(self.settings, tutorial_seen=True)
            self._save_settings()

    def _guide_page(self, spec, overlay):
        """Follow the guide: open the surface it describes and light the control."""
        if spec.get('voice'):
            self._toggle_voice_panel(collapsed=False)
        self._open_drawer(spec.get('drawer'))
        target = spec.get('target')
        # The overlay scrolls the control into view itself: a step about a setting has
        # to scroll the drawer it lives in, not the stage behind it.
        overlay.light(getattr(self, target, None) if target else None)

    def _maybe_show_tutorial(self):
        # Automated runs must never block on the modal guide.
        if self._closing or self.settings.tutorial_seen or "PYTEST_CURRENT_TEST" in os.environ:
            return
        self._show_tutorial()

    def _schedule_first_run_tutorial(self, splash_played=True):
        """Wait out the splash before the first-run guide, or those two overlap.

        The guide is a window of its own, so it is not covered by the splash overlay
        the way a child widget would be: opened too early it appears *on top of* the
        animation. ``splash_duration_ms()`` is the timeline the splash actually plays
        (0 when there is no splash to play), so the wait cannot drift from it.
        """
        from .splash import splash_duration_ms
        del splash_played  # the splash itself is the authority on its own length
        QTimer.singleShot(splash_duration_ms() + 400, self._maybe_show_tutorial)

    def _toggle_transport(self):
        if self.controller.engine.running:
            self._stop()
        else:
            self._start()

    def _lab_card(self, layout, title, hint=''):
        """A settings-style card for one lab feature."""
        card = QFrame()
        card.setObjectName('outputCard')
        column = QVBoxLayout(card)
        column.setContentsMargins(18, 14, 18, 16)
        column.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName('sectionTitle')
        column.addWidget(heading)
        if hint:
            note = QLabel(hint)
            note.setObjectName('muted')
            note.setWordWrap(True)
            note.setTextFormat(Qt.TextFormat.PlainText)
            column.addWidget(note)
        layout.addWidget(card)
        return card, column

    @staticmethod
    def _lab_status(column, text=''):
        label = QLabel(text)
        label.setObjectName('muted')
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        column.addWidget(label)
        return label

    def _voice_folder(self, voice_id):
        """The folder of a registered voice, or None for the shipped/none selection."""
        from src.runtime_paths import asset_root
        from src.vc.models import VOICE_PROFILES
        profile = VOICE_PROFILES.get(voice_id)
        if not profile or not profile.get('user_voice'):
            return None
        folder = asset_root() / 'models' / profile['folder']
        return folder if folder.is_dir() else None

    @staticmethod
    def _fill_voice_combo(combo, selected=''):
        """List the registered voices, preferring the one already selected."""
        from src.vc.models import VOICE_PROFILES
        current = selected or combo.currentData() or ''
        combo.blockSignals(True)
        combo.clear()
        combo.addItem('（未選択）', '')
        for key, profile in VOICE_PROFILES.items():
            combo.addItem(profile.get('name') or key, key)
        index = combo.findData(current)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)

    def _lab_feature(self, layout, key, title, hint, switch):
        """One optional feature: a titled row with its own switch.

        The switch is created by the caller so every feature keeps the widget the rest
        of the app already drives. The body is returned empty: the caller fills it, and
        it is only shown once the master switch and this feature's switch are both on.
        """
        card = QFrame()
        card.setObjectName('outputCard')
        column = QVBoxLayout(card)
        column.setContentsMargins(18, 14, 18, 16)
        column.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName('sectionTitle')
        # Long feature names must wrap: a one-line heading sets the card's
        # minimum width and pushed the whole lab drawer past its right edge.
        heading.setWordWrap(True)
        head.addWidget(heading, 1)
        head.addWidget(switch)
        column.addLayout(head)
        note = QLabel(hint)
        note.setObjectName('muted')
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.PlainText)
        column.addWidget(note)
        body = QWidget()
        body_column = QVBoxLayout(body)
        body_column.setContentsMargins(0, 6, 0, 2)
        body_column.setSpacing(6)
        column.addWidget(body)
        layout.addWidget(card)
        self.lab_rows[key] = (card, body, switch)
        return body_column

    def _update_lab_visibility(self):
        """Heads and switches always; bodies only for features in use.

        Every feature keeps its ON/OFF where it can always be reached: hiding
        the switches behind the master switch left running features with no
        visible way to stop them. The master decides whether anything takes
        effect; the bodies (settings and run buttons) still open only for
        features that are both armed and enabled.
        """
        enabled = self.lab_enabled.isChecked()
        self.lab_summary.setVisible(not enabled)
        for _key, (card, body, switch) in self.lab_rows.items():
            card.setVisible(True)
            body.setVisible(enabled and switch.isChecked())

    def _build_lab_drawer(self, layout):
        """実験: the optional features behind one master switch.

        Five features used to be five open cards. Now the drawer lists what exists, the
        master switch decides whether they can be used at all, and only the features a
        reader turned on show their settings and run buttons.
        """
        lab = self.lab_parameters
        self.lab_rows = {}
        self.lab_enabled = QCheckBox('実験機能有効（追加機能を使う）')
        self.lab_enabled.setChecked(bool(lab.enabled))
        self.lab_enabled.setToolTip('既定はOFF。OFFの間はこの一覧だけで、設定と実行ボタンは出しません。\n'
                                    '切っても本体の変換ルート（Original / Female DSP / AI Voice）は変わりません。')
        self.lab_enabled.toggled.connect(self._lab_changed)
        layout.addWidget(self.lab_enabled)
        self.lab_summary = QLabel(
            'ここに入っているもの:\n'
            '⏱️ ① レイテンシ・ガバナ（計測して品質を提案）\n'
            '🌐 ② 同時通訳ボイス（翻訳して喋る。翻訳モデルは将来のパック）\n'
            '🤫 ③ 囁きモード（囁きに周期成分を与える）\n'
            '📡 ④ 分散推論（LANのGPU機に音声合成・翻訳を任せる。認証なし・LAN内のみ）\n'
            '👥 ⑤ 複数話者の自動振り分け（声の高さと明るさで分ける。近い声は分けられません）\n\n'
            'まず上の「実験機能有効」をONにすると、各機能のON/OFFが効くようになります。'
            '使う機能をONにすると、その機能の中身がここに開きます。')
        self.lab_summary.setObjectName('muted')
        self.lab_summary.setWordWrap(True)
        self.lab_summary.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.lab_summary)

        # ① latency governor: measure, then suggest. Applying stays the reader's call.
        self.governor_on = QCheckBox('計測する')
        self.governor_on.setChecked(lab.governor)
        self.governor_auto = QCheckBox('停止中なら自動で反映する')
        self.governor_auto.setChecked(lab.governor_auto)
        self.governor_auto.setToolTip('提案が変わったとき、停止中に限り品質コンボを切り替えます。')
        self.governor_apply = QPushButton('提案を今すぐ反映')
        self.governor_apply.clicked.connect(self._lab_apply_governor)
        self.governor_on.toggled.connect(self._lab_changed)
        self.governor_auto.toggled.connect(self._lab_changed)
        column = self._lab_feature(
            layout, 'governor', '⏱️ ① レイテンシ・ガバナ（計測して品質を提案）',
            '変換コールバックが締切に間に合っているかを測り、このPCに合う品質プロファイルを提案します。'
            '提案が既定で、反映はあなたの操作。品質の変更は停止中のみ有効です。', self.governor_on)
        column.addWidget(self.governor_auto)
        row = QHBoxLayout()
        row.addWidget(self.governor_apply)
        row.addStretch(1)
        column.addLayout(row)
        self.governor_status = self._lab_status(column, '計測待ち（変換を開始すると測ります）')

        # ② translation voice: the switch is the provider; off means silence, not a
        # half-configured route.
        self.translate_on = QCheckBox('使う')
        column = self._lab_feature(
            layout, 'translate', '🌐 ② 同時通訳ボイス（翻訳して喋る）',
            '入力した文を合成してWAVに書き出します（試聴は各自で）。変換モードで'
            '「同時通訳」を選ぶと、話した日本語が英語音声になって出力先（Discordなど）'
            'へ流れます。翻訳モデルは拡張パック側の枠で、未導入の今は入力文をそのまま'
            '読み上げます。TTSは専用環境 vc_models/tts/.venv で動きます。',
            self.translate_on)
        form = QFormLayout()
        self.translate_provider = QComboBox()
        for label, key in (('Off（素通し・無音）', 'off'), ('Kokoro（英語TTS）', 'kokoro')):
            self.translate_provider.addItem(label, key)
        self.translate_provider.setCurrentIndex(self.translate_provider.findData(lab.translation_provider))
        self.translate_provider.currentIndexChanged.connect(self._lab_changed)
        form.addRow('合成', self.translate_provider)
        self.translate_voice = QLineEdit(lab.translate_voice or 'af_heart')
        self.translate_voice.setToolTip('Kokoro の声名（例 af_heart / am_michael）。')
        self.translate_voice.editingFinished.connect(self._lab_changed)
        form.addRow('TTSの声', self.translate_voice)
        self.translate_reference = QComboBox()
        self._fill_voice_combo(self.translate_reference, lab.translate_reference)
        self.translate_reference.currentIndexChanged.connect(self._lab_changed)
        form.addRow('声色を寄せるボイス', self.translate_reference)
        column.addLayout(form)
        self.translate_preserve = QCheckBox('声色を寄せる（選んだ登録ボイスへ寄せる）')
        self.translate_preserve.setChecked(lab.translate_preserve)
        self.translate_preserve.toggled.connect(self._lab_changed)
        column.addWidget(self.translate_preserve)
        self.translate_test_button = QPushButton('テスト発話を書き出す（「こんにちは、今日はいい天気ですね」）')
        self.translate_test_button.clicked.connect(self._lab_translate_test)
        column.addWidget(self.translate_test_button)
        self.translate_status = self._lab_status(column, '未実行')
        # Live route: speak Japanese and the English voice goes to the output
        # device (Discord). Same pipeline as the WAV test above, but the
        # translate route plays it instead of writing a file. Sentences, not
        # crosstalk: synthesis takes seconds per line on a weak CPU.
        self.translate_live_status = self._lab_status(column, '同時通訳ルートを選んでStartするとここに状態が出ます')
        # The switch and the provider are the same decision, so they follow each other.
        self.translate_on.setChecked(lab.translation_provider != 'off')
        self.translate_on.toggled.connect(self._translate_toggled)

        # ③ whisper mode: restore a periodic component the conversion can grip.
        self.whisper_on = QCheckBox('使う')
        self.whisper_on.setChecked(lab.whisper)
        self.whisper_on.toggled.connect(self._lab_changed)
        column = self._lab_feature(
            layout, 'whisper', '🤫 ③ 囁きモード（囁きに声を与える）',
            '囁き声は声の変換モデルが苦手です（声帯の振動が無いため）。使うと囁きを'
            '自動で見分け、周期成分を足して変換側が掴めるピッチを作ります。通常の声・'
            '無音・息の音（サ行など）はそのまま通します。', self.whisper_on)
        self.whisper_amount, self.whisper_amount_label = self._slider(
            column, '混ぜる量', 0, 100, round(lab.whisper_amount * 100))
        self.whisper_amount.valueChanged.connect(self._lab_changed)
        self.whisper_pitch, self.whisper_pitch_label = self._slider(
            column, '与えるピッチ', 80, 400, round(lab.whisper_pitch))
        self.whisper_pitch.valueChanged.connect(self._lab_changed)
        self.whisper_note = self._lab_status(column, '停止中でも切替できます。')

        # ④ distributed inference: the worker and its host.
        self.remote_enabled = QCheckBox('使う')
        self.remote_enabled.setChecked(lab.remote)
        self.remote_enabled.toggled.connect(self._lab_changed)
        column = self._lab_feature(
            layout, 'remote', '📡 ④ 分散推論（LANのGPU機に任せる）',
            '強いPCで python tools/remote_worker.py を起動し、そのホストを指定します。'
            '今できるのは音声合成と翻訳の依頼までで、声変換モデルのLAN配信は未対応です（認証も無いのでLAN内のみ）。',
            self.remote_enabled)
        form = QFormLayout()
        self.remote_host = QLineEdit(lab.remote_host)
        self.remote_host.setPlaceholderText('192.168.0.10 または gpu-box.local')
        self.remote_host.editingFinished.connect(self._lab_changed)
        form.addRow('ホスト', self.remote_host)
        self.remote_port = QSpinBox()
        self.remote_port.setRange(1, 65535)
        self.remote_port.setValue(lab.remote_port)
        self.remote_port.valueChanged.connect(self._lab_changed)
        form.addRow('ポート', self.remote_port)
        column.addLayout(form)
        self.remote_test_button = QPushButton('接続テスト')
        self.remote_test_button.clicked.connect(self._lab_remote_test)
        column.addWidget(self.remote_test_button)
        self.remote_status = self._lab_status(column, '未接続')

        # ⑤ speaker routing: two voices, measured with the real reference files.
        self.diarization_on = QCheckBox('話者で振り分ける')
        self.diarization_on.setChecked(lab.diarization)
        self.diarization_on.toggled.connect(self._lab_changed)
        column = self._lab_feature(
            layout, 'diarization', '👥 ⑤ 複数話者の自動振り分け（相方と自分の声を分ける）',
            'マイクに複数人が映るとき、声の高さと明るさで話者を見分け、話者ごとに別の声を割り当てます。'
            '「識別テスト」は登録ボイスの参照音声で実際に測ります。近い声は分けられません。',
            self.diarization_on)
        form = QFormLayout()
        self.diarization_speakers = QSpinBox()
        self.diarization_speakers.setRange(2, 4)
        self.diarization_speakers.setValue(lab.diarization_speakers)
        self.diarization_speakers.valueChanged.connect(self._lab_changed)
        form.addRow('話者数', self.diarization_speakers)
        self.speaker_a = QComboBox()
        self.speaker_b = QComboBox()
        self._fill_voice_combo(self.speaker_a)
        self._fill_voice_combo(self.speaker_b)
        form.addRow('話者1の声', self.speaker_a)
        form.addRow('話者2の声', self.speaker_b)
        column.addLayout(form)
        self.speaker_test_button = QPushButton('識別テスト（参照音声2つで測る）')
        self.speaker_test_button.clicked.connect(self._lab_speaker_test)
        column.addWidget(self.speaker_test_button)
        self.speaker_status = self._lab_status(column, '未実行')
        self._update_lab_visibility()
        self._lab_changed()

    def _translate_toggled(self, enabled):
        """One decision in two widgets: the row switch and the provider combo."""
        if not hasattr(self, 'translate_provider'):
            return
        wanted = 'kokoro' if enabled else 'off'
        index = self.translate_provider.findData(wanted)
        if index >= 0 and index != self.translate_provider.currentIndex():
            self.translate_provider.blockSignals(True)
            self.translate_provider.setCurrentIndex(index)
            self.translate_provider.blockSignals(False)
        self._lab_changed()

    def _capture_lab(self):
        """The lab settings exactly as the widgets show them."""
        provider = self.translate_provider.currentData()
        voice = (self.translate_voice.text() or '').strip()[:128]
        return LabParameters(
            enabled=self.lab_enabled.isChecked(),
            governor=self.governor_on.isChecked(),
            governor_auto=self.governor_auto.isChecked(),
            translation_provider=provider if provider in ('off', 'kokoro') else 'off',
            translate_voice=voice,
            translate_reference=(self.translate_reference.currentData() or ''),
            translate_preserve=self.translate_preserve.isChecked(),
            whisper=self.whisper_on.isChecked(),
            whisper_amount=self.whisper_amount.value() / 100,
            whisper_pitch=float(self.whisper_pitch.value()),
            remote=self.remote_enabled.isChecked(),
            remote_host=(self.remote_host.text() or '').strip()[:253],
            remote_port=int(self.remote_port.value()),
            diarization=self.diarization_on.isChecked(),
            diarization_speakers=int(self.diarization_speakers.value()))

    def _lab_changed(self, *_):
        """Push the live-safe parts now; the rest is read at save/start time."""
        if not hasattr(self, 'whisper_on'):
            return
        self.lab_parameters = self._capture_lab()
        # The master switch gates effect, not the choice itself: a feature left
        # ON while the master is OFF stays armed in the settings but does nothing.
        effective_whisper = (self.whisper_on.isChecked() and self.lab_enabled.isChecked())
        if self.whisper is not None:
            self.whisper.amount = (self.whisper_amount.value() / 100
                                   if effective_whisper else 0.0)
            self.whisper.set_control(pitch_hz=float(self.whisper_pitch.value()))
            self.whisper_note.setText(f'囁きモード: {"ON" if effective_whisper else "OFF"}'
                                      f' · ピッチ {self.whisper_pitch.value()} Hz'
                                      f' · 混ぜ {self.whisper_amount.value()}%'
                                      + ('' if self.lab_enabled.isChecked()
                                         else '（実験機能有効がOFFのため無効）'))
        else:
            self.whisper_note.setText('この構成では囁きステージを差し込めません（外部チェーン）。')
        self.whisper_amount_label.setText(f'{self.whisper_amount.value()}%')
        self.whisper_pitch_label.setText(f'{self.whisper_pitch.value()} Hz')
        # The row switch and the provider combo are the same decision, so whichever one
        # the reader touched, both end up agreeing.
        wanted = self.translate_provider.currentData() != 'off'
        if self.translate_on.isChecked() != wanted:
            self.translate_on.blockSignals(True)
            self.translate_on.setChecked(wanted)
            self.translate_on.blockSignals(False)
        self.translate_test_button.setEnabled(wanted)
        self._update_lab_visibility()

    def _lab_apply_governor(self):
        """Apply the governor's profile to the quality combos when that is legal."""
        decision = self.governor.last
        if decision is None:
            self.governor_status.setText('まだ計測していません。変換を開始すると測ります。')
            return
        if self.controller.engine.running:
            self.governor_status.setText('変換中は品質を変えられません。停止してから反映してください。')
            return
        profile = decision.profile
        # Both routes carry the same profile names, so one decision moves both.
        for combo in (getattr(self, 'ai_quality', None), getattr(self, 'quality', None)):
            if combo is None:
                continue
            index = combo.findData(profile)
            if index >= 0 and index != combo.currentIndex():
                combo.setCurrentIndex(index)
        self.governor_status.setText(f'反映しました: {decision.label}（{decision.reason}）')

    def _lab_remote_test(self):
        host = (self.remote_host.text() or '').strip()
        port = int(self.remote_port.value())
        self.remote_status.setText('接続しています…')
        self.files.submit('lab_remote', lambda: lab_actions.probe_worker(host, port, 3.0))

    def _lab_translate_test(self):
        # The TTS weights and its environment live in the installed tree, not in the
        # bundled one, and the WAV is user output: neither belongs under _internal.
        from src.runtime_paths import data_dir, model_root
        lab = self._capture_lab()
        reference = None
        if lab.translate_preserve:
            folder = self._voice_folder(lab.translate_reference)
            if folder is None:
                self.translate_status.setText('声色を寄せるには、②の「声色を寄せるボイス」に登録ボイスを選んでください。')
                return
            reference = folder / 'reference.wav'
        output = Path(data_dir()) / 'lab' / 'translate_test.wav'
        provider, voice = lab.translation_provider, lab.translate_voice
        self.translate_status.setText('生成しています（CPUでは数十秒かかることがあります）…')
        self.files.submit('lab_translate', lambda: lab_actions.translate_test(
            model_root(), provider, voice, reference, 'こんにちは、今日はいい天気ですね', output))

    def _lab_speaker_test(self):
        folder_a = self._voice_folder(self.speaker_a.currentData())
        folder_b = self._voice_folder(self.speaker_b.currentData())
        if folder_a is None or folder_b is None or folder_a == folder_b:
            self.speaker_status.setText('異なる登録ボイスを2つ選んでください（標準ボイスは対象外です）。')
            return
        speakers = int(self.diarization_speakers.value())
        self.speaker_status.setText('測定しています…')
        self.files.submit('lab_speakers', lambda: lab_actions.speaker_test(
            folder_a, folder_b, speakers))

    def _lab_results(self):
        """Drain the lab background jobs. Never touches a widget from the worker."""
        for key, label in (('lab_translate', self.translate_status),
                           ('lab_remote', self.remote_status),
                           ('lab_speakers', self.speaker_status)):
            message = self.files.results.pop(key, None)
            if message is None:
                continue
            label.setText(message if not message.startswith('ERROR: ') else message[7:])

    def _update_translation_driver(self):
        """Arm the live translation voice from the route, or park it.

        The driver (its ASR worker, thread and TTS jobs) exists only while the
        translate route runs with the lab master on and a synthesizer chosen.
        Stopping never blocks the GUI: the joins ride a background job.
        """
        driver = getattr(self, 'translate_driver', None)
        if driver is None or not hasattr(self, 'translate_on'):
            return
        lab = self._capture_lab()
        armed = (self.controller.engine.running
                 and self.mode.currentData() == 'translate'
                 and self.rate.currentData() == 48000
                 and self.lab_enabled.isChecked()
                 and lab.translation_provider != 'off')
        if armed and not driver.alive:
            self._translate_join_pending = False
            self._start_translation_driver(lab)
        elif not armed and driver.alive and not self._translate_join_pending:
            self._translate_join_pending = True
            driver.begin_stop()
            self.files.submit('translate_join', driver.join)
        self._render_translation_status(driver.status, armed, lab)

    def _start_translation_driver(self, lab):
        """Resolve the ASR/MT/TTS pieces and hand them to the driver thread."""
        from src.runtime_paths import model_root
        from src.translate.providers import build_pipeline
        from src.tts import provider as tts_provider
        driver = self.translate_driver
        root = model_root()
        import importlib.util
        from src.tts.client import worker_available
        if not tts_provider.available(root) or (
                importlib.util.find_spec('kokoro_onnx') is None
                and not worker_available(root)):
            self.translate_live_status.setText(
                'TTSが未導入のため同時通訳は動けません: tools/install_tts.py を実行してください。')
            return
        reference = None
        if lab.translate_preserve:
            folder = self._voice_folder(lab.translate_reference)
            if folder is not None and (folder / 'reference.wav').is_file():
                try:
                    _rate, reference = lab_actions.load_wav(folder / 'reference.wav', 48000)
                except ValueError:
                    reference = None
        voice = lab.translate_voice or 'af_heart'

        def build():
            pipeline, _status = build_pipeline(root, lab.translation_provider, voice,
                                               reference, 48000, None, 'ja', 'en')
            return pipeline

        try:
            asr_params = replace(self.settings.prosody_parameters(), voice_commands=True)
        except ValueError:
            asr_params = None
        if asr_params is None:
            self.translate_live_status.setText('音声認識の設定が壊れています: 抑揚補正を見直してください。')
            return
        driver.start(asr_params, build, voice=voice)
        self._render_translation_status(driver.status, True, lab)

    def _render_translation_status(self, status, armed, lab=None):
        """One honest line about the live pipeline, or what it still needs."""
        if not armed:
            if self.mode.currentData() == 'translate' and not self.controller.engine.running:
                self.translate_live_status.setText('Startを押すと同時通訳が始まります（文単位・遅延大）。')
            elif self.mode.currentData() == 'translate' and not self.lab_enabled.isChecked():
                self.translate_live_status.setText('「実験機能有効」をONにすると同時通訳が動きます。')
            elif self.mode.currentData() == 'translate':
                self.translate_live_status.setText('②で合成（Kokoro）を選ぶと同時通訳が動きます。')
            return
        heard, said = status.get('heard', ''), status.get('said', '')
        if status.get('error'):
            self.translate_live_status.setText(f'同時通訳エラー: {status["error"]}')
        elif said:
            self.translate_live_status.setText(
                f'「{heard}」→「{said}」· 待機 {status.get("queued_seconds", 0)}秒')
        elif heard:
            self.translate_live_status.setText(f'「{heard}」を翻訳・合成しています…')
        elif status.get('mic_silent'):
            self.translate_live_status.setText(
                'マイクの音が来ていません。入力デバイスと音量を確認してください。')
        else:
            self.translate_live_status.setText(
                f'聞き取っています（ASR: {status.get("asr", "?")}）…文を言い切ってください。')

    def _lab_governor_tick(self):
        """Read the callback counters and report. Snapshot only; no callback work."""
        if (not hasattr(self, 'governor_on') or not self.governor_on.isChecked()
                or not self.lab_enabled.isChecked()):
            return
        engine = self.controller.engine
        chain_stats = engine.chain.performance.snapshot()
        engine_stats = engine.performance.snapshot()
        decision = self.governor.observe(chain_stats if chain_stats.count else engine_stats)
        state = self.controller.snapshot.state
        self.governor_status.setText(
            f'{decision.label} を推奨 · 締切超過 {decision.deadline_ratio*100:.2f}% · '
            f'p99 {decision.p99_ms:.1f}ms / 予算 {self.governor.target_ms:.0f}ms · {decision.reason}'
            f'（状態: {state}）')
        if decision.changed and self.governor_auto.isChecked() and not engine.running:
            self._lab_apply_governor()

    def _build_prosody_ui(self,layout):
        p=self.settings.prosody_parameters()
        self.prosody_on=QCheckBox('Auto Intonation · AI Voiceのみ')
        self.prosody_on.setChecked(p.enabled)
        layout.addWidget(self.prosody_on)
        self.voice_commands=QCheckBox('音声コマンド（「コエイロ止めて」など・β）')
        self.voice_commands.setChecked(p.voice_commands)
        self.voice_commands.setToolTip('ウェイクワード「コエイロ」に続けて指示すると操作できます。\n'
                                       'ローカル日本語ASRを使うため、標準ボイス以外で有効です。')
        layout.addWidget(self.voice_commands)
        self.voice_command_status=QLabel('')
        self.voice_command_status.setObjectName('muted')
        self.voice_command_status.setWordWrap(True)
        layout.addWidget(self.voice_command_status)
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
        for widget in (self.prosody_on,self.ending_lift,self.voice_commands):
            widget.toggled.connect(self._prosody_changed)
        for widget in (self.anime_amount,self.prosody_range,self.prosody_energy):
            widget.valueChanged.connect(self._prosody_changed)
        self._prosody_changed()

    def _prosody_parameters(self):
        values={name:widget.value() for name,widget in self.prosody_advanced.items()}
        return ProsodyParameters(enabled=self.prosody_on.isChecked() and not is_meanvc2(self.ai_model.currentData()),preset=self.prosody_preset.currentText(),
            amount=self.anime_amount.value(),pitch_range=self.prosody_range.value(),energy=self.prosody_energy.value(),
            ending_emphasis=self.ending_lift.isChecked(),update_ms=self.prosody_update.currentData(),
            engine=self.prosody_engine.currentData(),voice_commands=self.voice_commands.isChecked(),
            text_influence=self.text_influence.value(),
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

    def _build_voice_data(self):
        """Hidden data owners for the voice and the route.

        The combos keep every existing signal path (worker parameters, settings,
        tests) while the visible face of each decision lives in step ② and the
        voice panel, so there is never a second state to disagree.
        """
        from src.vc.models import VOICE_PROFILES, DEFAULT_VOICE_ID
        self.ai_model = QComboBox()
        ordered = ([DEFAULT_VOICE_ID] if DEFAULT_VOICE_ID in VOICE_PROFILES else [])
        ordered += [key for key in VOICE_PROFILES if key != DEFAULT_VOICE_ID]
        for key in ordered:
            self.ai_model.addItem(VOICE_PROFILES[key]['name'], key)
        self.ai_model.setCurrentIndex(self.ai_model.findData(self.settings.ai_model))
        # The home page mirrors this list, so the voice can be changed where the flow
        # starts; both combos always show the same selection.
        self._sync_voice_combos()
        self.voice_quick.currentIndexChanged.connect(self._voice_quick_changed)
        self.ai_delivery = QComboBox()
        from src.runtime_paths import bundle_deliveries
        from src.vc.models import DEFAULT_DELIVERY, DELIVERY_MODES
        allowed_deliveries = bundle_deliveries()
        for key, spec in DELIVERY_MODES.items():
            if allowed_deliveries is not None and key not in allowed_deliveries:
                continue
            self.ai_delivery.addItem(spec['label'], key)
        # The saved value is the worker's delivery half, so match it back to the combo key.
        # The two utterance routes share those halves, so the experiment
        # name disambiguates them; a stale experiment falls back to the shipped route.
        saved = self.settings.ai_delivery
        wanted_experiment = getattr(self.settings, 'ai_experiment', 'none')
        delivery = next((key for key, spec in DELIVERY_MODES.items()
                         if spec['delivery'] == saved and spec.get('experiment', 'none') == wanted_experiment), None)
        if delivery is None:
            delivery = next((key for key, spec in DELIVERY_MODES.items()
                             if spec['delivery'] == saved), None)
        if delivery is None:
            # A saved mode this build does not ship must not leave the combo empty.
            delivery = (DEFAULT_DELIVERY if allowed_deliveries is None
                        or DEFAULT_DELIVERY in allowed_deliveries else 'streaming')
        self.ai_delivery.setCurrentIndex(self.ai_delivery.findData(delivery))
        self.ai_model.currentIndexChanged.connect(self._ai_model_changed)
        self.ai_delivery.currentIndexChanged.connect(self._ai_structure_changed)
        self.ai_delivery.currentIndexChanged.connect(self._sync_route_buttons)

    def _build_route_control(self, column):
        """Step ②, second half: how to wait. One decision, three buttons.

        逐次 (no waiting, lighter restoration), 一括 (wait, then restore),
        一括-高品質 (wait longer, upstream-style recipe) and 一括-最速 (shorter
        wait, unverified quality). Each button carries its consequence as a
        tooltip; the delivery note below states the cost in one line.
        """
        from src.vc.models import DELIVERY_MODES
        from src.runtime_paths import bundle_deliveries
        allowed = bundle_deliveries()
        label = QLabel('話し方（待つか待たないか）')
        label.setObjectName('muted')
        column.addWidget(label)
        self.route_box = QWidget()
        self.route_box.setObjectName('segmentBox')
        route_layout = QHBoxLayout(self.route_box)
        route_layout.setContentsMargins(3, 3, 3, 3)
        route_layout.setSpacing(2)
        self.route_group = QButtonGroup(self)
        self.route_group.setExclusive(True)
        self.route_buttons = {}
        for key, spec in DELIVERY_MODES.items():
            if allowed is not None and key not in allowed:
                continue
            button = QPushButton(spec['label'])
            button.setObjectName('routeButton')
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(spec['detail'])
            button.setMinimumHeight(40)
            button.clicked.connect(lambda _checked=False, name=key: self._select_route(name))
            self.route_group.addButton(button)
            route_layout.addWidget(button)
            self.route_buttons[key] = button
        column.addWidget(self.route_box)
        self.delivery_note = QLabel('')
        self.delivery_note.setWordWrap(True)
        self.delivery_note.setMinimumHeight(1)
        column.addWidget(self.delivery_note)
        self._update_delivery_note()
        self._sync_route_buttons()
        self.finish_phrase = QPushButton('今の発話を変換')
        self.finish_phrase.setToolTip('一括変換で、話し終わる前に区切って変換します。')
        self.finish_phrase.clicked.connect(self._finish_current_utterance)
        column.addWidget(self.finish_phrase)

    def _section(self, layout, title):
        """A labelled settings section: one decision per card."""
        card = QFrame()
        card.setObjectName('stepCard')
        column = QVBoxLayout(card)
        column.setContentsMargins(18, 14, 18, 16)
        column.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName('stepTitle')
        column.addWidget(heading)
        layout.addWidget(card)
        return column

    def _band(self, layout, glyph, number, name, purpose):
        """A named group of cards in a drawer (DESIGN.md section 4).

        The three bands are asked in the order a session happens: where the sound
        enters and leaves, what it becomes, and what the app remembers. Grouping is
        what keeps one decision in one place: a value that two cards used to share
        now lives in the band that owns it, and the cards go into ``band.body`` so
        what the heading names is what sits under it.
        """
        from .shell import Band
        band = Band(number, name, purpose, glyph=glyph)
        layout.addWidget(band)
        return band

    def _build_finish_section(self, layout):
        """Condition the converted output: de-click joins, tame harsh fricatives.

        A standard card, off by default like the denoise stage: it reshapes the
        voice, so the choice belongs to the reader. The stage itself only ever
        touches converted routes (the chain arms it per block), and the switch
        is live because the gain eases back to unity when turned off mid-stream.
        """
        column = self._section(layout, '出力の仕上げ（自然さ）')
        note = QLabel('変換後のつなぎ目のノイズと耳に刺さる息の音を抑えます（最大3dB・穏やか）。\n'
                      'Originalには掛かりません。効き目の聴感確認はお使いの耳でお願いします。')
        note.setObjectName('muted')
        note.setWordWrap(True)
        column.addWidget(note)
        self.finish_toggle = QCheckBox('仕上げを使う（変換後のみ）')
        self.finish_toggle.setChecked(self.settings.finish)
        self.finish_toggle.setToolTip('つなぎ目のクリック除去と歯擦音の抑制。\n'
                                      '既定はOFF。停止中でも変換中でも切替できます。')
        self.finish_toggle.toggled.connect(self._finish_changed)
        column.addWidget(self.finish_toggle)
        self._finish_changed()

    def _finish_changed(self, *_):
        """Live: the stage eases its gain back, so the stream never needs a restart."""
        if not hasattr(self, 'finish_toggle'):
            return
        if not self._closing:
            self.controller.set_finish(self.finish_toggle.isChecked())

    def _build_ai_formant_section(self, layout):
        """Brighten the AI output toward a feminine vocal tract, pitch untouched.

        A standard card, off by default like the denoise stage: it reshapes the
        voice, so the choice belongs to the reader. The lift applies to the AI
        route only (Original, Female DSP and translate pass by), adds the native
        stage's lookahead to the route while on, and switches live because the
        native engine morphs coefficients without reallocating.
        """
        column = self._section(layout, 'AI出力のフォルマント')
        note = QLabel('AIの声の高さは変えず、声の明るさだけ上げます（既定+2.0）。\n'
                      'AI Voiceにだけ掛かります。効き目の聴感確認はお使いの耳でお願いします。')
        note.setObjectName('muted')
        note.setWordWrap(True)
        column.addWidget(note)
        self.ai_formant_toggle = QCheckBox('フォルマントを持ち上げる（AIのみ）')
        self.ai_formant_toggle.setChecked(self.settings.ai_formant_on)
        self.ai_formant_toggle.setToolTip('共振を高域へ寄せ、女声らしく明るくします。\n'
                                          '既定はOFF。停止中でも変換中でも切替できます。')
        self.ai_formant_toggle.toggled.connect(self._ai_formant_changed)
        column.addWidget(self.ai_formant_toggle)
        self.ai_formant_amount, self.ai_formant_amount_label = self._slider(
            column, '持ち上げ量', 0, 60, round(self.settings.ai_formant * 10))
        self.ai_formant_amount.valueChanged.connect(self._ai_formant_changed)
        self.ai_formant_amount_label.setText(f'+{self.ai_formant_amount.value()/10:.1f} st')
        self._ai_formant_changed()

    def _ai_formant_changed(self, *_):
        """Live: the native stage morphs coefficients, so no restart is needed."""
        if not hasattr(self, 'ai_formant_toggle'):
            return
        self.ai_formant_amount_label.setText(f'+{self.ai_formant_amount.value()/10:.1f} st')
        if not self._closing:
            self.controller.set_ai_formant(self.ai_formant_toggle.isChecked(),
                                           self.ai_formant_amount.value() / 10)

    def _build_prosody_section(self, layout):
        column = self._section(layout, '抑揚補正（Prosody）')
        note = QLabel('標準ボイスのピッチと抑揚は同梱のReferenceに固定されています。'
                      '他モードでのみ使えます。')
        note.setObjectName('muted')
        note.setWordWrap(True)
        column.addWidget(note)
        self.prosody_tab = QWidget()
        self._build_prosody_ui(QVBoxLayout(self.prosody_tab))
        column.addWidget(self.prosody_tab)
        # Shown instead of the prosody controls for the standard voice, whose pitch and
        # intonation are fixed by the shipped reference: hidden, not dead.
        self.prosody_fixed_note = QLabel(
            'この声では抑揚補正（Prosody）は使いません。標準ボイスのピッチと抑揚は'
            '同梱のReferenceに固定されているためです。')
        self.prosody_fixed_note.setWordWrap(True)
        self.prosody_fixed_note.setObjectName('muted')
        self.prosody_fixed_note.setVisible(False)
        column.addWidget(self.prosody_fixed_note)
        # The prosody parameters were shown inside the diagnostics card, so the card
        # titled after the feature did not contain the feature's own controls.
        self._build_prosody_advanced(column)

    def _build_update_section(self, layout):
        """Settings top: version, check button, auto-check.

        Releases ship on GitHub; this section is how an installed copy learns about
        them. The check runs on a worker thread and only updates labels here.
        """
        from src.version import APP_VERSION
        card = QFrame()
        card.setObjectName('stepCard')
        column = QVBoxLayout(card)
        column.setContentsMargins(18, 14, 18, 16)
        column.setSpacing(10)
        heading = QLabel('アプリの更新')
        heading.setObjectName('stepTitle')
        column.addWidget(heading)
        self.update_version = QLabel(f'Koeiro v{APP_VERSION} ・ GitHubリリース配布')
        self.update_version.setObjectName('muted')
        column.addWidget(self.update_version)
        self.update_status = QLabel('まだ確認していません')
        self.update_status.setWordWrap(True)
        self.update_status.setObjectName('muted')
        column.addWidget(self.update_status)
        row = QHBoxLayout()
        self.update_check_btn = QPushButton('更新を確認')
        self.update_check_btn.setToolTip('GitHubリリースに新しいバージョンがないか確認します。')
        self.update_check_btn.clicked.connect(lambda: self._check_updates(manual=True))
        row.addWidget(self.update_check_btn)
        self.update_auto = QCheckBox('起動時に自動で確認する')
        self.update_auto.setChecked(self.settings.app_update_check)
        self.update_auto.toggled.connect(lambda: (self._capture_settings(), self._save_settings()))
        row.addWidget(self.update_auto)
        row.addStretch(1)
        column.addLayout(row)
        # Appended, not inserted: this card used to be pinned near the top of the
        # settings surface, and after the drawer reorganisation it landed straight
        # under the ① 配線 heading while its own band sat two bands lower.
        layout.addWidget(card)
        self._update_pending = None
        self._update_manual = True

    def _build_discord_section(self, layout):
        """Settings: what Koeiro shows on the reader's Discord profile.

        The two lines Discord renders below the app name are previewed here, so what
        is being published can be read without opening Discord.
        """
        column = self._section(layout, 'Discordのアクティビティ表示')
        note = QLabel('Discordを使っている間、プロフィールに「変換中かどうか」と'
                      '「選んでいる声」を表示します。標準ボイスか、後から追加した声かも'
                      '分かるように出します。\n'
                      'Discordデスクトップアプリが起動しているときだけつながります'
                      '（ログインも通信も不要で、このPCの中だけで完結します）。')
        note.setObjectName('muted')
        note.setWordWrap(True)
        column.addWidget(note)
        self.discord_toggle = QCheckBox('Discordに何をしているかを表示する')
        self.discord_toggle.setToolTip('オフにすると即座に表示を消します。設定は保存されます。')
        self.discord_toggle.setChecked(self.settings.discord_presence)
        self.discord_toggle.toggled.connect(self._discord_presence_toggled)
        column.addWidget(self.discord_toggle)
        self.discord_status = QLabel('')
        self.discord_status.setObjectName('muted')
        self.discord_status.setWordWrap(True)
        column.addWidget(self.discord_status)
        self.discord_preview = QLabel('')
        self.discord_preview.setObjectName('muted')
        self.discord_preview.setWordWrap(True)
        self.discord_preview.setTextFormat(Qt.TextFormat.PlainText)
        column.addWidget(self.discord_preview)
        self._show_presence_status()

    def _presence_state(self):
        """The session facts Discord is told, read from the widgets the UI already has."""
        from src.presence import VOICE_STANDARD, VOICE_USER, PresenceState
        snapshot = self.controller.snapshot
        running = snapshot.state == 'Running' and not snapshot.error
        # The elapsed timer must describe a run that happened, so it starts when the
        # engine first reports Running and is dropped the moment it stops.
        if running and self._run_started_at is None:
            self._run_started_at = int(time.time())
        elif not running:
            self._run_started_at = None
        mode = self.mode.currentData() if hasattr(self, 'mode') else 'original'
        name, kind = '', ''
        if mode == 'ai_voice' and hasattr(self, 'ai_model'):
            name = self.ai_model.currentText()
            try:
                from src.vc.models import profile
                selected = profile(self.ai_model.currentData())
            except (ValueError, TypeError):
                selected = {}
            if selected.get('standard'):
                kind = VOICE_STANDARD
            elif selected.get('user_voice'):
                kind = VOICE_USER
        return PresenceState(running=running, mode=mode, voice=name, voice_kind=kind,
                             started_at=self._run_started_at, failed=bool(snapshot.error))

    def _ensure_presence(self):
        """Start the Discord worker on the first show, unless it is switched off.

        Skipped under pytest so a test run never talks to the tester's own Discord
        (and never spends the RPC server's 2-connections-per-minute budget).
        """
        if self.presence is not None or not self.settings.discord_presence:
            return self.presence
        if "PYTEST_CURRENT_TEST" in os.environ:
            return None
        return self._start_presence()

    def _start_presence(self):
        if self.presence is None:
            self.presence = self._presence_factory()
        self.presence.start()
        self._show_presence_status()
        return self.presence

    def _stop_presence(self):
        client, self.presence = self.presence, None
        if client is not None:
            client.stop()
        self._show_presence_status()

    def _discord_presence_toggled(self, enabled):
        self._capture_settings()
        self._save_settings()
        if enabled:
            self._start_presence()
        else:
            self._stop_presence()

    def _show_presence_status(self):
        """Status and preview, built from the same activity Discord would receive."""
        if not hasattr(self, 'discord_status'):
            return
        from src.presence import build_activity
        state = self._presence_state()
        if self.presence is None:
            self.discord_status.setText('オフ · ' + ('表示するにチェックを入れるとつながります'
                                                   if not self.settings.discord_presence
                                                   else '起動待ちです'))
            self.discord_preview.setText('')
            return
        self.discord_status.setText(f'状態: {self.presence.label()}')
        activity = build_activity(state)
        lines = [f"{activity['details']}", f"{activity['state']}"]
        if 'timestamps' in activity:
            lines.append('経過時間つき')
        self.discord_preview.setText('Discordに表示される内容（1行目はDiscord側のアプリ名）:\n'
                                     + '\n'.join(lines))

    def _check_updates(self, manual=True):
        """Ask GitHub for the newest release on a worker thread."""
        if self._closing:
            return
        from src import app_update
        self._update_manual = manual
        self.update_status.setText('確認しています…')
        self.update_check_btn.setEnabled(False)
        self.files.submit('update_check', lambda: json.dumps(app_update.check_now()))

    def _auto_update_check(self):
        if self._closing or "PYTEST_CURRENT_TEST" in os.environ:
            return
        if not self.settings.app_update_check:
            return
        self._check_updates(manual=False)

    def _drain_update_results(self):
        raw = self.files.results.pop('update_check', None)
        if raw:
            self._finish_update_check(raw)
        raw = self.files.results.pop('update_download', None)
        if raw:
            self._finish_update_download(raw)

    def _finish_update_check(self, raw):
        self.update_check_btn.setEnabled(True)
        try:
            result = json.loads(raw[7:] if raw.startswith('ERROR: ') else raw)
        except ValueError:
            result = {"status": "error", "message": raw}
        if isinstance(result, str) or result.get("status") == "error":
            message = result if isinstance(result, str) else result.get("message", "確認できませんでした")
            self.update_status.setText(f'確認できませんでした：{message}')
            return
        if result.get("status") == "uptodate":
            self._update_pending = None
            self.update_check_btn.setText('更新を確認')
            self.update_status.setText(f'最新です（{result.get("tag", "")}）')
            return
        self._update_pending = result
        self.update_check_btn.setText('ダウンロードして更新')
        self.update_status.setText(f'新しいバージョンがあります：{result.get("tag", "")}')
        if self._update_manual or result.get("tag") != self.settings.app_update_ignored:
            self._offer_update()

    def _offer_update(self):
        """Found a release: update now, ignore this version, or later.

        The note is long (a release body is thousands of characters) and grows the
        dialog it sits in, so it scrolls inside a bounded area while the three
        buttons stay in the dialog's own layout. They used to be QMessageBox text,
        and a long note pushed 「更新して再起動」 out of the screen.
        """
        pending = self._update_pending
        if not pending:
            return
        from .update_offer import UpdateOfferDialog
        dialog = UpdateOfferDialog(self, str(pending.get("tag", "")),
                                   str(pending.get("notes") or ""),
                                   str(pending.get("url") or ""))
        dialog.exec()
        if dialog.action == 'update':
            self._download_and_restart()
        elif dialog.action == 'ignore':
            self.settings = replace(self.settings, app_update_ignored=str(pending.get("tag", "")))
            self._save_settings()
            self.update_status.setText(f'{pending.get("tag", "")} は無視中です')

    def _download_and_restart(self):
        pending = self._update_pending
        if not pending or self._closing:
            return
        asset = pending.get("asset") or {}
        if not asset.get("url"):
            self.update_status.setText('配布ファイルが見つかりませんでした')
            return
        kind = asset.get("kind")
        self.update_status.setText('インストーラをダウンロードしています…' if kind == "setup"
                                   else '更新データをダウンロードしています…' if kind == "bundle"
                                   else 'ダウンロードしています…')
        self.update_check_btn.setEnabled(False)
        tag = str(pending.get("tag", ""))
        self.files.submit('update_download',
                          lambda: json.dumps(self._fetch_update_package(asset, tag)))

    @staticmethod
    def _fetch_update_package(asset, tag):
        """Worker thread: download, verify, stage. Returns plain data only."""
        from src import app_update
        try:
            if asset.get("kind") == "bundle":
                return {"staging": str(app_update.prepare_bundle(asset, tag))}
            if asset.get("kind") == "setup":
                installer = app_update.prepare_installer(asset, tag)
                return {"installer": str(installer)}
            target = app_update.updates_dir() / str(asset.get("name") or f"koeiro-{tag}.pkg")
            app_update.download_asset(asset["url"], target)
            staging = app_update.updates_dir() / f"staging-{tag}"
            if staging.exists():
                import shutil
                shutil.rmtree(staging, ignore_errors=True)
            package = app_update.extract_package(target, staging)
            info = app_update.read_package_release(package)
            if not app_update._version_matches(info.get("version"), tag):
                return {"error": "更新パッケージが壊れています"}
            try:
                target.unlink()
            except OSError:
                pass
            return {"staging": str(package)}
        except Exception as error:
            return {"error": str(error)}

    def _finish_update_download(self, raw):
        self.update_check_btn.setEnabled(True)
        try:
            result = json.loads(raw[7:] if raw.startswith('ERROR: ') else raw)
        except ValueError:
            result = {"error": raw}
        if isinstance(result, dict) and result.get("installer"):
            from src import app_update
            try:
                app_update.run_installer_and_restart(result["installer"])
            except Exception as error:
                self.update_status.setText(f'更新を開始できませんでした：{error}')
                return
            self.update_status.setText('インストール後に自動的に再起動します…')
            self.close()
            return
        if isinstance(result, dict) and result.get("staging"):
            from src import app_update
            try:
                app_update.begin_update_and_restart(result["staging"],
                                                    str((self._update_pending or {}).get("tag", "")))
            except Exception as error:
                self.update_status.setText(f'更新を開始できませんでした：{error}')
                return
            self.update_status.setText('更新を適用するため再起動します…')
            self.close()
            return
        message = result.get("error", "ダウンロードできませんでした") if isinstance(result, dict) else raw
        self.update_status.setText(f'ダウンロードできませんでした：{message}')

    def _build_dsp_section(self, layout):
        column = self._section(layout, '声の加工（Female DSP）')
        note = QLabel('変換モードで Female DSP を選んだときだけ使えます。'
                      'AI Voice では使いません。')
        note.setObjectName('muted')
        note.setWordWrap(True)
        column.addWidget(note)
        self._build_dsp_ui(column)

    def _build_ai_details(self, layout):
        column = self._section(layout, '声の微調整（一括変換）')
        self.ai_details = QWidget()
        details = QVBoxLayout(self.ai_details)
        details.setContentsMargins(0, 0, 0, 0)
        self.ai_quality = QComboBox()
        for label, key in [('Low Latency · chunk 13 ms', 'low_latency'), ('Balanced · chunk 26 ms', 'balanced'), ('Stable · chunk 52 ms', 'stable')]:
            self.ai_quality.addItem(label, key)
        self.ai_quality.setToolTip('数値はAIの処理単位です。出力待ちやモデル遅延、Discordまでの実測遅延とは異なります。')
        self.ai_quality.setCurrentIndex(self.ai_quality.findData(self.settings.ai_quality))
        self.ai_threads = QComboBox()
        for count in range(1, 5):
            self.ai_threads.addItem(str(count), count)
        self.ai_threads.setCurrentIndex(self.ai_threads.findData(self.settings.ai_threads))
        self.ai_threads.setToolTip('AIの並列数。2が速いPCが多く、4が速い場合もあります。')
        self.ai_device = QComboBox()
        for label, key in [('自動（CPUで測定）', 'auto'), ('CPU', 'cpu'), ('GPU（CUDA・実測済み）', 'cuda')]:
            self.ai_device.addItem(label, key)
        self.ai_device.setCurrentIndex(self.ai_device.findData(self.settings.ai_device))
        self.ai_device.setToolTip('推論の実行場所。同梱の声はCPUで測定しました。「自動」もCPUを選びます。GPU（CUDA）は Kaggle の Tesla T4 で計測済み（CPUと2.0倍速、差は-76dB）で、選んだときだけ試します。聴感の確認はまだです。')
        for name, combo in [('処理構成', self.ai_quality), ('CPU threads', self.ai_threads), ('実行デバイス', self.ai_device)]:
            details.addWidget(self._labelled_row(name, combo))
        self.ai_brightness, self.ai_brightness_label = self._slider(details, 'AI Brightness', 0, 100, round(self.settings.ai_brightness))
        self.ai_low_cut = QCheckBox('Low Cut · 90 Hz')
        self.ai_limiter = QCheckBox('Limiter')
        self.ai_post_fx = QCheckBox('Light Post FX（Pitch/Formantなし）')
        for widget, value in [(self.ai_low_cut, self.settings.ai_low_cut), (self.ai_limiter, self.settings.ai_limiter), (self.ai_post_fx, self.settings.ai_post_fx)]:
            widget.setChecked(value)
            details.addWidget(widget)
        # The route's own fixed values. They were in the diagnostics card, one band away
        # from the voice they belong to, next to a column of read-outs.
        route_caption = QLabel('経路の固定値（この声で固定される項目は隠れます）')
        route_caption.setObjectName('muted')
        route_caption.setWordWrap(True)
        details.addWidget(route_caption)
        self.ai_pitch, self.ai_pitch_label = self._slider(details, 'Beatrice native Pitch', -96, 96, round(self.settings.ai_pitch*8))
        self.ai_pitch_label.setText(f'{self.ai_pitch.value()/8:+.3f} st')
        self.ai_pitch.valueChanged.connect(self._ai_fx_changed)
        self.ai_wait, self.ai_wait_label = self._slider(details, 'AI pre-roll wait · Experimental', 20, 156, round(self.settings.ai_output_wait_ms))
        self.ai_fade, self.ai_fade_label = self._slider(details, 'Mode crossfade', 20, 100, round(self.settings.ai_crossfade_ms))
        self.ai_wait_label.setText(self._preroll_text())
        self.ai_fade_label.setText(f'{self.ai_fade.value()} ms')
        self.ai_wait.valueChanged.connect(self._ai_structure_changed)
        self.ai_fade.valueChanged.connect(self._ai_structure_changed)
        self.tune_header = QLabel('声の調整（一括変換のみ）')
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
        details.addWidget(self.ai_status)
        details.addWidget(self.ai_performance)
        column.addWidget(self.ai_details)
        note = QLabel('声の追加・切り替えは停止中に行います。\n新しい声の似具合は録音内容によって変わります。')
        note.setWordWrap(True)
        column.addWidget(note)
        for combo in (self.ai_quality, self.ai_threads, self.ai_device):
            combo.currentIndexChanged.connect(self._ai_structure_changed)
        self.ai_brightness.valueChanged.connect(self._ai_fx_changed)
        for checkbox in (self.ai_low_cut, self.ai_limiter, self.ai_post_fx):
            checkbox.toggled.connect(self._ai_fx_changed)
        for slider in (self.tune_sib, self.tune_cons, self.tune_caps, self.tune_floor, self.tune_excess,
                       self.tune_mid, self.tune_match, self.tune_ptrans, self.tune_pcap, self.tune_level):
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
            for widget in (self.ai_model, self.ai_quality, self.ai_threads, self.ai_device, self.ai_brightness, self.ai_low_cut, self.ai_limiter, self.ai_post_fx):
                widget.setEnabled(False)

    def _build_diagnostics_section(self, layout):
        """診断: what the app measured, and no live setting.

        The route parameters and the prosody detail block used to live here, which put
        two live decisions inside a column of read-outs. They sit with the controls they
        belong to now, and this card answers only "what is happening right now".
        """
        column = self._section(layout, '診断（実測値）')
        note = QLabel('ここに出るのは計測値だけです。End-to-Endの遅延と聴感は含みません。')
        note.setObjectName('muted')
        note.setWordWrap(True)
        column.addWidget(note)
        self.latency_detail = QLabel()
        self.latency_detail.setWordWrap(True)
        self.latency_detail.setObjectName("muted")
        self.latency_detail.setTextFormat(Qt.TextFormat.PlainText)
        column.addWidget(self.latency_detail)
        self.xruns = QLabel("Underflow: 0 · Overflow: 0")
        self.xruns.setObjectName("muted")
        column.addWidget(self.xruns)
        self.health = QLabel('Realtime Health: Waiting')
        self.health.setObjectName('muted')
        self.health.setWordWrap(True)
        column.addWidget(self.health)
        self.performance = QLabel()
        self.performance.setWordWrap(True)
        self.performance.setObjectName("muted")
        column.addWidget(self.performance)
        self.file_status = QLabel('')
        self.file_status.setWordWrap(True)
        column.addWidget(self.file_status)
        self.debug_snapshot = QPushButton('Debug Snapshotを保存')
        self.debug_snapshot.clicked.connect(self._debug_snapshot)
        column.addWidget(self.debug_snapshot)

    def _build_voice_panel(self):
        """Right column: the voice list, and nothing else.

        The output device and the level used to live at the foot of this panel, which
        put the two ends of the audio path on different surfaces and left two level
        sliders showing one value. They belong to the ① 配線 band now.
        """
        from src.vc.models import DEFAULT_VOICE_ID
        panel = self.voice_panel
        self.voice_hint = QLabel('声を選んで Start。マイクと出力先は「調整」の①配線にあります。')
        self.voice_hint.setWordWrap(True)
        self.voice_hint.setTextFormat(Qt.TextFormat.PlainText)
        panel.actions_layout.addWidget(self.voice_hint)
        self.add_voice = QPushButton('＋ 声を追加')
        self.add_voice.clicked.connect(self._add_voice)
        panel.actions_layout.addWidget(self.add_voice)
        self.voice_note = QLabel('登録はローカルで完結。学習不要・元の音声ファイルはそのまま。')
        self.voice_note.setObjectName('muted')
        self.voice_note.setWordWrap(True)
        self.voice_note.setTextFormat(Qt.TextFormat.PlainText)
        panel.actions_layout.addWidget(self.voice_note)
        self.use_voice = QPushButton('選択した声を使う（AI Voice）')
        self.use_voice.clicked.connect(self._use_selected_voice)
        panel.actions_layout.addWidget(self.use_voice)
        panel.search.textChanged.connect(lambda _text: self._refresh_voice_cards())
        panel.tab_group.buttonClicked.connect(lambda _button: self._refresh_voice_cards())
        self.voice_cards = {}
        self._refresh_voice_cards()
        self._update_voice_info()

    def _focus_voice_panel(self):
        """Step ② helper: put the cursor in the voice search."""
        self.voice_panel.search.setFocus()

    def _refresh_voice_cards(self):
        """Rebuild the voice list from the profiles; search and tabs only filter."""
        from .shell import VoiceCard
        from src.vc.models import VOICE_PROFILES, DEFAULT_VOICE_ID
        box = self.voice_panel.cards_layout
        while box.count():
            item = box.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.voice_cards = {}
        query = self.voice_panel.search.text().strip().lower()
        tab = next((key for key, button in self.voice_panel.tabs.items() if button.isChecked()), 'all')
        ordered = ([DEFAULT_VOICE_ID] if DEFAULT_VOICE_ID in VOICE_PROFILES else [])
        ordered += [key for key in VOICE_PROFILES if key != DEFAULT_VOICE_ID]
        current = self.ai_model.currentData() if hasattr(self, 'ai_model') else None
        for key in ordered:
            profile = VOICE_PROFILES[key]
            standard = bool(profile.get('standard')) or key == DEFAULT_VOICE_ID
            user = bool(profile.get('user_voice')) and not standard
            if tab == 'standard' and not standard:
                continue
            if tab == 'user' and not user:
                continue
            name = profile.get('name', key)
            if query and query not in name.lower():
                continue
            sub = '標準ボイス（同梱）' if standard else ('追加した声' if user else '声')
            card = VoiceCard(key, name, sub)
            card.selected.connect(self._select_voice_key)
            card.set_selected(key == current)
            box.addWidget(card)
            self.voice_cards[key] = card
        self._sync_voice_highlight()

    def _sync_voice_highlight(self):
        current = self.ai_model.currentData() if hasattr(self, 'ai_model') else None
        for key, card in getattr(self, 'voice_cards', {}).items():
            card.set_selected(key == current)

    def _select_voice_key(self, key):
        """A panel card drives the library combo rather than duplicating state."""
        if self._pending or self._closing or self.controller.engine.running or self.voice_dialog:
            return
        index = self.ai_model.findData(key)
        if index >= 0 and index != self.ai_model.currentIndex():
            self.ai_model.setCurrentIndex(index)

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
        route = self.mode.currentData()
        return DSPParameters(mode='female_dsp' if route == 'ai_voice' else (route if route in MODES else 'original'), pitch=self.pitch.value()/10,
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
    def _ai_parameters(self):
        # The delivery combo owns both halves of the route, so the worker never receives
        # a delivery and an enhancer that disagree.
        delivery=DELIVERY_MODES.get(self.ai_delivery.currentData(),DELIVERY_MODES[DEFAULT_DELIVERY])
        from src.vc.models import default_voice_id, VOICE_PROFILES
        model = self.ai_model.currentData()
        if model not in VOICE_PROFILES:
            model = default_voice_id() or next(iter(VOICE_PROFILES), None)
        if model is None:
            raise ValueError('利用可能なAI音声がありません。右の声パネルから声を登録してください。')
        return AIParameters(model=model,quality=self.ai_quality.currentData(), threads=self.ai_threads.currentData(),
                            device=self.ai_device.currentData(),
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
            self.ai_wait_label.setText(self._preroll_text())
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
        self._sync_voice_highlight()
        if not self._pending and not self.controller.engine.running:
            self._apply_recommended()
            self._update_meanvc_controls()
            self._update_voice_info()
            self.startup_diagnostic = None
            model = self.ai_model.currentData()
            self._diagnostic_model = model
            self.files.submit('diagnostics', lambda: self._diagnose_voice(model))

    def _diagnose_voice(self, model):
        # Voice folders live in the recorded install tree, not beside __file__:
        # a frozen bundle holds only the default voice, so resolving the root
        # from this file reports every other voice (and the MeanVC2 assets)
        # missing while the worker loads them fine. model_root() is the same
        # tree the worker and the TTS lookup use.
        from src.runtime_paths import model_root
        result = startup_diagnostics(model_root(), model)
        if self._diagnostic_model == model:
            self.startup_diagnostic = result

    def _update_voice_info(self):
        try:
            from src.vc.models import profile
            selected = profile(self.ai_model.currentData())
        except ValueError:
            self.voice_hint.setText('声がありません。右の声パネルから声を登録してください。')
            return
        from src.vc.voice_library import metadata
        # Both voice branches quote numbers out of the user's own profile.json, and a
        # half-copied or hand-edited one used to raise through this slot -- Qt swallows
        # that, so the panel silently kept the previous voice's text. Reading it here
        # turns that into one honest line; the conversion itself never depended on it.
        if selected.get('standard') or selected.get('user_voice'):
            try:
                info = metadata(self.ai_model.currentData())
                seconds = float(info['reference_seconds'])
                name = info['name'] if isinstance(info.get('name'), str) else selected['name']
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                self.voice_hint.setText('この声の詳細は profile.json から読み込めません。変換は使えます。')
                return
        if selected.get('standard'):
            self.voice_hint.setText(
                f"アプリ標準の声「{selected['name']}」を選択中\n"
                f"Reference {seconds:.1f}秒 · 最初からこの声で使えます。"
                "自分の声に変えるには「＋ 音声ファイルから声を追加」から。")
            self.voice_note.setText('この声はアプリに同梱されています。変更・削除はできません。')
        elif selected.get('user_voice'):
            self.voice_hint.setText(
                f"登録した声「{name}」を選択中\n"
                f"Reference {seconds:.1f}秒 · 似具合は録音内容で変わります。")
            self.voice_note.setText('登録はローカルで完結。学習不要・元の音声ファイルはそのまま。')
        else:
            self.voice_hint.setText(selected['name']+'\n今までの声はそのまま選べます。')

    def _add_voice(self):
        if self._pending or self._closing or self.controller.engine.running or self.voice_dialog:
            return
        from .voice_registration import VoiceRegistrationDialog
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
            self._refresh_voice_cards()
            self._ai_model_changed()
            self._capture_settings()
            self._save_settings()
        if not self._pending:
            self._set_controls(not self.controller.engine.running)

    def _preroll_text(self):
        """The reader's own pre-roll wait, labelled as a setting and not a measurement.

        One source for the caption: it used to be spelled out at every place that
        touched the slider, and the three copies could disagree with each other.
        """
        return f'{self.ai_wait.value()} ms · pre-roll（実測遅延ではありません）'

    def _meanvc_preroll_text(self):
        return f'{self._ai_parameters().startup_frames/48:g} ms · MeanVC2固定pre-roll（実測遅延ではありません）'

    def _update_meanvc_controls(self):
        mean=is_meanvc2(self.ai_model.currentData())
        self.ai_delivery.setEnabled(mean and not self.controller.engine.running)
        self._sync_route_buttons()
        self._update_delivery_note()
        self.finish_phrase.setVisible(mean)
        self.ai_threads.setVisible(mean)
        self.ai_device.setVisible(mean)
        # The pre-roll text follows the model: a MeanVC2 note left behind after moving
        # to another voice would state a pre-roll that is not being used.
        self.ai_wait_label.setText(self._meanvc_preroll_text() if mean
                                   else self._preroll_text())
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
                  '遅延は小さいですが、帯域復元は行いません。')
        elif mode=='utterance_quality':
            text=('一括変換-高品質：ソルバ4ステップと、公式推論と同じ全長の条件補間・ノイズで変換します。\n'
                  '通常の一括変換より待ち時間が長くなります（帯域復元LavaSRは入ります。最大60秒）。\n'
                  '音質の採否は聴き比べで決めます。')
        elif mode=='utterance_fastest':
            text=('一括変換-最速：まとめ推論＋ソルバ1ステップ・スレッド2以上で待ち時間を減らします。\n'
                  'N150実測（24秒の発話1回）：約16秒（RTF 0.66）。他のアプリがCPUを食っていると1に近づきます。\n'
                  '速さの代わりに声色の精度は落ちます（帯域復元LavaSRは入ります）。')
        else:
            text=('一括変換：0.8秒の無音で発話を区切り、変換後にLavaSRで帯域を復元します。最大60秒。\n'
                  'N150実測（オフラインWAV）：25秒分に約8.5秒、60秒分に約21秒。'
                  '変換待ち・録音中は無音です。')
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
        # Text readout only: the LevelBar strip below already draws the level,
        # and a second full-width bar reads as decoration rather than signal.
        row = QHBoxLayout()
        row.addWidget(QLabel(name))
        text = QLabel("−80.0 dBFS")
        row.addStretch()
        row.addWidget(text)
        layout.addLayout(row)
        return None, text

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
        # Home ②: the quick voice picker belongs to the AI route.
        self.voice_quick.setVisible(ai_mode)
        self.voice_quick_note.setVisible(ai_mode)
        if ai_mode:
            self.voice_quick_note.setText(self._voice_summary_line())
        self.use_voice.setVisible(not ai_mode)
        # Utterance route: committing an unspoken utterance only exists on that route.
        self.finish_phrase.setVisible(mean and utterance)
        # Voice tuning belongs to the 一括変換 recipe.
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
                                    '明るさ / EQ）はAI Voiceでは使いません。声の選択は右の声パネルで行います。')
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

    def _denoise_changed(self, *_):
        """Live: 0 is a pass-through, so the stream never needs a restart."""
        if not hasattr(self, 'denoise_toggle'):
            return
        if hasattr(self, 'denoise_strength_label'):
            self.denoise_strength_label.setText(f'{self.denoise_strength.value()/100:.2f}')
        enabled = self.denoise_toggle.isChecked()
        self.denoise_strength.setEnabled(enabled)
        self.denoise_strength_label.setEnabled(enabled)
        if not self._closing:
            self.controller.set_denoise(self.denoise_strength.value()/100 if enabled else 0.0)

    def _auto_level_changed(self, *_):
        """Live: the follower eases from unity, so the stream never needs a restart."""
        if not hasattr(self, 'auto_level_toggle'):
            return
        if not self._closing:
            self.controller.set_auto_level(self.auto_level_toggle.isChecked())

    def _env_passthrough_changed(self, *_):
        """Live: the AI slot fades itself between raw and converted blocks."""
        if not hasattr(self, 'env_passthrough_toggle'):
            return
        if not self._closing:
            self.controller.set_env_passthrough(self.env_passthrough_toggle.isChecked())

    def _build_hotkeys(self):
        """In-app shortcuts for instant route switching.

        These are *application* shortcuts: they fire while any Koeiro window has focus,
        not system-wide. The app deliberately installs no OS keyboard hook, so a
        keystroke pressed while a game or Discord has focus is not captured. Ctrl+1/2/3
        pick a route outright; Ctrl+D flips between Original and the last real route,
        which is the A/B listen that a voice change actually needs.
        """
        from PySide6.QtGui import QKeySequence, QShortcut
        self._ab_last = 'ai_voice'
        bindings = (('Ctrl+1', lambda: self._select_mode('original')),
                    ('Ctrl+2', lambda: self._select_mode('female_dsp')),
                    ('Ctrl+3', lambda: self._select_mode('ai_voice')),
                    ('Ctrl+D', self._ab_toggle),
                    ('Ctrl+F', self._focus_drawer_search))
        self.hotkeys = {}
        for sequence, handler in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(handler)
            self.hotkeys[sequence] = shortcut

    def _select_mode(self, key):
        """Route switch used by the shortcuts; mirrors the mode combo the user drives."""
        if self._pending or self._closing:
            return
        index = self.mode.findData(key)
        if index < 0:
            return
        if key != 'original':
            self._ab_last = key
        if index != self.mode.currentIndex():
            self.mode.setCurrentIndex(index)

    def _ab_toggle(self):
        """Flip between Original and the last non-Original route."""
        if self._pending or self._closing:
            return
        current = self.mode.currentData()
        if current == 'original':
            self._select_mode(self._ab_last)
        else:
            self._ab_last = current
            self._select_mode('original')

    def _drain_voice_command(self):
        """Run a recognised wake-word command, if any. Cheap when the feature is off."""
        if (self.router is None or not hasattr(self, 'voice_commands')
                or not self.voice_commands.isChecked()):
            return
        command = self.router.ai.bridge.prosody.asr.take_command()
        if command is not None:
            self._run_voice_command(command)

    def _run_voice_command(self, command):
        """Act on a voice command by reusing the controls the user already drives."""
        from src.asr.commands import label
        log.info('Voice command: %s (%s)', command.kind, command.phrase)
        kind = command.kind
        if kind == 'stop':
            self._stop()
        elif kind == 'start':
            self._start()
        elif kind in ('next_voice', 'prev_voice'):
            self._cycle_voice(1 if kind == 'next_voice' else -1)
        elif kind == 'original':
            self._select_mode('original')
        elif kind == 'ai_voice':
            self._select_mode('ai_voice')
        else:
            return
        if hasattr(self, 'voice_command_status'):
            self.voice_command_status.setText(f'音声コマンド: {label(command)}')

    def _cycle_voice(self, step):
        count = self.voice_quick.count()
        if count <= 1:
            return
        self.voice_quick.setCurrentIndex((self.voice_quick.currentIndex() + step) % count)

    # -- scenes (named settings profiles) --------------------------------------

    def _scene_store(self):
        return SceneStore(self.manager.path.with_name('scenes.json'))

    def _build_scenes_section(self, layout):
        column = self._section(layout, 'シーン（設定プロファイル）')
        note = QLabel('いまの設定（マイク・声・ルート・音の加工・抑揚）を名前を付けて保存し、'
                      'あとで1クリックで戻せます。変換中は切り替えできません。')
        note.setObjectName('muted')
        note.setWordWrap(True)
        column.addWidget(note)
        self.scene_name = QLineEdit()
        self.scene_name.setPlaceholderText('例：配信用 / 通話用 / ささやき')
        self.scene_name.setMaxLength(40)
        column.addWidget(self.scene_name)
        self.scene_combo = QComboBox()
        column.addWidget(self._labelled_row('保存済みのシーン', self.scene_combo))
        row = QHBoxLayout()
        self.scene_save = QPushButton('現在の設定を保存')
        self.scene_load = QPushButton('読み込む')
        self.scene_delete = QPushButton('削除')
        self.scene_save.clicked.connect(self._save_scene)
        self.scene_load.clicked.connect(self._load_scene)
        self.scene_delete.clicked.connect(self._delete_scene)
        for button in (self.scene_save, self.scene_load, self.scene_delete):
            button.setMinimumHeight(34)
            row.addWidget(button)
        column.addLayout(row)
        self.scene_status = QLabel('')
        self.scene_status.setObjectName('muted')
        self.scene_status.setWordWrap(True)
        column.addWidget(self.scene_status)
        self._refresh_scenes()

    def _refresh_scenes(self):
        names = self._scene_store().names()
        current = self.scene_combo.currentText()
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        self.scene_combo.addItems(names)
        if current in names:
            self.scene_combo.setCurrentText(current)
        self.scene_combo.blockSignals(False)
        self.scene_status.setText(f'{len(names)} 件のシーン' if names else 'まだシーンがありません')

    def _save_scene(self):
        self._capture_settings()
        try:
            ok = self._scene_store().save(self.scene_name.text(), self.settings)
        except ValueError as error:
            self.scene_status.setText(f'保存できません：{error}')
            return
        self.scene_status.setText('保存しました' if ok else '保存に失敗しました')
        self._refresh_scenes()

    def _load_scene(self):
        name = self.scene_combo.currentText()
        if not name:
            self.scene_status.setText('読み込むシーンを選んでください')
            return
        try:
            scene = self._scene_store().load(name)
        except KeyError:
            self.scene_status.setText('そのシーンはありません')
            self._refresh_scenes()
            return
        if self._apply_scene(scene):
            self.scene_status.setText(f'「{name}」を適用しました')
        else:
            self.scene_status.setText('変換中はシーンを切り替えられません。停止してください。')

    def _delete_scene(self):
        name = self.scene_combo.currentText()
        if not name:
            return
        self.scene_status.setText('削除しました' if self._scene_store().delete(name)
                                  else 'そのシーンはありません')
        self._refresh_scenes()

    @staticmethod
    def _select_device(combo, identity):
        """Select the saved device, leaving the current choice when it is absent."""
        if not identity:
            return
        for index in range(combo.count()):
            item = combo.itemData(index)
            if item is not None and item.identity == identity:
                combo.setCurrentIndex(index)
                return

    def _apply_scene(self, settings):
        """Adopt a saved scene by driving the same widgets the user would.

        Values travel the existing signal paths, so the worker/controller update the
        way they do for a manual edit instead of through a second apply path that
        could drift from them. Only usable while stopped.
        """
        if self._pending or self._closing or self.controller.engine.running:
            return False
        self.settings = settings
        self._select_device(self.input_device, settings.input_device)
        self._select_device(self.output_device, settings.output_device)
        self._select_device(self.monitor_device, settings.monitor_device)
        self.monitor_toggle.setChecked(settings.monitor)
        self.monitor_volume.setValue(round(settings.monitor_volume_db * 10))
        self.rate.setCurrentIndex(self.rate.findData(settings.sample_rate))
        self.buffer.setCurrentIndex(self.buffer.findData(settings.buffer_size))
        self.gain.setValue(round(settings.gain_db * 10))
        self.gate.setValue(round(settings.noise_gate_db * 10))
        self.denoise_toggle.setChecked(settings.denoise)
        self.denoise_strength.setValue(round(settings.denoise_strength * 100))
        self.auto_level_toggle.setChecked(settings.auto_level)
        self.env_passthrough_toggle.setChecked(settings.env_passthrough)
        self.finish_toggle.setChecked(settings.finish)
        self.ai_formant_toggle.setChecked(settings.ai_formant_on)
        self.ai_formant_amount.setValue(round(settings.ai_formant * 10))
        voice_index = self.ai_model.findData(settings.ai_model)
        if voice_index >= 0:
            self.ai_model.setCurrentIndex(voice_index)
        self._apply_dsp_widgets(settings.dsp_parameters(), settings.preset)
        self._apply_prosody_widgets(settings.prosody_parameters())
        self.mode.setCurrentIndex(self.mode.findData(settings.voice_mode))
        self._ai_structure_changed()
        self._route_changed()
        return True

    def _apply_prosody_widgets(self, p):
        widgets = (self.prosody_on, self.voice_commands, self.prosody_engine, self.prosody_preset,
                   self.anime_amount, self.prosody_range, self.prosody_energy, self.ending_lift,
                   self.prosody_update, self.text_influence, self.asr_threads, self.asr_update,
                   self.asr_window, self.text_strategy, self.log_transcripts)
        for widget in widgets:
            widget.blockSignals(True)
        self.prosody_on.setChecked(p.enabled)
        self.voice_commands.setChecked(p.voice_commands)
        self.prosody_engine.setCurrentIndex(self.prosody_engine.findData(p.engine))
        self.prosody_preset.setCurrentText(p.preset)
        self.anime_amount.setValue(round(p.amount))
        self.prosody_range.setValue(round(p.pitch_range))
        self.prosody_energy.setValue(round(p.energy))
        self.ending_lift.setChecked(p.ending_emphasis)
        self.prosody_update.setCurrentIndex(self.prosody_update.findData(int(p.update_ms)))
        for name, widget in self.prosody_advanced.items():
            widget.setValue(getattr(p, name))
        self.text_influence.setValue(p.text_influence)
        self.asr_threads.setValue(p.asr_threads)
        self.asr_update.setCurrentIndex(self.asr_update.findData(p.asr_update_ms))
        self.asr_window.setCurrentIndex(self.asr_window.findData(p.asr_window_ms))
        self.text_strategy.setCurrentIndex(self.text_strategy.findData(p.text_strategy))
        self.log_transcripts.setChecked(p.transcript_logging)
        for widget in widgets:
            widget.blockSignals(False)
        self._prosody_changed()

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
        # Stage state: with no microphone at all, the stage says so in one line instead
        # of drawing a level that can never move.
        has_input = self.input_device.count() > 0 and self.input_device.currentData() is not None
        self.empty_state.setVisible(not has_input)
        for widget in (self.input_bar, self.input_db, self.output_db, self.relay):
            widget.setVisible(has_input)

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
            hint = ('Input / OutputともWindows WASAPIにしてください（MMEでは音切れを記録しています）。')
        elif "cable input" in output_device.name.lower():
            hint = "App → CABLE Input ／ Discordのマイク → CABLE Output"
        elif is_virtual(output_device, "output"):
            hint = ("App → %s ／ 通話アプリのマイク → 対応する仮想入力に設定してください。"
                    % output_device.name)
        elif sys.platform != "win32":
            hint = ("仮想出力が未選択です。Linuxでは `sudo modprobe snd-aloop` で Loopback を"
                    "有効化し、出力先に Loopback を選ぶと通話アプリへ流せます（詳細はREADME）。"
                    "スピーカー出力時はハウリングに注意。")
        else:
            hint = "VB-CABLE未選択。スピーカーへの出力時はハウリングに注意し、ヘッドホンを使用してください。"
        self.route_hint.setText(hint)

    def _start(self):
        input_device = self.input_device.currentData()
        output_device = self.output_device.currentData()
        if self._pending or self.voice_dialog is not None or not input_device or not output_device:
            return
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
        for button in getattr(self, 'route_buttons', {}).values():
            button.setEnabled(stopped and not self._closing and mean and self.router is not None)
        voice_locked = stopped and not self._pending and not self._closing and self.router is not None
        self.voice_panel.search.setEnabled(voice_locked)
        for button in self.voice_panel.tabs.values():
            button.setEnabled(voice_locked)
        for card in getattr(self, 'voice_cards', {}).values():
            card.setEnabled(voice_locked)
        self.finish_phrase.setEnabled(not stopped and mean and self.router is not None and self._utterance_active() and not self._closing)
        self._update_control_visibility()
        self.ai_threads.setEnabled(stopped and not self._closing and mean)
        self.ai_device.setEnabled(stopped and not self._closing and mean)
        self.ai_wait.setEnabled(stopped and not self._closing and not mean)
        self.ai_quality.setEnabled(stopped and not self._closing and not mean)
        self.ai_restart.setEnabled(self.router is not None and not self._pending and not self._closing)
        self.mode.setEnabled(self.router is not None and not self._pending and not self._closing)
        self._update_monitor_ui()
        available = self.input_device.currentData() is not None and self.output_device.currentData() is not None
        self.start_button.setEnabled(stopped and available and not self._closing)
        self.stop_button.setEnabled(not stopped and not self._pending and not self._closing)
        self.add_voice.setEnabled(stopped and not self._pending and not self._closing and self.voice_dialog is None and self.router is not None)
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
            suffix = f" · {snapshot.error}" if snapshot.error else ""
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
        self.input_db.setText(f"{amplitude_to_db(self._display_input):.1f} dBFS")
        self.output_db.setText(f"出力 {amplitude_to_db(self._display_output):.1f} dBFS")
        self.input_bar.set_level(self._display_input)
        self._update_stage(snapshot)
        self._update_ai_readiness()
        self.xruns.setText(f"Underflow: {snapshot.underflows} · Overflow: {snapshot.overflows}"
                          + (" · 音切れ時はBufferを増やしてください" if snapshot.underflows or snapshot.overflows else ""))
        self._update_latency()
        now = time.monotonic()
        if now < self._next_performance_update:
            return
        self._next_performance_update = now + 0.25
        self._show_presence_status()
        self._lab_results()
        self._lab_governor_tick()
        self._update_translation_driver()
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
        self._drain_update_results()
        if self.presence is not None:
            # Cheap when nothing changed: the worker compares the snapshot itself, so
            # this each 30 Hz frame never sends anything Discord did not need.
            self.presence.update(self._presence_state())
        self._drain_voice_command()
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
                if params.experiment=='fastest':
                    post+='（最速）'
                utterance=ai.get('utterance',{})
                self.ai_performance.setText(f"発話を録音 → MeanVC2を発話ごとにリセット・末尾まで変換 → {post} → 全体を出力\n"
                    f"直近VC {ai.get('utterance_vc_seconds',0):.1f}秒 / 後処理 {ai.get('utterance_post_seconds',0):.1f}秒 · 全体RTF {ai.get('rtf',0):.2f}\n"
                    f"受け付け不可 {ai.get('utterance',{}).get('rejected',0)}発話 · 30秒分割 {ai.get('utterance',{}).get('limit_splits',0)}\n"
                    f"環境音そのまま通す {utterance.get('env_routed_seconds',0):.1f}秒（変換内）+ {utterance.get('env_filled_seconds',0):.1f}秒（発話の間）\n"
                    f"{('FlashSR失敗・MeanVC2のみで出力: '+ai.get('enhancer_error','')) if ai.get('enhancer_error') else '話し終えるまで出力しません。ライブ遅延・声質は要試聴。'}"
                    + (f"\n環境音バイパス: {utterance.get('env_error')}" if utterance.get('env_error') else '')
                    + (f"\n入力目安: メーター緑〜黄（-20dB前後）。現在 {amplitude_to_db(max(self._display_input,1e-6)):.0f}dB"
                        if params.experiment == 'fastest' else ''))
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

    def _update_stage(self, snapshot):
        """Feed the stage from the same numbers the meters show.

        Nothing here invents activity: while the engine is stopped the level is zero,
        the relay is grey and the error line is empty, so the stage cannot suggest
        signal or a fault that was never measured. The engine state is the only thing
        that colours anything: cyan while running, red on error, grey otherwise.
        """
        running = snapshot.state == 'Running' and not snapshot.error
        kind = 'error' if snapshot.error else ('running' if running else 'stopped')
        self.relay.set_state(kind)
        self.relay.set_level(self._display_input if running else 0.0)
        self.header.set_badge(kind)
        self.stage_error.setVisible(bool(snapshot.error))
        if snapshot.error:
            self.stage_error.setText(
                f'原因: {snapshot.error}\n'
                '直し方: Stopを押し、マイクと出力先を確認してからもう一度Startを押してください。')
        else:
            self.stage_error.setText('')
        if running:
            self.stage_voice.setText(self._describe_output())
        else:
            self.stage_voice.setText('停止中')
        # The drawer's 出力音声 card reuses the same attribute name; it only
        # exists in the full build, so an injected test controller skips it.
        if hasattr(self, 'output_card_text'):
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
        if mode == 'translate':
            return '同時通訳（English）'
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
        headline = 2 * one
        if reported is not None and self.controller.snapshot.state == "Running":
            text += f"\nPortAudio報告 I/O推定: {reported:.1f} ms"
            delay = (self.dsp.algorithmic_latency_samples / rate * 1000 if self.dsp and self.mode.currentData() == 'female_dsp' else 0)
            text += f" · DSP内部遅延: {delay:.1f} ms"
            text += "\n処理時間とDSP内部遅延は別の値です。"
            headline = reported
        self.latency_value.setText(f"{headline:.0f} ms")
        # The stage keeps the one number and its one caveat; the read-out behind the
        # numbers lives in 診断. Both used to be written to the same label, and the
        # stage's copy was left empty because this line rebound the name.
        self.latency.setText("いずれもエンドツーエンドの実測値ではありません。")
        self.latency_detail.setText(text)

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
            app_update_check=self.update_auto.isChecked() if hasattr(self, 'update_auto') else self.settings.app_update_check,
            discord_presence=(self.discord_toggle.isChecked() if hasattr(self, 'discord_toggle')
                              else self.settings.discord_presence),
            sample_rate=self.rate.currentData(), buffer_size=self.buffer.currentData(),
            gain_db=self.gain.value() / 10, noise_gate_db=self.gate.value() / 10,
            denoise=self.denoise_toggle.isChecked() if hasattr(self, 'denoise_toggle') else self.settings.denoise,
            denoise_strength=(self.denoise_strength.value() / 100 if hasattr(self, 'denoise_strength')
                              else self.settings.denoise_strength),
            auto_level=self.auto_level_toggle.isChecked() if hasattr(self, 'auto_level_toggle') else self.settings.auto_level,
            env_passthrough=(self.env_passthrough_toggle.isChecked() if hasattr(self, 'env_passthrough_toggle')
                             else self.settings.env_passthrough),
            finish=self.finish_toggle.isChecked() if hasattr(self, 'finish_toggle') else self.settings.finish,
            ai_formant_on=self.ai_formant_toggle.isChecked() if hasattr(self, 'ai_formant_toggle') else self.settings.ai_formant_on,
            ai_formant=(self.ai_formant_amount.value() / 10 if hasattr(self, 'ai_formant_amount')
                        else self.settings.ai_formant),
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
            ai_quality=ai.quality, ai_threads=ai.threads, ai_device=ai.device, ai_brightness=ai.brightness,
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
        if hasattr(self, 'governor_on'):
            self.lab_parameters = self._capture_lab()
            self.settings = replace(self.settings, lab=asdict(self.lab_parameters))

    def _save_settings(self):
        settings = replace(self.settings)
        self.files.submit('settings',lambda: self.manager.save(settings))

    def closeEvent(self, event: QCloseEvent):
        if self._can_close:
            self.timer.stop()
            self._release_edge_resize()
            event.accept()
            return
        event.ignore()
        if not self._closing:
            if self.voice_dialog:
                self.voice_dialog.reject()
            self._capture_settings()
            self._save_settings()
            # Leave Discord's activity clean: stop first so the last frame sent is the
            # close, not a stale "変換中".
            self._stop_presence()
            self.files.close()  # drain final save without blocking Qt
            self._closing = True
            self.status.setText("Status: Closing · 音声デバイスを解放しています")
            self._set_controls(False)
            self.controller.shutdown(self.files.wait_closed)
