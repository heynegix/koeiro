"""Nonblocking registration dialog; heavy work lives in an isolated CPU process.

Registration runs in two worker phases. The first reports what it selected and leaves
an unpublished bundle; the dialog shows those measurements, lets the person drop
individual clips, and only then asks the worker to publish. Re-selecting therefore
never re-runs the encoder.
"""
import json
import time
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton, QScrollArea, QTextEdit, QToolBox,
    QVBoxLayout, QWidget, QSizePolicy)

from src.vc.client import worker_python
from src.vc.voice_library import ROOT

# Accepted by decodability rather than by suffix, so the dialog must not imply a
# short list either.
FILE_FILTER = ('音声ファイル (*.wav *.mp3 *.m4a *.flac *.ogg *.oga *.aac *.wma '
               '*.opus *.aiff *.aif *.mp4 *.webm);;すべてのファイル (*)')
MAX_REPORTED_CLIPS = 12
DETAIL_HEIGHT = 200


def format_report(report):
    """Render the worker's measurements as plain text, so every number is visible."""
    if not isinstance(report, dict):
        return ''
    lines = []
    selections = report.get('selections') or []
    retained = report.get('retained_seconds')
    budget = report.get('budget_seconds')
    if isinstance(retained, (int, float)):
        lines.append('採用 {}秒 / {}区間'.format(float(retained), len(selections)))
        if isinstance(budget, (int, float)):
            lines[-1] += '（上限 {:.0f}秒）'.format(float(budget))
    for key, label in (('source_seconds', '元音声'), ('speech_seconds', '発話時間'),
                       ('region_count', '発話区間'), ('candidate_count', '判定区間'),
                       ('accepted_count', '採用候補')):
        value = report.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            lines.append('{}: {}'.format(label, '{:.2f}'.format(float(value))
                                         if isinstance(value, float) else value))
    contamination = report.get('contamination')
    if isinstance(contamination, dict) and contamination.get('flags'):
        labels = {'background_music_suspected': 'BGM疑い', 'little_silence': '無音が少ない',
                  'multiple_speakers_possible': '複数話者の疑い'}
        lines.append('注意: ' + ', '.join(labels.get(flag, flag)
                                          for flag in contamination['flags']))
        gap = contamination.get('gap_level_db')
        if isinstance(gap, (int, float)):
            lines.append('  発話/無音の差: {:.1f}dB'.format(float(gap)))
    rejected = report.get('rejected') or {}
    if rejected:
        lines.append('不採用: ' + ', '.join('{}×{}'.format(k, v) for k, v in sorted(rejected.items())))
    if selections:
        lines.append('')
        lines.append('{:>3} {:>8} {:>7} {:>8} {:>8} {:>6}'.format(
            '#', '開始', '長さ', 'SNR', '声高', '品質'))
        for index, row in enumerate(selections[:MAX_REPORTED_CLIPS], 1):
            snr = row.get('snr_db')
            pitch = row.get('f0_median_hz')
            lines.append('{:>3} {:>7}s {:>6}s {:>8} {:>8} {:>6}'.format(
                index,
                float(row.get('offset_seconds', 0.0)),
                float(row.get('duration_seconds', 0.0)),
                '{:.0f}dB'.format(float(snr)) if isinstance(snr, (int, float)) else '-',
                '{:.0f}Hz'.format(float(pitch)) if isinstance(pitch, (int, float)) else '-',
                float(row.get('quality', 0.0))))
        if len(selections) > MAX_REPORTED_CLIPS:
            lines.append('… 他{}区間'.format(len(selections) - MAX_REPORTED_CLIPS))
    for advice in report.get('advice') or []:
        lines.append('※ ' + str(advice))
    return '\n'.join(lines)


class VoiceRegistrationDialog(QDialog):
    def __init__(self, parent=None, ready=lambda: True, voices=None):
        super().__init__(parent)
        self.setWindowTitle('音声ファイルから声を追加')
        self.setMinimumWidth(460)
        self.resize(500, 620)
        self.setSizeGripEnabled(True)
        self.setModal(True)
        self.ready = ready
        self.registered_id = None
        self.process = None
        self.source = ''
        self.sources = []
        self.buffer = b''
        self.last_error = ''
        self.started = 0.
        self.log_file = None
        self.cancel_requested = False
        # Pending-bundle state: set after phase 1, consumed by the publish phase.
        self.bundle = ''
        self.bundle_report = None
        self.bundle_sources = []
        self.target_voice = ''
        self.clips = []
        self.pending_row = None
        self.on_success = None
        self.last_report = None
        self.layout = QVBoxLayout(self)
        heading = QLabel('新しい声をライブラリへ')
        heading.setObjectName('sectionTitle')
        self.layout.addWidget(heading)
        # The body scrolls so a long clip list can never push the confirm
        # buttons off-screen; the buttons stay fixed at the bottom.
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(body)
        self.layout.addWidget(scroll, 1)
        self.toolbox = QToolBox()
        body_layout.addWidget(self.toolbox)
        # Page 1: source audio and name.
        page_files = QWidget()
        files_layout = QVBoxLayout(page_files)
        files_layout.setContentsMargins(4, 4, 4, 4)
        note = QLabel('一人で話している、BGMのない録音が適しています。\n'
                      '20秒以上を推奨。話している部分だけを自動で選びます。')
        note.setWordWrap(True)
        files_layout.addWidget(note)
        self.file_button = QPushButton('音声ファイルを選ぶ…')
        self.file_button.clicked.connect(self.choose_file)
        files_layout.addWidget(self.file_button)
        self.filename = QLabel('WAV / MP3 / M4A / FLAC / OGG / AAC / WMA / Opus など')
        self.filename.setWordWrap(True)
        self.filename.setTextFormat(Qt.TextFormat.PlainText)
        files_layout.addWidget(self.filename)
        files_layout.addWidget(QLabel('声の名前'))
        self.name = QLineEdit()
        self.name.setMaxLength(80)
        self.name.setPlaceholderText('例：落ち着いた声')
        files_layout.addWidget(self.name)
        # Phase 5: extend a registered voice instead of creating one.
        self.add_group = QWidget()
        add_layout = QHBoxLayout(self.add_group)
        add_layout.setContentsMargins(0, 0, 0, 0)
        self.add_to = QCheckBox('既存の声に追加する')
        add_layout.addWidget(self.add_to)
        self.voice_picker = QComboBox()
        self.voice_picker.setPlaceholderText('追加先の声を選ぶ')
        self.voice_picker.setEnabled(False)
        self.set_voices(voices or [])
        add_layout.addWidget(self.voice_picker, 1)
        self.add_to.toggled.connect(self.voice_picker.setEnabled)
        files_layout.addWidget(self.add_group)
        self.toolbox.addItem(page_files, '① 音声ファイルと名前')
        # Page 2: clip selection with bulk toggles.
        page_clips = QWidget()
        clips_layout = QVBoxLayout(page_clips)
        clips_layout.setContentsMargins(4, 4, 4, 4)
        bulk = QHBoxLayout()
        self.select_all_button = QPushButton('すべて選択')
        self.select_all_button.clicked.connect(lambda: self.set_all_clips(True))
        self.select_none_button = QPushButton('すべて外す')
        self.select_none_button.clicked.connect(lambda: self.set_all_clips(False))
        bulk.addWidget(self.select_all_button)
        bulk.addWidget(self.select_none_button)
        bulk.addStretch(1)
        clips_layout.addLayout(bulk)
        self.clip_rows = QWidget()
        self.clip_layout = QVBoxLayout(self.clip_rows)
        self.clip_layout.setContentsMargins(0, 0, 0, 0)
        self.clip_layout.setSpacing(2)
        clips_layout.addWidget(self.clip_rows)
        self.toolbox.addItem(page_clips, '② 使う区間を選ぶ')
        # Page 3: full measurement report.
        page_details = QWidget()
        details_layout = QVBoxLayout(page_details)
        details_layout.setContentsMargins(4, 4, 4, 4)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.details.setMinimumHeight(DETAIL_HEIGHT)
        self.details.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        details_layout.addWidget(self.details)
        self.toolbox.addItem(page_details, '③ 詳細レポート')
        # Clip and report pages stay disabled until phase 1 produces a bundle.
        self.toolbox.setItemEnabled(1, False)
        self.toolbox.setItemEnabled(2, False)
        self.status = QLabel('元のファイルは変更しません。登録後も、今の声へ戻せます。')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.hide()
        self.layout.addWidget(self.progress)
        actions = QHBoxLayout()
        self.submit = QPushButton('この声を登録')
        self.submit.setObjectName('start')
        self.submit.clicked.connect(self.on_submit)
        self.cancel_button = QPushButton('閉じる')
        self.cancel_button.clicked.connect(self.reject)
        actions.addWidget(self.submit)
        actions.addWidget(self.cancel_button)
        self.layout.addLayout(actions)
        self.setTabOrder(self.file_button, self.name)
        self.setTabOrder(self.name, self.add_to)
        self.setTabOrder(self.add_to, self.voice_picker)
        self.setTabOrder(self.voice_picker, self.submit)
        self.setTabOrder(self.submit, self.cancel_button)
        self.watchdog = QTimer(self)
        self.watchdog.setInterval(750)
        self.watchdog.timeout.connect(self.check_resources)

    def voices(self):
        return self._voices

    def set_voices(self, voices):
        """Offer existing voices as add-to destinations.

        Each entry is either ``(label, identifier)`` or a bare identifier string. The
        worker resolves the folder from the identifier, so the label is only ever
        displayed; passing the label as the target silently failed validation.
        """
        entries = []
        for entry in voices or []:
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                label, identifier = str(entry[0]), str(entry[1])
            else:
                label = identifier = str(entry)
            entries.append((label, identifier))
        self._voices = [identifier for _label, identifier in entries]
        if not hasattr(self, 'voice_picker'):
            return
        self.voice_picker.clear()
        for label, identifier in entries:
            self.voice_picker.addItem(label, identifier)
        self.voice_picker.setCurrentIndex(0 if entries else -1)
        # Adding audio to an existing voice only exists when there is a voice to
        # extend: the shipped standard voice is fixed, so it is never offered and
        # an empty list hides the whole row instead of showing a dead picker.
        if hasattr(self, 'add_group'):
            self.add_group.setVisible(bool(entries))

    def choose_file(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Reference音声を選択', '', FILE_FILTER)
        if paths:
            self.set_sources(paths)

    def set_sources(self, paths):
        paths = list(paths)
        self.sources = paths
        self.source = paths[0] if paths else ''
        if len(paths) == 1:
            self.filename.setText(Path(paths[0]).name)
        elif paths:
            self.filename.setText('{} 個を選択（{}…）'.format(len(paths), Path(paths[0]).name))
        if not self.name.text().strip() and paths:
            self.name.setText(Path(paths[0]).stem[:80])

    def on_submit(self):
        """One button for both phases: analyse first, then publish what was kept."""
        if self.bundle:
            self.publish_selected()
        else:
            self.begin()

    def begin(self):
        self.target_voice = (str(self.voice_picker.currentData() or '').strip()
                             if self.add_to.isChecked() else '')
        if self.add_to.isChecked() and not self.target_voice:
            self.status.setText('追加先の声を選んでください。')
            return
        self.start_process(self.build_arguments(), self.on_analysis_ready)

    def build_arguments(self):
        arguments = []
        for path in self.sources:
            arguments += ['--source', path]
        arguments += ['--name', self.name.text().strip()]
        if self.target_voice:
            arguments += ['--add-to', self.target_voice]
        return arguments

    def start_process(self, arguments, on_success):
        if self.process is not None:
            return
        if not self.ready():
            self.status.setText('変換Workerの停止を待っています。少し待ってから登録してください。')
            return
        try:
            import psutil
            if psutil.virtual_memory().available < 2.5*1024**3:
                raise RuntimeError('空きRAMが不足しています。他の重いアプリを閉じてから登録してください。')
            executable, environment = worker_python(ROOT)
            qt_environment = QProcessEnvironment()
            for key, value in environment.items():
                qt_environment.insert(key, value)
            qt_environment.insert('PYTHONIOENCODING', 'utf-8')
            qt_environment.insert('PYTHONDONTWRITEBYTECODE', '1')
            (ROOT/'logs').mkdir(exist_ok=True)
            self.log_file = (ROOT/'logs/voice-registration.log').open('ab', buffering=0)
            self.buffer = b''
            self.last_error = ''
            self.cancel_requested = False
            self.on_success = on_success
            self.pending_row = None
            process = QProcess(self)
            process.setProcessEnvironment(qt_environment)
            process.setWorkingDirectory(str(ROOT))
            process.readyReadStandardOutput.connect(self.read_output)
            process.readyReadStandardError.connect(self.read_error)
            process.finished.connect(self.process_finished)
            process.errorOccurred.connect(self.process_error)
            self.process = process
            self.started = time.monotonic()
            for widget in (self.file_button, self.name, self.add_to, self.voice_picker):
                widget.setEnabled(False)
            self.cancel_button.setText('登録を中止')
            self.progress.setRange(0, 0)
            self.progress.show()
            self.status.setText('準備しています…')
            process.start(str(executable), ['-u', str(ROOT/'tools/register_voice.py')] + arguments)
            self.watchdog.start()
        except (OSError, RuntimeError, ImportError) as error:
            self.status.setText(str(error))

    def on_analysis_ready(self, row):
        """Phase 1 finished: show the measurements and let the person adjust the clips."""
        self.bundle = row.get('bundle', '')
        # The report may arrive on its own progress line before the bundle line.
        self.bundle_report = row.get('report') or self.last_report
        self.bundle_sources = row.get('sources') or []
        if not self.bundle or not self.bundle_report:
            self.status.setText('選択データを受信できませんでした。最初からやり直してください。')
            return
        self.show_report(self.bundle_report)
        self.build_clip_rows()
        self.toolbox.setItemEnabled(1, True)
        self.toolbox.setItemEnabled(2, True)
        self.toolbox.setCurrentIndex(1)
        self.submit.setText('この内容で登録')
        self.submit.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.hide()
        self.status.setText('{}区間を選択しました。必要なら外してから登録してください。'
                            .format(len(self.bundle_report.get('selections') or [])))

    def build_clip_rows(self):
        while self.clip_layout.count():
            item = self.clip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.clips = []
        selections = (self.bundle_report or {}).get('selections') or []
        for index, row in enumerate(selections):
            box = QCheckBox('区間 {}：{:.1f}s〜{:.1f}s（{:.1f}秒 / SNR {} / 声高 {}）'.format(
                index + 1, float(row.get('offset_seconds', 0.0)),
                float(row.get('offset_seconds', 0.0)) + float(row.get('duration_seconds', 0.0)),
                float(row.get('duration_seconds', 0.0)),
                ('{:.0f}dB'.format(float(row['snr_db']))
                 if isinstance(row.get('snr_db'), (int, float)) else '-'),
                ('{:.0f}Hz'.format(float(row['f0_median_hz']))
                 if isinstance(row.get('f0_median_hz'), (int, float)) else '-')), self)
            box.setChecked(True)
            box.toggled.connect(self.update_selected_summary)
            self.clip_layout.addWidget(box)
            self.clips.append(box)
        self.update_selected_summary()

    def set_all_clips(self, checked):
        for box in self.clips:
            box.setChecked(checked)
        self.update_selected_summary()

    def selected_indices(self):
        return [index for index, box in enumerate(self.clips) if box.isChecked()]

    def update_selected_summary(self):
        indices = self.selected_indices()
        report = self.bundle_report or {}
        total = sum(float((report.get('selections') or [])[index].get('duration_seconds', 0.0))
                    for index in indices)
        self.status.setText('{}区間 / {:.1f}秒 を選択中。'.format(len(indices), total))

    def publish_selected(self):
        indices = self.selected_indices()
        if not indices:
            self.status.setText('発話区間を1つ以上選んでください。')
            return
        self.progress.setRange(0, 0)
        self.progress.show()
        self.status.setText('登録しています…')
        arguments = ['--finalize', self.bundle, '--name', self.name.text().strip() or '声',
                     '--clips', ','.join(str(index) for index in indices)]
        self.start_process(arguments, self.on_registered)

    def on_registered(self, row):
        self.registered_id = row.get('registered')
        self.show_report(row.get('report'))
        self.accept()

    def show_report(self, report):
        text = format_report(report)
        if not text:
            return
        self.details.setPlainText(text)

    def read_output(self):
        if self.process is None:
            return
        self.buffer += bytes(self.process.readAllStandardOutput())
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            if self.log_file:
                self.log_file.write(line+b'\n')
            try:
                row = json.loads(line.decode('utf-8'))
            except (ValueError, TypeError):
                continue
            if 'message' in row:
                self.status.setText(row['message'])
            if 'error' in row:
                self.last_error = row['error']
            if 'report' in row:
                self.last_report = row['report']
                self.show_report(row['report'])
            if 'registered' in row:
                self.registered_id = row['registered']
            if 'bundle' in row:
                self.pending_row = row
        self.buffer = self.buffer[-65536:]

    def read_error(self):
        if self.process is not None:
            data = bytes(self.process.readAllStandardError())
            if self.log_file:
                self.log_file.write(data)

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.last_error = '登録用Pythonを起動できませんでした。'+self.process.errorString()
            self.process_finished(-1, QProcess.ExitStatus.CrashExit)

    def process_finished(self, code, status):
        self.read_output()
        self.read_error()
        self.watchdog.stop()
        if self.process is not None:
            self.process.deleteLater()
            self.process = None
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        if self.cancel_requested:
            self.discard_bundle()
            super().reject()
            return
        handler = getattr(self, 'on_success', None)
        if code == 0 and status == QProcess.ExitStatus.NormalExit and handler:
            row = getattr(self, 'pending_row', None)
            if row is None and self.registered_id:
                row = dict(registered=self.registered_id)
            self.pending_row = None
            if row:
                handler(row)
                return
        self.registered_id = None
        self.progress.hide()
        self.status.setText(self.last_error or '登録に失敗しました。logs/voice-registration.logを確認してください。')
        for widget in (self.file_button, self.name, self.add_to, self.voice_picker):
            widget.setEnabled(True)
        self.cancel_button.setText('閉じる')
        self.submit.setText('この声を登録')
        self.submit.setEnabled(True)

    def discard_bundle(self):
        """A cancelled or failed publish must not leave an unpublished bundle behind."""
        if not self.bundle:
            return
        path = Path(self.bundle)
        parent = path.parent
        if path.is_dir() and parent.name == 'user_voices' and path.name.startswith('.pending_'):
            import shutil
            shutil.rmtree(path, ignore_errors=True)
        self.bundle = ''

    def check_resources(self):
        if self.process is None or not self.process.processId():
            return
        import psutil
        try:
            process = psutil.Process(self.process.processId())
            memory = sum(p.memory_info().rss for p in [process]+process.children(recursive=True))
            if memory > 4.5*1024**3 or psutil.virtual_memory().available < 1.25*1024**3:
                self.last_error = 'RAM使用量が上限に達したため、登録を停止しました。'
                self.terminate_worker()
            elif time.monotonic()-self.started > 900:
                self.last_error = '登録が15分以内に完了しなかったため、処理を停止しました。'
                self.terminate_worker()
        except psutil.Error:
            pass

    def terminate_worker(self):
        if self.process is None:
            return
        import psutil
        try:
            for child in psutil.Process(self.process.processId()).children(recursive=True):
                child.kill()
        except psutil.Error:
            pass
        self.process.kill()

    def reject(self):
        if self.process is not None:
            self.cancel_requested = True
            self.status.setText('登録を中止しています…')
            self.terminate_worker()
        elif self.bundle:
            self.discard_bundle()
            super().reject()
        else:
            super().reject()