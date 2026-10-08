"""Offline registration + WAV RPC verification. Never opens microphone/output devices."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
import numpy as np
import psutil
from src.gui.voice_registration import VoiceRegistrationDialog
from src.vc.voice_library import digest, metadata


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--registered-id')
    selection.add_argument('--profile', help='Built-in or completed naturalness comparison profile ID')
    parser.add_argument('--source', type=Path, default=ROOT/'recordings/v011_mega_tournament/source/source_normal.wav')
    parser.add_argument('--seconds', type=int, default=8)
    parser.add_argument('--output-label', help='Separate report/WAV stem; does not change saved app settings')
    args = parser.parse_args()
    folder = ROOT/'validation/v011'
    folder.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    built = {path.as_posix(): digest(path) for profile in ('meanvc2_ref20', 'meanvc2_ref60')
             for path in (ROOT/'models'/profile).glob('*') if path.suffix in ('.wav', '.npy', '.json')}
    report = dict(kind='OFFLINE REGISTRATION + WAV RPC; no audio devices, no human listening',
                  registration_peak_ram_bytes=0, rpc_peak_ram_bytes=0)
    identifier = args.profile or args.registered_id
    report_path = folder/(args.profile+'_verification.json' if args.profile else 'voice_library_verification.json')
    if args.output_label:
        if not args.output_label.replace('_','').replace('-','').isalnum():
            raise ValueError('Output label must be alphanumeric with underscores/hyphens')
        report_path = folder/(args.output_label+'_verification.json')
    if identifier is None:
        dialog = VoiceRegistrationDialog()
        dialog.source = str(ROOT/'recordings/v011_mega_tournament/reference/reference_05s.wav')
        dialog.name.setText('自分の声 · 登録機能の確認')
        dialog.open()
        dialog.begin()
        until = time.monotonic()+910
        while dialog.process is not None and time.monotonic() < until:
            app.processEvents()
            pid = dialog.process.processId() if dialog.process else 0
            if pid:
                try:
                    report['registration_peak_ram_bytes'] = max(report['registration_peak_ram_bytes'], psutil.Process(pid).memory_info().rss)
                except psutil.Error:
                    pass
            time.sleep(.05)
        if dialog.process is not None:
            dialog.reject()
            raise RuntimeError('Registration deadline exceeded')
        if not dialog.registered_id:
            raise RuntimeError(dialog.status.text())
        identifier = dialog.registered_id
        print('REGISTERED '+identifier, flush=True)
    if args.profile:
        from src.vc.models import profile
        model_folder = ROOT/'models'/profile(identifier)['folder']
        report.update(profile=identifier, kind='OFFLINE WAV RPC; no audio devices, no human listening',
                      registration=dict(embedding_sha256=digest(model_folder/'fixed_embedding.npy')))
    else:
        report.update(registered_id=identifier, registration=metadata(identifier))
    # Persist a resume record before neural replay, so reference extraction needn't repeat.
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    from src.vc.models import refresh_user_profiles
    refresh_user_profiles()
    from src.vc.config import AIParameters
    from src.vc.client import ServiceClient
    client = ServiceClient(AIParameters(model=identifier, threads=1))
    done = threading.Event()
    guard_error = []
    def guard():
        deadline = time.monotonic()+180
        while not done.wait(.15):
            try:
                if client.process:
                    memory = psutil.Process(client.process.pid).memory_info().rss
                    report['rpc_peak_ram_bytes'] = max(report['rpc_peak_ram_bytes'], memory)
                    if memory > 4.5*1024**3 or psutil.virtual_memory().available < 1.25*1024**3 or time.monotonic() > deadline:
                        guard_error.append('RPC RAM/time guard triggered')
                        client.interrupt()
                        return
            except psutil.Error:
                pass
    client.start()
    monitor = threading.Thread(target=guard, daemon=True)
    monitor.start()
    try:
        while True:
            header, _ = client.read()
            if header['status'] == 'Ready':
                report['worker_model'] = header['model']
                break
        assert report['worker_model']['reference_fixed']
        assert report['worker_model']['fixed_embedding_sha256'] == report['registration']['embedding_sha256']
        path = args.source
        with wave.open(str(path), 'rb') as source:
            assert source.getframerate() == 48000 and source.getsampwidth() == 2 and source.getnchannels() == 1
            audio = np.frombuffer(source.readframes(48000*args.seconds), dtype='<i2').astype(np.float32)/32768
        hop = 7680
        output = []
        start = time.monotonic()
        for offset in range(0, len(audio)+48000*4, hop):
            chunk = np.zeros(hop, dtype=np.float32)
            piece = audio[offset:offset+hop]
            chunk[:len(piece)] = piece
            header, result = client.request(dict(op='process'), chunk)
            if 'stats' in header:
                report['latest_worker_stats'] = header['stats']
            assert len(result) == hop and np.isfinite(result).all()
            output.append(result)
        converted = np.concatenate(output)
        assert np.max(np.abs(converted)) > .001
        destination = ROOT/'recordings/v011_voice_library'/(args.profile+'_rpc.wav' if args.profile else 'registration_rpc.wav')
        if args.output_label:
            destination = destination.with_name(args.output_label+'_rpc.wav')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), 'wb') as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(48000)
            target.writeframes((np.clip(converted, -1, 1)*32767).astype('<i2').tobytes())
        report.update(rpc_conversion_seconds=time.monotonic()-start,
            final_worker_header=header,
            input_seconds=len(audio)/48000, output_seconds=len(converted)/48000,
            output=str(destination), output_peak=float(np.max(np.abs(converted))))
    finally:
        done.set()
        client.close()
        monitor.join(2)
    if guard_error:
        raise RuntimeError(guard_error[0])
    report['built_in_files_unchanged'] = all(digest(path) == value for path, value in built.items())
    assert report['built_in_files_unchanged']
    report['status'] = 'SUCCESS'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
