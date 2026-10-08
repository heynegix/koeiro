"""基準RTF測定用の最小スタブ。

まずAI Voice RPC経路全体の壁時計RTFと、ワーカ側の主要statsを出す。
この段階ではアプリGUIも音声デバイスも使わず、決められた短いWAVを
固定モデル・固定設定で往復させる。品質には手を入れず、測定だけ。
"""
import json
from pathlib import Path
import sys
import time
import wave
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vc.bridge import AIBridge
from src.vc.config import AIParameters
from src.processors.ai_voice import AIVoiceProcessor
from src.processors.voice_router import VoiceRouter


def main():
    from tools.compare_meanvc2_continuity import require_idle_voice_worker
    require_idle_voice_worker()

    root = Path.cwd()
    report_path = root / 'test-results/rtf_baseline_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model = 'meanvc2_ref20'
    threads = 2
    quality = 'balanced'

    ai_params = AIParameters(model=model, threads=threads, quality=quality,
                             delivery='streaming', enhancer='none')
    bridge = AIBridge(ai_params)
    processor = AIVoiceProcessor(bridge=bridge)

    class UnusedDSP:
        def prepare(self, *args): pass
        def reset(self): pass
        def process(self, audio, *args): return audio

    router = VoiceRouter(dsp=UnusedDSP(), ai=processor, mode='ai_voice')

    report = dict(
        stage='setup',
        model=model,
        threads=threads,
        quality=quality,
        ai_parameters=ai_params.__dict__ if hasattr(ai_params, '__dict__') else {},
        input_kind='Offline RPC replay; no audio devices',
    )

    def save_now(payload):
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding='utf-8')

    try:
        bridge.load()
        t0 = time.monotonic()
        while bridge.status != 'Ready':
            if bridge.status == 'Error' or time.monotonic() - t0 > 120:
                raise RuntimeError(bridge.error or 'Load timeout')
            time.sleep(0.01)
        load_seconds = time.monotonic() - t0
        report['load_seconds'] = load_seconds

        router.prepare(48000, 256)

        source_wav = root / 'recordings/v011_mega_tournament/source/source_normal.wav'
        if not source_wav.is_file():
            raise FileNotFoundError(f'Reference source not found: {source_wav}')

        with wave.open(str(source_wav), 'rb') as src:
            sr_src = src.getframerate()
            if sr_src != 48000 or src.getnchannels() != 1:
                raise ValueError('Baseline source must be 48 kHz mono')
            max_seconds = 6.0
            raw = src.readframes(round(48000 * max_seconds))
            audio = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768
            audio = audio[:round(48000 * max_seconds)]

        start = time.monotonic()
        next_callback = start
        parts = []
        sent = False
        finish_time = first_output = None

        def feed_block():
            nonlocal next_callback, sent, finish_time, first_output
            pos = len(parts) * 256
            block = np.zeros((256, 1), dtype=np.float32)
            piece = audio[pos:pos + 256]
            block[:len(piece), 0] = piece
            router.process(block, 48000)
            parts.append(block[:, 0].copy())
            if np.max(np.abs(block)) > 0.0001 and first_output is None:
                first_output = time.monotonic()
            if not sent and pos + 256 >= len(audio):
                bridge.finish_utterance.set()
                bridge.input_ready.signal()
                sent = True
                finish_time = time.monotonic()
            if bridge.status == 'Error':
                raise RuntimeError(bridge.error)
            next_callback += 256 / 48000
            return time.monotonic() - next_callback

        # 発話区切り付きの壁時計再生を模倣して、最初の1発話が出るまで回す
        deadline = start + 120
        while time.monotonic() < deadline:
            sleep_for = feed_block()
            if sleep_for > 0:
                time.sleep(sleep_for)
            if bridge.utterance_stats.get('completed', 0) >= 1 and bridge.output.available == 0:
                break

        end = time.monotonic()
        wall_seconds = end - start

        snapshot = bridge.snapshot()
        result = np.concatenate(parts)

        report.update(
            stage='complete',
            wall_seconds=wall_seconds,
            first_output_seconds=first_output - start if first_output else None,
            finish_signal_seconds=finish_time - start if finish_time else None,
            sent_at_seconds=finish_time - start if finish_time else None,
            after_finish_first_output_seconds=(first_output - finish_time
                                               if first_output and finish_time else None),
            snapshot=snapshot,
            overrun=bridge.overruns,
            underrun=bridge.underrun,
            output_overrun=bridge.output_overruns,
            dropped_frames=bridge.dropped_frames,
            source_seconds=len(audio) / 48000,
            source_sample_count=len(audio),
        )

        save_now(report)

        summary = {
            'status': 'SUCCESS',
            'model': model,
            'threads': threads,
            'quality': quality,
            'wall_seconds': wall_seconds,
            'load_seconds': load_seconds,
            'first_output_seconds': report['first_output_seconds'],
            'finish_signal_seconds': report['finish_signal_seconds'],
            'after_finish_first_output_seconds': report['after_finish_first_output_seconds'],
            'source_seconds': report['source_seconds'],
            'rtf': wall_seconds / report['source_seconds'],
            'ai': {k: snapshot.get(k) for k in ('status', 'rtf', 'inference', 'process_total', 'resample',
                                                   'postprocess', 'ram_bytes', 'worker_gc_max_ms')},
            'bridge': {k: snapshot.get(k) for k in ('generation', 'worker_state', 'rpc', 'reset',
                                                      'queue_current', 'queue_max', 'queue_capacity',
                                                      'ai_underrun', 'ai_overrun', 'dropped_chunks')},
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        t1 = time.monotonic()
        bridge.stop()
        report['stop_seconds'] = time.monotonic() - t1
        save_now(report)


if __name__ == '__main__':
    main()
