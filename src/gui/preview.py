"""Offline playback of converted audio, produced without touching an output device.

The preview converts a WAV through the same MeanVC2 worker route the live path uses,
writes the result to a cache directory, and plays it with QtMultimedia. It exists so a
person can hear what a voice sounds like *before* committing to it in conversation, and
so two settings can be compared back to back. The conversion runs in the worker process
like every other inference; the GUI thread only plays the finished file.
"""
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/'logs'/'preview'
# Anything longer than this is not a preview; the collector would split it anyway.
MAX_SECONDS = 30
SAMPLE_RATE = 48000
# Bumped whenever the conversion path or its options change, so a stale cached file
# from an older build is never replayed as if it were current.
PREVIEW_FORMAT_VERSION = 'v4-tune11-natural'


class PreviewError(RuntimeError):
    """Conversion or playback preparation failed; the message is shown to the user."""


def read_wav(path, rate=SAMPLE_RATE):
    """Read a WAV file as mono float32 at `rate`. Only PCM and float RIFF are read."""
    try:
        with wave.open(str(path), 'rb') as stream:
            source_rate = stream.getframerate()
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            frames = stream.readframes(stream.getnframes())
    except (wave.Error, EOFError, OSError) as error:
        raise PreviewError('WAVファイルを読めませんでした: '+str(error)) from error
    if width == 2:
        import numpy as np
        audio = np.frombuffer(frames, dtype='<i2').astype(np.float32)/32768.0
    elif width == 4:
        import numpy as np
        audio = np.frombuffer(frames, dtype='<i4').astype(np.float32)/2147483648.0
    elif width == 1:
        import numpy as np
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32)-128.0)/128.0
    else:
        raise PreviewError('未対応のWAV形式です（16bit/24bit/32bit/floatのみ）')
    import numpy as np
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if source_rate != rate and len(audio):
        # Linear interpolation rather than scipy: the preview must work in the GUI
        # environment, which does not ship scipy, and this path only feeds playback.
        count = max(1, int(round(len(audio)*rate/source_rate)))
        audio = np.interp(np.linspace(0, len(audio)-1, count),
                          np.arange(len(audio)), audio).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def write_wav(path, audio, rate=SAMPLE_RATE):
    import numpy as np
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((clipped*32767).astype('<i2').tobytes())
    return path


def worker_python(executable=None, environment=None):
    """Reuse the client's worker resolution so the preview runs the same interpreter."""
    from src.vc.client import worker_python as resolve
    from src.vc.voice_library import ROOT as library_root
    return resolve(executable if callable(executable) else library_root)


def convert(source, model, enhancer='none', denoise=False, experiment='none', timeout=600, runner=None,
            tune_sib_db=3.0, tune_cons_db=3.0, tune_caps=1.0, tune_floor_db=3.0, tune_excess_db=9.0,
            tune_mid=0.8, tune_match=0.25, tune_ptrans=0.20, tune_pcap=1.0,
            tune_combined=True, tune_level_db=-20.0):
    """Run one utterance conversion through the worker and return the output path.

    The worker speaks the same line protocol as the live bridge, so this exercises the
    real route including its delay trimming rather than a simplified offline shortcut.
    """
    import numpy as np
    source = Path(source)
    if not source.is_file():
        raise PreviewError('変換する音声が見つかりません。')
    audio = read_wav(source)
    if not len(audio):
        raise PreviewError('変換する音声が空です。')
    if len(audio) > MAX_SECONDS*SAMPLE_RATE:
        audio = audio[:MAX_SECONDS*SAMPLE_RATE]
    if float(np.max(np.abs(audio), initial=0.0)) < 1e-5:
        raise PreviewError('変換する音声が無音です。')

    from src.vc.client import worker_python as resolve
    from src.vc.voice_library import ROOT as library_root
    from src.vc.protocol import receive, send
    resolved = resolve(library_root)
    executable, environment = resolved if isinstance(resolved, tuple) else (resolved, None)
    if environment is None:
        environment = os.environ.copy()
    environment.update(PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
                      HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    command = [str(executable), '-u', '-m', 'src.vc.service', '--factor', '1',
               '--threads', '1', '--model', str(model), '--delivery', 'utterance',
               '--enhancer', str(enhancer), '--experiment', str(experiment),
               '--tune-sib-db', str(tune_sib_db), '--tune-cons-db', str(tune_cons_db),
               '--tune-caps', str(tune_caps), '--tune-floor-db', str(tune_floor_db),
               '--tune-excess-db', str(tune_excess_db), '--tune-mid', str(tune_mid),
               '--tune-match', str(tune_match), '--tune-ptrans', str(tune_ptrans),
               '--tune-pcap', str(tune_pcap), '--tune-level-db', str(tune_level_db),
               '--tune-combined' if tune_combined else '--no-tune-combined']
    if denoise:
        command.append('--lavasr-denoise')
    process = runner(command, environment) if runner else subprocess.Popen(
        command, cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env=environment,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        # The worker announces Loading, Warming Up and then Ready. Requests sent before
        # Ready would be answered by the loading path rather than the conversion path,
        # which is why the handshake is drained instead of assumed.
        send(process.stdin, {'op': 'reset'})
        deadline = time.monotonic()+timeout
        ready = False
        while time.monotonic() < deadline:
            header, _ = receive(process.stdout)
            status = header.get('status')
            if status == 'Error':
                raise PreviewError(header.get('error', '変換の準備に失敗しました。'))
            if status == 'Ready':
                ready = True
                break
        if not ready:
            raise PreviewError('変換ワーカーが時間内に準備できませんでした。')
        send(process.stdin, {'op': 'utterance'}, audio)
        while True:
            header, result = receive(process.stdout)
            status = header.get('status')
            if status == 'Error':
                raise PreviewError(header.get('error', '変換に失敗しました。'))
            if header.get('op') == 'utterance':
                break
        send(process.stdin, {'op': 'stop'})
    except PreviewError:
        raise
    except (OSError, ValueError) as error:
        raise PreviewError('変換ワーカーと通信できませんでした: '+str(error)) from error
    finally:
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
    result = np.asarray(result, dtype=np.float32)
    if result.size != len(audio):
        raise PreviewError('変換後の長さが元音声と一致しません（%d → %dサンプル）。'
                           % (len(audio), result.size))
    CACHE.mkdir(parents=True, exist_ok=True)
    # Every input that changes the result must reach the key. The enhancer name is in
    # here, which is what stops a LavaSR preview from being served a plain conversion.
    import hashlib
    digest = hashlib.sha256(b''.join((
        str(model).encode('utf-8'), str(enhancer).encode('utf-8'),
        str(bool(denoise)).encode('utf-8'), str(experiment).encode('utf-8'),
        str(tune_sib_db).encode('utf-8'), str(tune_cons_db).encode('utf-8'),
        str(tune_caps).encode('utf-8'), str(tune_floor_db).encode('utf-8'),
        str(tune_excess_db).encode('utf-8'), str(tune_mid).encode('utf-8'),
        str(tune_match).encode('utf-8'), str(tune_ptrans).encode('utf-8'),
        str(tune_pcap).encode('utf-8'), str(bool(tune_combined)).encode('utf-8'),
        str(tune_level_db).encode('utf-8'),
        source.name.encode('utf-8'), str(len(audio)).encode('utf-8'),
        PREVIEW_FORMAT_VERSION.encode('utf-8'),
        audio.tobytes(),
    ))).hexdigest()[:16]
    target = CACHE/('preview_%s.wav' % digest)
    write_wav(target, result)
    return target, header.get('stats') or {}


def describe(stats):
    """One-line summary of the worker's own measurements for the preview."""
    if not isinstance(stats, dict) or 'rtf' not in stats:
        return ''
    guards = stats.get('guards') or {}
    parts = ['RTF %.3f' % float(stats['rtf'])]
    if guards.get('input_silence_seconds') is not None:
        parts.append('無音 %.1f秒' % float(guards['input_silence_seconds']))
    if guards.get('input_fry') is not None:
        try:
            parts.append('入力の荒れ %.2f' % float(guards['input_fry']['fry']))
        except (TypeError, KeyError, ValueError):
            pass
    if guards.get('enhancer_skipped'):
        parts.append('復元は省略（%s）' % guards.get('enhancer_skip_reason', '無音'))
    if stats.get('enhancer') and stats.get('enhancer') != 'none':
        parts.append(stats['enhancer'])
    return ' · '.join(parts)


if __name__ == '__main__':
    print(json.dumps({'cache': str(CACHE), 'root': str(ROOT), 'max_seconds': MAX_SECONDS},
                     ensure_ascii=False))