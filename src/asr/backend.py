"""Local simulated partial ASR; ReazonSpeech is an OFFLINE transducer."""
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class ASRBackend(ABC):
    @abstractmethod
    def load(self): ...
    @abstractmethod
    def transcribe(self,audio): ...
    def reset(self): pass
    def unload(self): pass


class ReazonBackend(ASRBackend):
    def __init__(self,path,threads=1):
        self.path=Path(path); self.threads=threads; self.recognizer=None

    def load(self):
        import sherpa_onnx
        names=('encoder-epoch-99-avg-1.int8.onnx','decoder-epoch-99-avg-1.onnx',
               'joiner-epoch-99-avg-1.onnx','tokens.txt')
        files=[self.path/n for n in names]
        if not all(p.is_file() and p.stat().st_size>0 for p in files):
            raise ValueError('ReazonSpeech model missing: run tools/install_asr.py')
        # Native token parsing may terminate the process on malformed rows.
        # Validate before entering it; model loading still runs in an isolated worker.
        try:
            rows=files[3].read_text(encoding='utf-8').splitlines()
            tokens=[row.rsplit(maxsplit=1) for row in rows if row.strip()]
            ids=[int(row[1]) for row in tokens if len(row)==2]
            if len(ids)!=len(tokens) or not ids or len(set(ids))!=len(ids) or min(ids)<0:
                raise ValueError('Invalid token table')
            if any(p.stat().st_size<64 for p in files[:3]):
                raise ValueError('Invalid ONNX model file')
        except (UnicodeError,IndexError,ValueError) as error:
            raise ValueError('ReazonSpeech model invalid: '+str(error)) from error
        if type(self.threads) is not int or not 1<=self.threads<=4:
            raise ValueError('ASR threads must be 1..4')
        self.recognizer=sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(files[0]),decoder=str(files[1]),joiner=str(files[2]),tokens=str(files[3]),
            num_threads=self.threads,sample_rate=16000,feature_dim=80,
            decoding_method='greedy_search',provider='cpu',debug=False)

    def transcribe(self,audio):
        if self.recognizer is None: raise RuntimeError('ASR not loaded')
        if audio.ndim!=1 or audio.dtype!=np.float32 or not len(audio) or len(audio)>16000*4 or not np.isfinite(audio).all():
            raise ValueError('ASR input must be finite float32 mono, at most 4 seconds')
        stream=self.recognizer.create_stream()
        stream.accept_waveform(16000,audio)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()[:160]

    def unload(self): self.recognizer=None

    def warmup(self):
        # Exercise encoder and non-silent decoder before accepting live analysis.
        # Deterministic synthetic input, no user recording or disk access in runtime.
        t=np.arange(16000,dtype=np.float32)/16000
        probe=.08*(np.sin(2*np.pi*140*t)+.4*np.sin(2*np.pi*280*t))
        self.transcribe(probe.astype(np.float32)); self.transcribe(np.zeros(6400,dtype=np.float32))


class VoskBackend(ASRBackend):
    """True incremental Japanese decoder; no window re-decoding."""
    def __init__(self,path,threads=1):
        self.path=Path(path); self.model=self.recognizer=None

    def load(self):
        from vosk import Model, SetLogLevel
        import os
        if not (self.path/'am/final.mdl').is_file(): raise ValueError('Vosk Japanese model missing: run tools/install_asr.py --vosk')
        SetLogLevel(-1)  # native startup diagnostics only; no transcript logging
        # Vosk's Windows native file loader does not reliably accept a Unicode
        # absolute path. Workers run from project root; relative path is ASCII.
        self.model=Model(os.path.relpath(self.path,Path.cwd())); self.reset()

    def reset(self):
        if self.model is not None:
            from vosk import KaldiRecognizer
            self.recognizer=KaldiRecognizer(self.model,16000)

    def push(self,audio):
        import json
        if self.recognizer is None: raise RuntimeError('ASR not loaded')
        if audio.ndim!=1 or audio.dtype!=np.float32 or not 0<len(audio)<=16000 or not np.isfinite(audio).all():
            raise ValueError('Invalid Vosk audio chunk')
        pcm=(np.clip(audio,-1,1)*32767).astype('<i2').tobytes()
        final=bool(self.recognizer.AcceptWaveform(pcm))
        result=json.loads(self.recognizer.Result() if final else self.recognizer.PartialResult())
        return str(result.get('text' if final else 'partial','')).replace(' ','')[:160],final

    def transcribe(self,audio):
        import json
        self.reset()
        for start in range(0,len(audio),16000): self.push(audio[start:start+16000])
        text=json.loads(self.recognizer.FinalResult()).get('text','').replace(' ','')
        self.reset(); return text

    def unload(self): self.recognizer=self.model=None
