from abc import ABC, abstractmethod


class VoiceConversionBackend(ABC):
    """Worker-only, stateful mono float32 backend at its declared sample rate."""
    sample_rate = 16000

    @abstractmethod
    def load(self, model_path): ...

    @abstractmethod
    def warmup(self): ...

    @abstractmethod
    def process_chunk(self, audio): ...

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def unload(self): ...

    @abstractmethod
    def get_stats(self): ...
