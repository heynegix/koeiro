from abc import ABC, abstractmethod

import numpy as np


class AudioProcessor(ABC):
    """Contract: writable float32 mono (frames, 1) -> same shape and dtype.

    prepare() runs before capture starts: allocate buffers/load resources there.
    process() runs on the PortAudio callback: no blocking, I/O, GUI, or loading.
    Prefer in-place processing and reuse prepared storage. An expensive future
    model must use its own bounded worker bridge, not inference in this callback.
    """

    def prepare(self, sample_rate: int, max_frames: int) -> None:
        pass

    def reset(self) -> None:
        pass

    def stop(self) -> None:
        """Control-thread resource teardown after capture is stopped."""
        pass

    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        raise NotImplementedError
