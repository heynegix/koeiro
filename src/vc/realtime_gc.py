"""Collection policy for the disposable, dedicated AI service process only."""
import gc


class RealtimeGC:
    def __enter__(self):
        self.was_enabled = gc.isenabled()
        # Loaded native objects and warmup temporaries can be collected before
        # Ready. Refcount cleanup continues during inference; cyclic collection
        # is postponed until this process stops, never the GUI/audio process.
        gc.collect()
        gc.disable()
        return self

    def __exit__(self, *args):
        if self.was_enabled:
            gc.enable()
        else:
            gc.disable()
