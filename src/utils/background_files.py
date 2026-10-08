"""Coalescing bounded file jobs. GUI publishes data; disk work stays here."""
from collections import OrderedDict
import logging
from threading import Condition, Thread

log = logging.getLogger(__name__)


class BackgroundFiles:
    def __init__(self):
        self._condition = Condition()
        self._pending = OrderedDict()
        self._closing = False
        self.error = ''
        self.result = ''
        # Per-job messages the GUI drains in its own tick, so a worker thread never
        # touches a widget. Keys are the submit() keys.
        self.results = {}
        self._thread = Thread(target=self._run, name='background-files', daemon=True)
        self._thread.start()

    @property
    def alive(self):
        return self._thread.is_alive()

    def submit(self, key, action):
        with self._condition:
            if self._closing:
                return False
            if key not in self._pending and len(self._pending) >= 2:
                return False
            self._pending[key] = action
            self._condition.notify()
            return True

    def close(self):
        with self._condition:
            self._closing = True
            self._condition.notify()

    def wait_closed(self):
        """Control thread only: finish final save before publishing shutdown."""
        self._thread.join(timeout=3)
        if self.alive:
            raise RuntimeError('Background file operation did not finish')

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closing)
                if not self._pending:
                    return
                key, action = self._pending.popitem(last=False)
            try:
                result = action()
                if result is False:
                    raise OSError('File operation failed')
                self.result = str(result or '')
                self.error = ''
                self.results[key] = self.result
            except Exception as error:
                self.error = str(error)
                self.results[key] = 'ERROR: '+str(error)
                log.exception('Background file operation failed')
