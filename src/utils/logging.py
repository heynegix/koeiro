from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import atexit
import logging
from pathlib import Path
import queue
import sys
import threading

_listener = None


class BoundedLogHandler(QueueHandler):
    """Log bursts never wait for disk or grow an unbounded backlog."""
    dropped = 0

    def enqueue(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1


def shutdown_logging():
    global _listener
    if _listener is not None:
        listener, _listener = _listener, None
        # Shutdown is outside capture. Make room for QueueListener's sentinel.
        listener.queue.join()
        listener.stop()
        for handler in listener.handlers:
            handler.close()


atexit.register(shutdown_logging)


def configure_logging(directory: Path):
    shutdown_logging()
    directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(directory / "app.log", maxBytes=2_000_000,
                                  backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    pending = queue.Queue(maxsize=1024)
    logging.basicConfig(level=logging.INFO, handlers=[BoundedLogHandler(pending)],
                        format="%(message)s", force=True)
    global _listener
    _listener = QueueListener(pending, handler)
    _listener.start()

    def exception_hook(kind, value, traceback):
        logging.getLogger("app").critical("Unhandled exception", exc_info=(kind, value, traceback))
        sys.__excepthook__(kind, value, traceback)

    sys.excepthook = exception_hook
    threading.excepthook = lambda args: exception_hook(args.exc_type, args.exc_value, args.exc_traceback)
