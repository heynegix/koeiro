import sys
import threading
import time
import pytest
from src.vc.wakeup import InputWakeup


@pytest.mark.skipif(sys.platform!='win32',reason='Windows x64 input notification')
def test_auto_reset_notification_retains_signal_and_closes():
    wake=InputWakeup()
    wake.open()
    try:
        wake.signal()
        assert wake.kernel.WaitForSingleObject(wake.handle,0)==0
        assert wake.kernel.WaitForSingleObject(wake.handle,0)==258
        handle=wake.handle
        wake.open()
        assert wake.handle==handle
    finally:wake.close()
    assert wake.handle is None
    wake.signal();wake.close()


@pytest.mark.skipif(sys.platform!='win32',reason='Windows input wakeup')
def test_worker_wait_wakes_from_callback_notification():
    wake=InputWakeup();wake.open()
    stopped=threading.Event();arrived=threading.Event()
    def worker():
        wake.wait(stopped)
        arrived.set()
    thread=threading.Thread(target=worker)
    try:
        thread.start()
        wake.signal()
        assert arrived.wait(1)
        thread.join(timeout=1)
        assert not thread.is_alive()
    finally:
        wake.signal();thread.join(timeout=1);wake.close()
