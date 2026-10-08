"""Isolated process lifecycle smoke, no audio hardware."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.prosody.bridge import ProsodyBridge
from src.prosody.parameters import ProsodyParameters

def main():
    bridge=ProsodyBridge(ProsodyParameters(enabled=True))
    bridge.load()
    try:
        until=time.monotonic()+15
        while bridge.status=='Starting' and time.monotonic()<until:
            time.sleep(.02)
        print(bridge.status,bridge.error,flush=True)
        assert bridge.status=='Ready'
        bridge.set_active(True)
        time.sleep(.03)
        for i in range(10):
            bridge.submit((.1*np.sin(2*np.pi*150*(np.arange(1440)+1440*i)/48000)).astype(np.float32))
            time.sleep(.030)
        state=bridge.snapshot()
        print(json.dumps(state),flush=True)
        assert state['status']=='Running' and state['control']['f0']>0
    finally:
        bridge.stop()
    assert not bridge.alive

if __name__=='__main__': main()
