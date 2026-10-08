"""Windows auto-reset notification: callback signals, only the worker waits."""
import ctypes
import sys


class InputWakeup:
    def __init__(self):
        self.handle=None
        self.kernel=None
        if sys.platform=='win32' and ctypes.sizeof(ctypes.c_void_p)==8:
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.CreateEventW.argtypes=(ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_wchar_p)
            kernel.CreateEventW.restype=ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes=(ctypes.c_void_p,ctypes.c_uint32)
            kernel.WaitForSingleObject.restype=ctypes.c_uint32
            kernel.CloseHandle.argtypes=(ctypes.c_void_p,)
            kernel.CloseHandle.restype=ctypes.c_int
            # Windows x64 uses a unified calling convention. Retain the GIL for
            # this short non-waiting signal, avoiding a callback GIL reacquire.
            signal=ctypes.PyDLL('kernel32',use_last_error=True).SetEvent
            signal.argtypes=(ctypes.c_void_p,)
            signal.restype=ctypes.c_int
            self.kernel,self._signal=kernel,signal

    def open(self):
        if self.kernel is not None and self.handle is None:
            self.handle=self.kernel.CreateEventW(None,False,False,None)
            if not self.handle:raise ctypes.WinError(ctypes.get_last_error())

    def signal(self):
        if self.handle is not None:self._signal(self.handle)

    def wait(self,stop):
        if self.handle is None:
            stop.wait(.001)
        elif self.kernel.WaitForSingleObject(self.handle,20)==0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle=None
