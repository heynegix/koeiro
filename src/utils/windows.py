from contextlib import contextmanager
import ctypes
import sys


class _PowerState(ctypes.Structure):
    _fields_=[('Version',ctypes.c_uint32),('ControlMask',ctypes.c_uint32),('StateMask',ctypes.c_uint32)]


@contextmanager
def audio_process_policy(kernel=None):
    """Process-local HighQoS/timer policy, restored on worker exit.

    Optional on older Windows. Never changes system power plans or priorities.
    Query first so settings owned by another component are preserved.
    """
    info={'high_qos':False,'honor_timer_resolution':False}
    changed=False;previous=None;handle=None
    try:
        if kernel is None and sys.platform=='win32':
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.GetCurrentProcess.restype=ctypes.c_void_p
            for name in ('GetProcessInformation','SetProcessInformation'):
                fn=getattr(kernel,name)
                fn.argtypes=(ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32)
                fn.restype=ctypes.c_int
        if kernel is not None:
            handle=kernel.GetCurrentProcess();previous=_PowerState(1,0,0)
            if kernel.GetProcessInformation(handle,4,ctypes.byref(previous),ctypes.sizeof(previous)):
                desired=_PowerState(1,previous.ControlMask|5,previous.StateMask&~5)
                changed=bool(kernel.SetProcessInformation(handle,4,ctypes.byref(desired),ctypes.sizeof(desired)))
                info.update(high_qos=changed,honor_timer_resolution=changed)
            if not changed:info['unsupported_or_error']=True
    except (OSError,AttributeError) as error:
        info['error']=str(error)
    try:
        yield info
    finally:
        if changed:
            try:
                info['restored']=bool(kernel.SetProcessInformation(handle,4,ctypes.byref(previous),ctypes.sizeof(previous)))
            except (OSError, AttributeError) as error:
                info['restore_error']=str(error)


@contextmanager
def audio_scheduling(avrt=None, winmm=None, *, register_mmcss=True):
    """Opt this worker into Windows' audio task scheduling; always release it.

    Timer resolution is scoped to this process on supported Windows versions.
    Registration failure is diagnostic information, never an audio startup error.
    No registry, power plan, or system-wide priority settings are changed.
    """
    info = {'mmcss': False, 'timer_1ms': False}
    handle = None
    if sys.platform != 'win32' and avrt is None and winmm is None:
        yield info
        return
    try:
        if not register_mmcss:
            avrt = None
        elif avrt is None:
            avrt = ctypes.WinDLL('avrt', use_last_error=True)
            avrt.AvSetMmThreadCharacteristicsW.argtypes = (ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32))
            avrt.AvSetMmThreadCharacteristicsW.restype = ctypes.c_void_p
            avrt.AvRevertMmThreadCharacteristics.argtypes = (ctypes.c_void_p,)
            avrt.AvRevertMmThreadCharacteristics.restype = ctypes.c_int
        task_index = ctypes.c_uint32(0)
        handle = avrt.AvSetMmThreadCharacteristicsW('Pro Audio', ctypes.byref(task_index)) if register_mmcss else None
        info['mmcss'] = bool(handle)
    except OSError as error:
        info['mmcss_error'] = str(error)
    try:
        if winmm is None:
            winmm = ctypes.WinDLL('winmm')
            winmm.timeBeginPeriod.argtypes = (ctypes.c_uint,)
            winmm.timeBeginPeriod.restype = ctypes.c_uint
            winmm.timeEndPeriod.argtypes = (ctypes.c_uint,)
            winmm.timeEndPeriod.restype = ctypes.c_uint
        info['timer_1ms'] = winmm.timeBeginPeriod(1) == 0
    except OSError as error:
        info['timer_error'] = str(error)
    try:
        yield info
    finally:
        if info['timer_1ms']:
            winmm.timeEndPeriod(1)
        if handle:
            avrt.AvRevertMmThreadCharacteristics(handle)


@contextmanager
def com_apartment(ole32=None):
    """Initialize COM on the thread opening WASAPI streams; balance cleanup.

    PortAudio's import-time initialization on the main thread does not initialize
    a Python worker. WASAPI activation/marshalling also needs a COM apartment on
    that worker. STA matches PortAudio's Windows initialization convention.
    """
    if sys.platform != "win32" and ole32 is None:
        yield
        return
    if ole32 is None:
        ole32 = ctypes.WinDLL("ole32")
        ole32.CoInitializeEx.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        ole32.CoInitializeEx.restype = ctypes.c_long
        ole32.CoUninitialize.argtypes = ()
        ole32.CoUninitialize.restype = None
    result = ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED
    if result not in (0, 1):  # S_OK / S_FALSE both require CoUninitialize.
        raise OSError(f"Audio control COM initialization failed: HRESULT 0x{result & 0xFFFFFFFF:08X}")
    try:
        yield
    finally:
        ole32.CoUninitialize()
