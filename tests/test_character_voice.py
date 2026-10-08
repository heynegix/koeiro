import ctypes
import pytest

from src.vc.models import DEFAULT_VOICE_ID, profile
from src.utils.windows import audio_process_policy, _PowerState


def test_profile_copy_does_not_mutate_the_shared_table():
    value = profile(DEFAULT_VOICE_ID)
    value['voice'] = 99
    assert profile(DEFAULT_VOICE_ID)['voice'] == 0


@pytest.mark.parametrize('name', ['../model', '', 'unknown'])
def test_profile_disallows_arbitrary_model_paths(name):
    with pytest.raises(ValueError):
        profile(name)


class FakeKernel:
    def __init__(self, supported=True, restore_error=False):
        self.supported = supported
        self.restore_error = restore_error
        self.writes = []

    def GetCurrentProcess(self):
        return 1

    def GetProcessInformation(self, handle, kind, pointer, size):
        state = ctypes.cast(pointer, ctypes.POINTER(_PowerState)).contents
        state.ControlMask = 7
        state.StateMask = 7
        return self.supported

    def SetProcessInformation(self, handle, kind, pointer, size):
        state = ctypes.cast(pointer, ctypes.POINTER(_PowerState)).contents
        self.writes.append((state.ControlMask, state.StateMask))
        if self.restore_error and len(self.writes) == 2:
            raise OSError('restore unavailable')
        return True


def test_process_policy_preserves_other_bits_and_restores_on_exception():
    kernel = FakeKernel()
    with pytest.raises(ValueError):
        with audio_process_policy(kernel) as info:
            assert info['high_qos']
            raise ValueError('worker stopped')
    assert kernel.writes == [(7,2),(7,7)]
    assert info['restored']


def test_unsupported_process_policy_is_nonfatal_and_does_not_write():
    kernel = FakeKernel(False)
    with audio_process_policy(kernel) as info:
        assert not info['high_qos']
    assert not kernel.writes


def test_process_policy_restore_failure_is_diagnostic():
    with audio_process_policy(FakeKernel(restore_error=True)) as info:
        pass
    assert info['restore_error'] == 'restore unavailable'



