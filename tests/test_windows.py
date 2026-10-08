import pytest

from src.utils.windows import com_apartment


class FakeCom:
    def __init__(self, result):
        self.result = result
        self.closed = 0

    def CoInitializeEx(self, reserved, mode):
        assert reserved is None and mode == 2
        return self.result

    def CoUninitialize(self):
        self.closed += 1


@pytest.mark.parametrize("result", [0, 1])
def test_com_is_balanced_on_worker_exception(result):
    com = FakeCom(result)
    with pytest.raises(ValueError):
        with com_apartment(com):
            raise ValueError("Worker failed")
    assert com.closed == 1


def test_com_failure_does_not_uninitialize():
    com = FakeCom(-2147417850)
    with pytest.raises(OSError, match="HRESULT"):
        with com_apartment(com):
            pytest.fail("Failed COM must not enter worker")
    assert com.closed == 0
