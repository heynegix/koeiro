"""Print the packaged layout a frozen build resolves to (diagnostic only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import runtime_paths as rp

print('frozen      :', rp.is_frozen())
print('executable  :', sys.executable)
print('meipass     :', getattr(sys, '_MEIPASS', None))
print('asset_root  :', rp.asset_root())
print('install_root:', rp.install_root())
print('worker_env  :', rp.worker_environment())
print('worker_exe  :', rp.worker_executable())
