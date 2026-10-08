from pathlib import Path
import subprocess
from src.vc.client import ServiceClient, worker_python


class ProsodyClient(ServiceClient):
    def start(self):
        root=Path(__file__).resolve().parents[2]
        executable,environment=worker_python(root)  # lightweight NumPy/SciPy, no torch
        (root/'logs').mkdir(exist_ok=True)
        self.log_file=(root/'logs/prosody-worker.log').open('ab',buffering=0)
        self.process=subprocess.Popen([str(executable),'-u','-m','src.prosody.service'],cwd=root,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log_file,env=environment,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
