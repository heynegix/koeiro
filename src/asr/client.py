from pathlib import Path
import subprocess
from src.vc.client import ServiceClient
from .protocol import send,receive


class ASRClient(ServiceClient):
    def read(self):
        header,audio=receive(self.process.stdout)
        if header.get('status')=='Error': raise RuntimeError(header.get('error','ASR worker error'))
        return header,audio

    def request(self,header,audio=None):
        send(self.process.stdin,header,audio); return self.read()

    def start(self):
        root=Path(__file__).resolve().parents[2]
        # Separate dependencies and process. No network or implicit download.
        launcher=root/'.venv-asr/Scripts/python.exe'
        if not launcher.is_file(): raise RuntimeError('Local ASR environment missing')
        import os
        environment=os.environ.copy()
        for k in ('PYTHONHOME','PYTHONPATH','__PYVENV_LAUNCHER__'): environment.pop(k,None)
        config=(root/'.venv-asr/pyvenv.cfg').read_text(encoding='utf-8')
        values={k.strip():v.strip() for k,v in (line.split('=',1) for line in config.splitlines() if '=' in line)}
        executable=Path(values['home'])/'python.exe' if 'home' in values else launcher
        if os.name=='nt': environment['__PYVENV_LAUNCHER__']=str(launcher)
        (root/'logs').mkdir(exist_ok=True)
        self.log_file=(root/'logs/asr-worker.log').open('ab',buffering=0)
        self.process=subprocess.Popen([str(executable),'-u','-m','src.asr.service',
            '--model',str(root/'models/asr-reazon'),'--threads',str(self.parameters.asr_threads)],
            cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log_file,
            env=environment,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
