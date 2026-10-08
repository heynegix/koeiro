"""Worker-only subprocess ownership. Binary pipes never touch audio/GUI callbacks."""
from .models import default_voice_id, is_meanvc2
from pathlib import Path
import os
import subprocess

from ..runtime_paths import asset_root, cache_dir, is_frozen, worker_executable
from .protocol import send, receive


def worker_python(root,model=None):
    """Launch CPython directly, retaining the isolated venv without its Windows stub.

    CPython's venv redirector uses the same launcher variable. Owning the actual
    interpreter PID lets Stop terminate hung native inference without orphaning it.
    In a packaged build the worker is a separate frozen executable, so the same
    ownership and termination behaviour survives without an installed Python.
    """
    if is_frozen():
        executable = worker_executable()
        if not executable.is_file():
            raise RuntimeError('同梱のAIワーカーが見つかりません。フォルダを再インストールしてください。')
        return executable, os.environ.copy()
    environment_root=root/'vc_models/meanvc2/.venv' if (model is None or is_meanvc2(model)) else root/'.venv-ai'
    launcher = environment_root/'Scripts/python.exe'
    if not launcher.is_file():
        raise RuntimeError('AI環境がありません。READMEのAIセットアップを実行してください。')
    environment = os.environ.copy()
    for key in ('PYTHONHOME', 'PYTHONPATH', '__PYVENV_LAUNCHER__'):
        environment.pop(key, None)
    executable = launcher
    if os.name == 'nt':
        config = (environment_root/'pyvenv.cfg').read_text(encoding='utf-8')
        values = {key.strip(): value.strip() for key, value in
                  (line.split('=', 1) for line in config.splitlines() if '=' in line)}
        home = values.get('home', '').strip()
        if not home:
            raise RuntimeError('AI Pythonのhome設定がありません。AI環境を作り直してください。')
        executable = Path(home)/'python.exe'
        if not executable.is_file():
            raise RuntimeError('AI Python本体がありません。AI環境を作り直してください。')
        environment['__PYVENV_LAUNCHER__'] = str(launcher)
    return executable, environment


class ServiceClient:
    def __init__(self, parameters):
        self.parameters = parameters
        self.process = None
        self.log_file = None

    def start(self):
        from .config import QUALITY_FACTORS
        root = asset_root()
        executable, environment = worker_python(root,self.parameters.model)
        if is_meanvc2(self.parameters.model):
            cache=root/'vc_models/cache'
            for key,folder in {'TEMP':'temp','TMP':'temp','HF_HOME':'hf-home','HF_HUB_CACHE':'hf','TORCH_HOME':'torch','MPLCONFIGDIR':'matplotlib','NUMBA_CACHE_DIR':'numba'}.items():
                target=cache/folder
                if is_frozen():
                    # Never write into the install directory; it may be read-only.
                    from ..runtime_paths import cache_dir as writable_cache
                    target=writable_cache()/folder
                    target.mkdir(parents=True,exist_ok=True)
                environment[key]=str(target)
            environment.update(OMP_NUM_THREADS=str(self.parameters.threads),MKL_NUM_THREADS=str(self.parameters.threads),
                               CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1')
        (root/'logs').mkdir(exist_ok=True)
        self.log_file = (root/'logs/ai-worker.log').open('ab', buffering=0)
        command = [str(executable), '-u', '-m', 'src.vc.service',
            '--factor', str(QUALITY_FACTORS[self.parameters.quality]), '--threads', str(self.parameters.threads),
            '--model',self.parameters.model,'--delivery',self.parameters.delivery,'--enhancer',self.parameters.enhancer,
            '--experiment',self.parameters.experiment,
            '--tune-sib-db',str(self.parameters.tune_sib_db),
            '--tune-cons-db',str(self.parameters.tune_cons_db),
            '--tune-caps',str(self.parameters.tune_caps),
            '--tune-floor-db',str(self.parameters.tune_floor_db),
            '--tune-excess-db',str(self.parameters.tune_excess_db),
            '--tune-mid',str(self.parameters.tune_mid),
            '--tune-match',str(self.parameters.tune_match),
            '--tune-ptrans',str(self.parameters.tune_ptrans),
            '--tune-pcap',str(self.parameters.tune_pcap),
            '--tune-level-db',str(self.parameters.tune_level_db)] + (
            ['--tune-combined'] if self.parameters.tune_combined else ['--no-tune-combined'])
        if self.parameters.lavasr_denoise:
            command.append('--lavasr-denoise')
        self.process = subprocess.Popen(command,
            cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_file,
            env=environment,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def read(self):
        header, audio = receive(self.process.stdout)
        if header['status'] == 'Error':
            raise RuntimeError(header.get('error', 'AI worker error'))
        return header, audio

    def request(self, header, audio=None):
        send(self.process.stdin, header, audio)
        return self.read()

    def interrupt(self):
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()

    def close(self):
        process, log_file = self.process, self.log_file
        self.process = self.log_file = None
        try:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            # Buffered stdin can fail while flushing after a worker crash. Every
            # owned handle must still close so repeated recovery cannot leak it.
            for stream in (getattr(process, 'stdin', None),
                           getattr(process, 'stdout', None), log_file):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
