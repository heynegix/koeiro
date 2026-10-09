"""Worker-only subprocess ownership. Binary pipes never touch audio/GUI callbacks."""
from .models import default_voice_id, is_meanvc2
from pathlib import Path
import os
import subprocess

from ..runtime_paths import (asset_root, cache_dir, data_dir, install_root,
                             is_bundled_worker, is_frozen, worker_environment,
                             worker_executable)
from .protocol import send, receive


def _interpreter(environment_root):
    """The real interpreter and environment for one worker virtualenv.

    CPython's venv redirector uses the same launcher variable, so the interpreter is
    read out of ``pyvenv.cfg`` and ``__PYVENV_LAUNCHER__`` is set back to the stub.
    """
    launcher = environment_root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not launcher.is_file():
        raise RuntimeError('AI環境がありません。tools/setup_release.py を実行するか、'
                           'READMEのAIセットアップに従ってください。')
    environment = os.environ.copy()
    for key in ('PYTHONHOME', 'PYTHONPATH', '__PYVENV_LAUNCHER__'):
        environment.pop(key, None)
    executable = launcher
    if os.name == 'nt':
        config = (environment_root / 'pyvenv.cfg').read_text(encoding='utf-8')
        values = {key.strip(): value.strip() for key, value in
                  (line.split('=', 1) for line in config.splitlines() if '=' in line)}
        home = values.get('home', '').strip()
        if not home:
            raise RuntimeError('AI Pythonのhome設定がありません。AI環境を作り直してください。')
        executable = Path(home) / 'python.exe'
        if not executable.is_file():
            raise RuntimeError('AI Python本体がありません。AI環境を作り直してください。')
        environment['__PYVENV_LAUNCHER__'] = str(launcher)
    return executable, environment


def worker_python(root, model=None):
    """The worker command's interpreter and environment.

    A packaged build looks for the fully frozen worker next to itself first. That is
    not built for release -- a torch-bundling exe cannot fit the hosting limit -- so
    it falls back to the environment ``tools/setup_release.py`` installed beside the
    executable. Refusing that fallback is what made a packaged GUI report a missing
    AI worker while a working environment sat in its own folder.
    """
    if is_frozen():
        bundle = worker_executable()
        if bundle is not None:
            # A frozen worker is not Python: it takes the service arguments directly,
            # so the module invocation is dropped by ServiceClient.start().
            return bundle, os.environ.copy()
        environment_root = worker_environment()
        if environment_root is None:
            raise RuntimeError('AI環境が見つかりません。tools/setup_release.py を実行してください。')
        return _interpreter(environment_root)
    environment_root = (root / 'vc_models/meanvc2/.venv'
                        if (model is None or is_meanvc2(model)) else root / '.venv-ai')
    return _interpreter(environment_root)


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
                           PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1')
        # Hide GPUs only when CPU inference was explicitly requested. 'auto'
        # keeps them visible so the worker can use CUDA when available.
        if self.parameters.device == 'cpu':
            environment['CUDA_VISIBLE_DEVICES'] = ''
        # The worker's stderr goes to a log the reader can hand over. It belongs with
        # the app's own log rather than in the bundle: a packaged build is read-only
        # (Program Files, or the unpacked _internal tree), where opening this file
        # failed with PermissionError before the worker was ever launched.
        log_dir = data_dir() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (log_dir / 'ai-worker.log').open('ab', buffering=0)
        command = [str(executable)] + ([] if is_bundled_worker(executable) else ['-u', '-m', 'src.vc.service']) + [
            '--factor', str(QUALITY_FACTORS[self.parameters.quality]), '--threads', str(self.parameters.threads),
            '--device', self.parameters.device,
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
            cwd=install_root(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log_file,
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
