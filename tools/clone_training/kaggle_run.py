"""Kaggle orchestration around the unmodified official Beatrice trainer."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import zipfile

REVISION = 'f34836de014b86956096878aecb8d3b17feaaa0b'
ROOT = Path('/kaggle/working')
OUTPUT = ROOT / 'beatrice_clone_v1'


def main():
    OUTPUT.mkdir(exist_ok=True)
    input_root = Path('/kaggle/input')
    # Resolve each mount explicitly: Path.rglob does not follow directory symlinks.
    mounts = [input_root] + [p.resolve() for p in input_root.iterdir() if p.is_dir()]
    manifests = {p.resolve() for mount in mounts for p in mount.rglob('accepted_manifest.json')}
    archives = {p.resolve() for mount in mounts for p in mount.rglob('anime_voice_clone_v1_kaggle.zip')}
    print('Input mounts:', [str(p) for p in input_root.iterdir()], flush=True)
    if not manifests and not archives:
        raise RuntimeError('Accepted dataset not mounted; attach private dataset before execution')
    from huggingface_hub import snapshot_download
    repo = Path(snapshot_download('fierce-cats/beatrice-trainer', revision=REVISION))
    # The upstream module loads assets relative to its repository root.
    local_repo = ROOT / 'beatrice-trainer'
    if not local_repo.exists():
        shutil.copytree(repo, local_repo)
    # Official repo_root() looks for .git; a Hub snapshot omits Git metadata.
    # Initialize this copied upstream checkout without modifying trainer code.
    subprocess.run(['git', 'init', str(local_repo)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(local_repo), '--no-deps'], check=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU REQUIRED: CPU training is forbidden')
    environment = {'torch':torch.__version__, 'trainer_revision':REVISION,
                   'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   'training_device':0, 'note':'Official trainer uses one GPU; second T4 is not distributed.'}
    (OUTPUT / 'environment.json').write_text(json.dumps(environment, indent=2))
    data = ROOT / 'clone_input'
    data.mkdir(exist_ok=True)
    if archives:
        with zipfile.ZipFile(sorted(archives)[0]) as z:
            for name in z.namelist():
                if not (data / name).resolve().is_relative_to(data.resolve()):
                    raise ValueError('Unsafe archive member')
            z.extractall(data)
    else:
        # Kaggle may unpack uploaded ZIPs into the Input filesystem.
        if len(manifests) != 1:
            raise RuntimeError('Expected exactly one accepted dataset input')
        shutil.copytree(next(iter(manifests)).parent, data, dirs_exist_ok=True)
    manifest = json.loads((data / 'accepted_manifest.json').read_text())
    files = sorted((data / 'train' / 'own_female').glob('*.wav'))
    if len(files) != 67 or len(manifest) != 67:
        raise RuntimeError('Accepted dataset must contain all 67 WAVs')
    for item in manifest:
        if hashlib.sha256((data / item['file']).read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Training input checksum mismatch')
    config = json.loads((local_repo / 'assets/default_config.json').read_text())
    config.update(batch_size=8, learning_rate_g=5e-5, learning_rate_d=5e-5,
                  num_workers=2, record_metrics=True, save_interval=500,
                  evaluation_interval=500, in_test_wav_dir=str(data / 'evaluation'))
    train_out = OUTPUT / 'training'
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            with (OUTPUT / 'gpu_metrics.csv').open('a') as f:
                f.write(result.stdout)
            stop.wait(30)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    results = []
    try:
        for index, step in enumerate((1000, 2500, 5000)):
            config['n_steps'] = step
            config_path = OUTPUT / f'config_{step}.json'
            config_path.write_text(json.dumps(config, indent=2))
            command = [sys.executable, 'beatrice_trainer', '-d', str(data / 'train'),
                       '-o', str(train_out), '-c', str(config_path)]
            if index:
                command.append('-r')
            started = time.time()
            with (OUTPUT / f'training_{step}.log').open('w') as log:
                process = subprocess.Popen(command, cwd=local_repo, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'PYTHONUNBUFFERED':'1'})
                while process.poll() is None:
                    time.sleep(15)
                    checkpoints = sorted(train_out.glob('checkpoint_train_*.pt.gz'))
                    current = checkpoints[-1].name if checkpoints else 'initializing'
                    print(f'Target={step}, latest durable checkpoint={current}', flush=True)
                    log.flush()
                    print((OUTPUT / f'training_{step}.log').read_text(errors='replace')[-1800:], flush=True)
                if process.returncode:
                    tail = (OUTPUT / f'training_{step}.log').read_text(errors='replace')[-6000:]
                    print(tail, flush=True)
                    raise RuntimeError(f'Official trainer failed with code {process.returncode}; see log')
            checkpoint = train_out / f'checkpoint_train_{step:08d}.pt.gz'
            export = train_out / f'paraphernalia_train_{step:08d}'
            if not checkpoint.is_file() or not export.is_dir():
                raise RuntimeError(f'Missing checkpoint/export at {step}')
            # Confirm integrity, step number and all model tensors before archive.
            with gzip.open(checkpoint, 'rb') as f:
                state = torch.load(f, map_location='cpu', weights_only=True)
            if state['iteration'] != step:
                raise RuntimeError('Wrong checkpoint iteration')
            for group in ('net_g', 'net_d', 'phone_extractor', 'pitch_estimator'):
                for name, tensor in state[group].items():
                    if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                        raise RuntimeError(f'NaN/Inf in {group}.{name}')
            del state
            stage = OUTPUT / f'clone_{step}'
            stage.mkdir(exist_ok=True)
            shutil.copy2(checkpoint, stage / checkpoint.name)
            shutil.copytree(export, stage / export.name, dirs_exist_ok=True)
            shutil.copy2(config_path, stage / 'config.json')
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            accumulator = EventAccumulator(str(train_out), size_guidance={'scalars':0}).Reload()
            metrics = {tag:[{'step':v.step,'value':v.value,'wall_time':v.wall_time}
                            for v in accumulator.Scalars(tag)] for tag in accumulator.Tags()['scalars']}
            (stage / 'metrics.json').write_text(json.dumps(metrics))
            results.append({'step':step,'seconds':time.time()-started,'checkpoint':checkpoint.name,
                            'finite_tensors':True,'resume':bool(index)})
            (OUTPUT / 'training_summary.json').write_text(json.dumps(results, indent=2))
            print('STAGE COMPLETE', results[-1], flush=True)
    finally:
        stop.set()
        thread.join(timeout=5)


if __name__ == '__main__':
    main()
