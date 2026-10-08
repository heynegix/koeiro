"""Bounded child processes with sampled process-tree metrics and cleanup."""
import os
import subprocess
import time
from pathlib import Path
import psutil

def run(command, cwd, log, timeout=600, ram_limit_gb=12):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONPATH=str(cwd),
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               HF_HOME=str(Path(__file__).resolve().parents[2]/'vc_models/cache/hf-home'),
               HF_HUB_CACHE=str(Path(__file__).resolve().parents[2]/'vc_models/cache/hf'),
               TORCH_HOME=str(Path(__file__).resolve().parents[2]/'vc_models/cache/torch'),
               HF_HUB_DISABLE_XET='1', HF_HUB_DOWNLOAD_TIMEOUT='60', UV_LINK_MODE='hardlink')
    cache = Path(__file__).resolve().parents[2] / 'vc_models/cache'
    if os.name == 'nt' and cache.resolve().drive.upper() != 'D:':
        raise RuntimeError('Tournament cache must physically reside on D:')
    for variable, directory in {
        'UV_CACHE_DIR':'uv', 'UV_PYTHON_INSTALL_DIR':'python', 'TEMP':'temp', 'TMP':'temp',
        'WANDB_DIR':'wandb', 'WANDB_CACHE_DIR':'wandb', 'MPLCONFIGDIR':'matplotlib',
        'NUMBA_CACHE_DIR':'numba', 'XDG_CACHE_HOME':'xdg',
        'PIP_CACHE_DIR':'pip', 'NLTK_DATA':'nltk', 'MODELSCOPE_CACHE':'modelscope',
        'HF_ASSETS_CACHE':'hf-assets', 'WANDB_CONFIG_DIR':'wandb-config',
    }.items():
        target = cache / directory
        target.mkdir(parents=True, exist_ok=True)
        env[variable] = str(target)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); peak=0; main_peak=0; cpu_max=0; cpu_total=0; samples=0
    seen={}; reason=None; last_rss=None; disk_min=None
    with open(log,'w',encoding='utf-8') as f:
        f.write('ARGV: '+repr([str(c) for c in command])+'\n'); f.flush()
        p=subprocess.Popen([str(c) for c in command],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        parent=psutil.Process(p.pid)
        while p.poll() is None:
            storage_paths=[Path(__file__).resolve().parents[2],Path(cwd).resolve()]
            cache=storage_paths[0]/'vc_models/cache'
            if cache.exists():storage_paths.append(cache.resolve())
            free=min(psutil.disk_usage(str(path)).free for path in storage_paths)
            disk_min=free if disk_min is None else min(disk_min,free)
            try:
                procs=[parent]+parent.children(recursive=True)
                rss=0; cp=0
                for proc in procs:
                    try:
                        if proc.pid not in seen:
                            proc.cpu_percent(None);seen[proc.pid]=proc
                        rss+=proc.memory_info().rss
                        # Keep the sampling instance: fresh psutil.Process
                        # objects have no previous CPU-time sample.
                        cp+=seen[proc.pid].cpu_percent(None)
                    except psutil.Error: pass
                main_rss=parent.memory_info().rss
                main_peak=max(main_peak,main_rss); last_rss=main_rss
                peak=max(peak,rss);cpu_max=max(cpu_max,cp);cpu_total+=cp;samples+=1
            except psutil.Error: pass
            if time.perf_counter()-started > timeout:
                reason='TIMEOUT'
            elif peak>ram_limit_gb*1024**3:
                reason='RAM_LIMIT'
            elif free<2.5*1024**3:
                reason='DISK_LIMIT_2_5_GIB_RESERVE'
            if reason:
                try:
                    children=parent.children(recursive=True)
                    for child in reversed(children):
                        try: child.kill()
                        except psutil.Error: pass
                    parent.kill()
                except psutil.Error: pass
                break
            time.sleep(.2)
        p.wait()
    return dict(status='SUCCESS' if p.returncode==0 and not reason else 'FAILED',
                returncode=p.returncode, reason=reason, wall_seconds=time.perf_counter()-started,
                peak_ram_bytes=peak, process_peak_ram_bytes=main_peak, process_ram_bytes=last_rss,
                cpu_percent_mean=cpu_total/max(1,samples), cpu_percent_peak=cpu_max,
                cpu_measurement_schema=2,
                cpu_percent_definition='psutil persistent process-tree samples; 100% = one core; 200ms sampling',
                sampling_interval_seconds=.2, log=str(log),
                workspace_min_free_bytes=disk_min,
                last_error=Path(log).read_text(encoding='utf-8',errors='replace')[-4000:] if p.returncode or reason else None)
