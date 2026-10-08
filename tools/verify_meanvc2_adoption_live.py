"""One three-minute physical audio trial, bounded to five minutes including setup."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.compare_meanvc2_continuity import read,save,require_idle_voice_worker
from tools.vc_tournament.process import run


def main():
    import psutil
    require_idle_voice_worker()
    if not read(ROOT/'validation/v011/meanvc2_adoption_validation.json')['passed']:
        raise RuntimeError('Production replay validation must pass first')
    if psutil.virtual_memory().available<4*1024**3:
        raise RuntimeError('Need at least 4 GiB free RAM')
    os.environ['QT_QPA_PLATFORM']='offscreen'
    report=ROOT/'validation/v011/meanvc2_adoption_live.json'
    args=['tools/verify_meanvc2_live.py','--model','meanvc2_ref60',
          '--input','68','--output','58','--cable-return','74','--seconds','180','--report',str(report)]
    bootstrap='import psutil,runpy,sys; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS); sys.argv='+repr(args)+'; runpy.run_path(sys.argv[0],run_name="__main__")'
    process=run([Path(os.environ['LOCALAPPDATA'])/'Programs/Python/Python313/python.exe','-c',bootstrap],ROOT,
                report.with_suffix('.log'),timeout=300,ram_limit_gb=4.5)
    save(report.with_name(report.stem+'_process.json'),process)
    result=read(report) if report.exists() else {}
    passed=process['status']=='SUCCESS' and result.get('passed') and result.get('worker_stopped') and result.get('controller_stopped')
    print(dict(passed=passed,process=process['status'],elapsed_wall_seconds=result.get('elapsed_wall_seconds')))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main())
