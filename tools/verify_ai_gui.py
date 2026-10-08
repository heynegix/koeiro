"""Opt-in native Qt AI / three-mode / parameter test; audio goes only to CABLE."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import random

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from src.settings.manager import SettingsManager
from src.processors.chain import ProcessorChain
from tools.verify_audio import VerificationSignal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=int,required=True)
    parser.add_argument('--output',type=int,required=True)
    parser.add_argument('--wav',type=Path,required=True)
    parser.add_argument('--cycles',type=int,default=20)
    parser.add_argument('--prosody',action='store_true')
    parser.add_argument('--text-aware',action='store_true')
    parser.add_argument('--text-strategy',choices=['hybrid','events'],default='hybrid')
    parser.add_argument('--model',choices=['girl_01','anime_cute','anime_soft'],default='girl_01')
    parser.add_argument('--report',type=Path,default=Path('test-results/v03-ai-gui.json'))
    args=parser.parse_args()
    if not 1<=args.cycles<=20:parser.error('cycles must be 1..20 (short trial policy)')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    sys.setswitchinterval(.001)
    app=QApplication([])
    manager=SettingsManager(args.report.with_suffix('.settings.json'))
    window=MainWindow(manager)
    window.show()
    trial_end=time.monotonic()+270  # 5-minute GUI stress incl cleanup
    def pump(predicate=lambda:False,seconds=10,required=True):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            if time.monotonic()>trial_end:
                raise RuntimeError('Short GUI stress time limit reached')
            app.processEvents()
            if predicate():return
            time.sleep(.002)
        if required:raise RuntimeError('Native AI GUI timed out')
    report={'input_kind':'Human WAV replay at Main, not live microphone','cycles':[]}
    try:
        pump(lambda:window.start_button.isEnabled())
        for combo,index in ((window.input_device,args.input),(window.output_device,args.output)):
            combo.setCurrentIndex(next(i for i in range(combo.count()) if combo.itemData(i).index==index))
        if 'cable input' not in window.output_device.currentData().name.lower():
            raise ValueError('Only CABLE Input is allowed')
        window.buffer.setCurrentIndex(window.buffer.findData(256))
        window.rate.setCurrentIndex(window.rate.findData(48000))
        window.prosody_on.setChecked(args.prosody)
        if args.text_aware:
            window.prosody_on.setChecked(True)
            window.prosody_engine.setCurrentIndex(window.prosody_engine.findData('text_v3'))
            window.text_strategy.setCurrentIndex(window.text_strategy.findData(args.text_strategy))
        window.ai_model.setCurrentIndex(window.ai_model.findData(args.model))
        pump(lambda:not window._pending,seconds=30)
        replay=VerificationSignal(window.router,args.wav)
        replay.stop=window.router.stop
        replay.configure_ai=window.router.configure_ai
        replay.restart_ai=window.router.restart_ai
        window.controller.engine.chain=ProcessorChain(replay)
        modes = random.Random(40)
        for cycle in range(args.cycles):
            if args.prosody:
                window.prosody_on.setChecked(True)
            window.mode.setCurrentIndex(0)
            window.mode.setCurrentIndex(2)
            pump(lambda:window.router.ai.bridge.status=='Ready',seconds=30)
            if args.prosody:
                pump(lambda:window.router.ai.bridge.prosody.status=='Ready',seconds=30)
            window.start_button.click()
            pump(lambda:window.stop_button.isEnabled())
            for change in range(10):
                window.mode.setCurrentIndex(modes.choice([i for i in range(3) if i!=window.mode.currentIndex()]))
                window.ai_pitch.setValue((48 if args.model!='girl_01' else 24)+change)
                window.ai_brightness.setValue((cycle*10+change)%70+15)
                window.ai_low_cut.setChecked(change%2==0)
                window.ai_post_fx.setChecked(change%4!=0)
                window.ai_limiter.setChecked(change%3!=0)
                if args.prosody:
                    window.anime_amount.setValue(25+change*7)
                    window.prosody_range.setValue(30+change*5)
                    window.prosody_energy.setValue(20+change*6)
                    window.prosody_preset.setCurrentText(('Natural','Anime Light','Anime Expressive')[change%3])
                    window.prosody_update.setCurrentIndex(change%3)
                if args.text_aware:
                    window.prosody_engine.setCurrentIndex(window.prosody_engine.findData(('rule_v2','text_v3','off')[change%3]))
                pump(seconds=.15,required=False)
            window.mode.setCurrentIndex(2)
            pump(seconds=1,required=False)
            if args.prosody:
                pump(lambda:window.router.ai.bridge.prosody.status=='Running',seconds=10)
            engine=window.controller.engine
            state=window.router.ai.bridge.snapshot()
            report['cycles'].append(dict(ai=state,callback=asdict(engine.performance.snapshot()),
                main=asdict(engine.chain.performance.snapshot()),underflow=engine.underflows,
                overflow=engine.overflows,diagnostics=engine.diagnostics.snapshot(),switches=10))
            window.stop_button.click()
            pump(lambda:window.start_button.isEnabled())
            report['cycles'][-1]['worker_stopped']=not window.router.ai.bridge.alive
            report['cycles'][-1]['prosody_worker_stopped']=not window.router.ai.bridge.prosody.alive
            report['cycles'][-1]['asr_worker_stopped']=not window.router.ai.bridge.prosody.asr.alive
            args.report.with_suffix('.progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'cycle':cycle+1,'ai_underrun':state['ai_underrun'],
                'ai_overrun':state['ai_overrun'],'drop':state['dropped_chunks'],
                'callback_max':report['cycles'][-1]['callback']['maximum_ms']}),flush=True)
            window.ai_quality.setCurrentIndex((cycle+1)%3)
            pump(lambda:not window._pending,seconds=30)
        window.ai_quality.setCurrentIndex(0)
        pump(lambda:not window._pending,seconds=30)
        window.mode.setCurrentIndex(0)
        window.mode.setCurrentIndex(2)
        pump(lambda:window.router.ai.bridge.status=='Ready',seconds=30)
        window.ai_pitch.setValue(32)
        window.start_button.click()
        pump(lambda:window.stop_button.isEnabled())
        pump(seconds=.5,required=False)
        bridge=window.router.ai.bridge
        if args.prosody:
            pump(lambda:bridge.prosody.status=='Running',seconds=10)
            prosody_pid=bridge.prosody.client.process.pid
            bridge.prosody.client.process.terminate()
            pump(lambda:bridge.prosody.status=='Error' and not bridge.prosody.alive and not window.prosody_on.isChecked(),seconds=5)
            report['prosody_crash']=dict(pid=prosody_pid,prosody=bridge.prosody.snapshot(),ai_status=bridge.status,
                audio_running=window.controller.engine.running,effective_mode=window.router._current)
            assert bridge.status=='Ready' and window.router._current=='ai_voice' and window.controller.engine.running
            window.prosody_on.setChecked(True)
            pump(lambda:bridge.prosody.status=='Running',seconds=15)
            report['prosody_crash']['restarted_pid']=bridge.prosody.client.process.pid
            report['prosody_crash']['recovered']=True
        pid=bridge.client.process.pid
        bridge.client.process.terminate()  # only this test owns the worker
        pump(lambda:bridge.status=='Error' and not bridge.alive,seconds=5)
        pump(seconds=.1,required=False)
        report['crash']={'pid':pid,'error':bridge.error,'router_effective':window.router._current,
                         'audio_running':window.controller.engine.running}
        assert window.router._current=='original' and window.controller.engine.running
        window.ai_restart.click()
        pump(lambda:bridge.status=='Ready' and not window._pending,seconds=30)
        pump(seconds=.5,required=False)
        report['crash']['recovered']=window.router._current=='ai_voice'
        report['crash']['new_pid']=bridge.client.process.pid
        report['crash']['ai_stats_after_recovery']=bridge.snapshot()
        report['crash']['callback']=asdict(window.controller.engine.performance.snapshot())
        report['crash']['underflow']=window.controller.engine.underflows
        report['crash']['overflow']=window.controller.engine.overflows
        assert report['crash']['recovered'] and report['crash']['new_pid']!=pid
        window.stop_button.click()
        pump(lambda:window.start_button.isEnabled())
        # Capturing the stopped window cannot stall its audio callback.
        window.tabs.setCurrentIndex(1)
        pump(seconds=.3,required=False)
        window.grab().save(str(args.report.with_suffix('.png')))
        report['passed']=True
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        window.close()
        pump(lambda:not window.controller.alive,seconds=10)
        app.processEvents()
        report['controller_stopped']=not window.controller.alive
        report['worker_stopped']=not window.router.ai.bridge.alive
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(passed=report.get('passed'),cycles=len(report['cycles']),
            worker_stopped=report['worker_stopped'],controller_stopped=report['controller_stopped'])),flush=True)


if __name__=='__main__':main()
