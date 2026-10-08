"""Offline gate retention audit. Energy is not a speech/intelligibility label."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.processors.noise_gate import NoiseGateProcessor
from tools.compare_presets import read_wav,write_wav


def audit(audio,threshold,rate=48000,block=256):
    gate=NoiseGateProcessor(threshold);gate.prepare(rate,block)
    result=np.empty_like(audio);ratios=[];energy=[]
    for start in range(0,len(audio),block):
        frame=audio[start:start+block].copy().reshape(-1,1)
        before=float(np.mean(frame.astype(np.float64)**2))
        gate.process(frame,rate)
        after=float(np.mean(frame.astype(np.float64)**2))
        result[start:start+len(frame)]=frame[:,0]
        ratios.append(np.sqrt(after/max(before,1e-15)));energy.append(np.sqrt(before))
    energy=np.asarray(energy);ratios=np.asarray(ratios)
    active=energy>10**(-55/20)
    return result,dict(threshold_db=threshold,active_reference_db=-55,
        active_blocks=int(active.sum()),active_attenuated_over_6db=int(np.count_nonzero(active&(ratios<.5))),
        active_retained_fraction=float(np.mean(ratios[active]>=.5)) if active.any() else None,
        energy_retained_ratio=float(np.sum(result.astype(np.float64)**2)/max(np.sum(audio.astype(np.float64)**2),1e-15)),
        definition='Reference RMS > -55 dBFS; not an ASR or human speech label')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wav',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();audio,rate=read_wav(args.wav)
    folder=Path('recordings/input_gate');folder.mkdir(parents=True,exist_ok=True)
    reports=[]
    for threshold in (-43.8,-50.,-55.,-60.):
        output,report=audit(audio,threshold,rate)
        write_wav(folder/f'gate_{threshold:g}.wav',output,rate)
        reports.append(report)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(dict(input=str(args.wav),results=reports),indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
