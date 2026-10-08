"""Print comparable summaries from developer JSON reports without audio I/O."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',type=Path,nargs='+')
    args=parser.parse_args()
    for path in args.reports:
        report=json.loads(path.read_text(encoding='utf-8'))
        print(path.name, 'error=',report.get('error'))
        for i,cycle in enumerate(report.get('cycles',[])):
            ai=cycle['ai']
            print(json.dumps(dict(cycle=i+1,rtf=ai.get('rtf'),inference=ai.get('inference'),
                post=ai.get('postprocess'),resample=ai.get('resample'),rpc=ai.get('rpc'),
                callback=cycle['callback'],main=cycle['main'],queue_max=ai['queue_max'],
                under=ai['ai_underrun'],over=ai['ai_overrun'],drop=ai['dropped_chunks'],
                pa=[cycle['underflow'],cycle['overflow']],ram=ai.get('ram_bytes'),
                gaps=cycle['diagnostics']['late_arrivals'],max_gap=cycle['diagnostics']['maximum_callback_gap_ms']),
                ensure_ascii=True))
            for observation in cycle.get('observations',[]):
                state=observation['ai']
                print('observation',json.dumps(dict(seconds=observation['seconds'],
                    input_queue=state['queue_current'],output_queue=state['output_queue_current'],
                    queue_max=state['queue_max'],ram=state.get('ram_bytes',state['model'].get('ram_bytes')),
                    cpu_seconds=state.get('cpu_seconds',state['model'].get('cpu_seconds')))))


if __name__=='__main__':main()
