"""Save transcript-free numeric benchmark evidence; no audio devices opened."""
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/'validation/v09'
    benchmarks={}
    for source in sorted(folder.glob('asr-*.json')):
        data=json.loads(source.read_text(encoding='utf-8'))
        benchmarks[source.stem]={k:v for k,v in data.items() if k!='partials'}
    preview=root/'recordings/v09/text_comparison/comparison.json'
    evidence=dict(benchmarks=benchmarks,
        comparison=json.loads(preview.read_text(encoding='utf-8')),
        test_counts=dict(main=421,ai_dataset=31,prosodynet=8,asr=7,total_unique=467),
        human_listening='Unverified',discord='Unverified',assessment='PARTIAL')
    (folder/'summary.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
