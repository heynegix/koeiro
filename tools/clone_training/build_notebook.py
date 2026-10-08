"""Create a private Kaggle notebook using the official trainer CLI, not a new trainer."""
import json
from pathlib import Path


def build():
    folder = Path(__file__).resolve().parent
    cells = []
    def code(text):
        cells.append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=text.splitlines(True)))
    cells.append(dict(cell_type='markdown', metadata={}, source=[
        '# Custom Beatrice single-speaker clone POC\n',
        'Official fierce-cats trainer, fixed revision. Private user-owned dataset.\n',
        '67 accepted clips only. GPU required. Staged 1000 → 2500 → 5000 with official resume.\n',
        'No realtime integration. Quality requires held-out listening; loss is not a quality verdict.\n']))
    code("import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.8.0', 'torchaudio==2.8.0', 'torchvision==0.23.0', '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)\nsubprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub', 'pyworld', 'tensorboard', 'soundfile'], check=True)\nimport torch, torchaudio\nassert torch.cuda.is_available(), 'GPU required; CPU training is forbidden'\nassert tuple(map(int, torchaudio.__version__.split('+')[0].split('.')[:2])) < (2,9), 'Official trainer requires torchaudio < 2.9'\nprint(torch.__version__, torchaudio.__version__, torch.cuda.get_device_name(0))\n")
    code((folder / 'kaggle_run.py').read_text(encoding='utf8'))
    for index, cell in enumerate(cells):
        cell['id'] = f'clone-cell-{index}'
    notebook = dict(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}, nbformat=4, nbformat_minor=5)
    path = folder / 'custom_beatrice_clone_kaggle.ipynb'
    path.write_text(json.dumps(notebook, indent=2), encoding='utf8')
    return path


if __name__ == '__main__':
    print(build())
