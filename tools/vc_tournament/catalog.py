"""Official upstream inventory. Variants are not independent model families."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Model:
    id: str
    family: str
    repo: str
    reference: str = '10s'
    python: str = '3.10'
    note: str = ''

MODELS = [
    Model('meanvc2_40', 'meanvc2', 'ASLP-lab/MeanVC2', python='3.11'),
    Model('meanvc2_120', 'meanvc2', 'ASLP-lab/MeanVC2', python='3.11'),
    Model('meanvc2_cpu', 'audiocpp', '0xShug0/audio.cpp', note='Windows standalone CPU; 120ms/40ms GGUF package only'),
    Model('meanvc2_cpu_q4', 'audiocpp', '0xShug0/audio.cpp', note='Windows standalone CPU; 120ms/40ms Q4_K GGUF; separate quality/RAM comparison'),
    Model('meanvc', 'meanvc', 'ASLP-lab/MeanVC', python='3.11'),
    Model('conan', 'conan', 'MaxMax2016/Conan-Voice-Conversion'),
    Model('conan_fast', 'conan', 'MaxMax2016/Conan-Voice-Conversion',note='Official Fast folder currently has config/logs but no main VC checkpoint; Fast Emformer alone is insufficient'),
    Model('seedvc_tiny', 'seedvc', 'Plachtaa/seed-vc', note='25M Tiny; 10 diffusion steps; F0 correction OFF'),
    Model('seedvc', 'seedvc', 'Plachtaa/seed-vc', note='98M offline; 30 diffusion steps; F0 correction OFF'),
    Model('xvc', 'xvc', 'Jerrister/X-VC', note='Upstream GPU-oriented; CPU WAV inference verified; measured N150 RTF exceeds 2, offline comparison only'),
    Model('vevo', 'amphion', 'open-mmlab/Amphion', note='Vevo-Timbre only; 32 flow matching steps'),
    Model('facodec', 'amphion', 'open-mmlab/Amphion', note='FACodec V2: source codes without residual + reference timbre'),
    Model('noro', 'amphion', 'open-mmlab/Amphion', note='Official path outputs mel; separate BigVGAN reconstruction required'),
    Model('knnvc_20', 'knnvc', 'bshall/knn-vc', reference='20s', note='prematched HiFiGAN, topk=4'),
    Model('knnvc_full', 'knnvc', 'bshall/knn-vc', reference='full', note='prematched HiFiGAN, topk=4'),
    Model('freevc', 'freevc', 'OlaWod/FreeVC', python='3.9', note='CPU device adaptation; upstream CUDA calls not used'),
    Model('fragmentvc', 'fragmentvc', 'yistLin/FragmentVC', python='3.8', note='Legacy TorchScript / fairseq compatibility risk'),
    Model('ezvc', 'ezvc', 'EZ-VC/EZ-VC', note='XEUS / ESPnet dependency; pretrained weights CC-BY-NC'),
    Model('openvoice', 'openvoice', 'myshell-ai/OpenVoice', note='V2 ToneColorConverter only; source and reference embeddings; no TTS'),
]

def families():
    return {m.family: m for m in MODELS}
