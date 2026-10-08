# Third-party notices (shipped set)

This file covers third-party components included when this software is
distributed as a bundle. Items under "Not shipped" are never part of the
distributed bundle. License texts live in their own packages; names and
sources are listed here so the bundle can be reassembled and attributed.

## Shipped

| Component | License | Source / notes |
|---|---|---|
| MeanVC2 code and checkpoints (encoder, Vocos vocoder, VC checkpoint under `vc_models/meanvc2/repo`) | Apache-2.0 | https://github.com/ASLP-lab/MeanVC2 and https://huggingface.co/ASLP-lab/MeanVC2. Keep NOTICE/attribution. |
| LavaSR (`vc_models/post_lavasr/repo`, local `enhancer_v2` weights) | Apache-2.0 | Bundled `LICENSE` in the repo applies. Do not re-download at runtime. |
| PyTorch / torchaudio CPU, SciPy, NumPy, soundfile, psutil, safetensors and other worker dependencies | BSD-style / MIT / Apache-2.0 (per package) | See the worker environment lock (`vc_models/meanvc2/requirements.lock.txt`, minus Windows-only entries on Linux). |
| PySide6 / Qt | LGPLv3 (or commercial, at the distributor's choice) | Dynamic linking as installed by pip. Honor LGPL obligations: allow relinking, ship license text, document where the sources come from. |
| sounddevice / PortAudio | MIT (sounddevice); PortAudio has its own MIT-style license | Linux users install the system PortAudio library separately. |
| sherpa-onnx runtime | Apache-2.0 | ASR is optional and loads only when a text engine is enabled. |
| ReazonSpeech ASR model (`models/asr-reazon`, fetched by `tools/install_asr.py`) | Apache-2.0 | Downloaded by the user at setup time; not embedded in source distributions. |
| Vosk small Japanese model (optional, `tools/install_asr.py --vosk`) | Apache-2.0 | Same handling as above. |
| Signalsmith DSP (used through `female_dsp_x64.dll`-style binaries where present) | MIT (library); bundled binaries keep their copyright notice | Do not strip attributions; do not present as original work. |
| FCPE / torchfcpe dataset tooling | MIT | Offline tooling only. |

## Not shipped (excluded from every distribution)

- Beatrice official models, standing art and bundled files: redistribution is
  prohibited by their terms, including free distribution. Users fetch them
  from the official distribution after agreeing to the terms
  (`tools/install_character_model.py --help`).
- JVS / JVS-MuSiC audio data and paraphernalia: follow their terms; the voice
  data itself is not redistributed.
- Research models and weights under `vc_models/` other than MeanVC2/LavaSR
  (Seed-VC is GPLv3 code; Vevo2 weights are CC-BY-NC-ND-4.0; EZ-VC weights are
  CC-BY-NC-4.0; other checkpoints carry their own terms): never bundle these
  without per-checkpoint clearance.
- pedalboard (GPLv3) and the legacy Beatrice runtime environment: not part of
  the shipped routes.
- User-registered voices, reference audio, embeddings, recordings and logs:
  private by default. Only the sanitized built-in default voice ships; see
  `PUBLISHING.md`.
- FFmpeg and virtual-audio drivers (VB-CABLE etc.): users install them
  separately; their own redistribution terms apply if you bundle them.

## What distributors must still do

- Ship this file plus `LICENSE` with every distribution.
- Keep upstream copyright/attribution files inside redistributed packages.
- Complete the checklist in `PUBLISHING.md` (bundle file list, history scrub
  for source publication, store-specific rules).
