# Third-party notices (shipped set)

This file covers third-party components included when this software is
distributed as a bundle. Items under "Not shipped" are never part of the
distributed bundle. License texts live in their own packages; names and
sources are listed here so the bundle can be reassembled and attributed.

## Shipped

| Component | License | Source / notes |
|---|---|---|
| MeanVC2 code and checkpoints (encoder, VC checkpoints `meanvc2_*_40ms.safetensors` under `vc_models/meanvc2/repo`) | Apache-2.0 | https://github.com/ASLP-lab/MeanVC2 and https://huggingface.co/ASLP-lab/MeanVC2. Upstream requires the copyright notice and disclaimer to travel with the model. |
| MeanVC2 pre-processing checkpoints: WavLM-Large (`wavlm_large.pt`), ECAPA speaker encoder (`wavlm_large_finetune.pth`), `fastu2pp_*.pt` | WavLM-Large: upstream statements disagree (the `microsoft/unilm` repository is MIT, the Hugging Face card links the UniSpeech LICENSE = CC BY-SA 3.0). ECAPA recipe: Apache-2.0 | Attribution is required either way; both licenses do allow commercial use. Resolve before a commercial bundle (`microsoft/unilm` issue 1757, opened 2026-09-07, is still unanswered). |
| Vocos vocoder weights used by MeanVC2 (`vc_models/meanvc2/repo/ckpts/vocos/vocos.pt`) | MIT | https://github.com/charactr-platform/vocos |
| LavaSR code and `enhancer_v2` weights (`vc_models/post_lavasr`) | Apache-2.0 | The bundled upstream `LICENSE` applies; the README states "The model and code are licensed under the Apache-2.0 license." Do not re-download at runtime. |
| Vendored `vocos` that LavaSR imports (`vc_models/post_lavasr/vendor/vocos*`) | MIT (upstream charactr-platform/vocos; LavaSR pins the `langtech-bsc` "matcha" fork) | Keep the package metadata and license file with the copy you ship. |
| PyTorch / torchaudio CPU, SciPy, NumPy, soundfile, psutil, safetensors and other worker dependencies | BSD-style / MIT / Apache-2.0 (per package) | See the worker environment lock (`vc_models/meanvc2/requirements.lock.txt`, minus Windows-only entries on Linux). |
| PySide6 / Qt | LGPLv3 (or commercial, at the distributor's choice) | Dynamic linking as installed by pip. Honor LGPL obligations: allow relinking, ship license text, document where the sources come from. |
| sounddevice / PortAudio | MIT (sounddevice); PortAudio has its own MIT-style license | Linux users install the system PortAudio library separately. |
| sherpa-onnx runtime | Apache-2.0 | ASR is optional and loads only when a text engine is enabled. |
| ReazonSpeech ASR model (`models/asr-reazon`, fetched by `tools/install_asr.py`) | Apache-2.0 | Downloaded by the user at setup time; not embedded in source distributions. |
| Vosk small Japanese model (optional, `tools/install_asr.py --vosk`) | Apache-2.0 | Same handling as above. |
| Signalsmith Linear / Stretch (headers under `third_party/signalsmith-*`, DLL built by `tools/build_native.py`) | MIT | Unmodified upstream headers and their `LICENSE.txt` files ship in the source repository. Keep the copyright notices; do not present as original work. |
| FCPE (`third_party/fcpe/LICENSE`) with its `torchfcpe` tooling | MIT | Offline dataset work only. The application's own F0 analysis is a NumPy YIN estimator and never imports FCPE or torch. |
| LLVC code copied under `third_party/llvc` | MIT (Copyright (c) 2023 Koe AI) | Research route only; the shipped conversion path does not use it. |

## Not shipped (excluded from every distribution)

- `vc_models/post_lavasr/vendor/encodec*` and `vendor/bin/encodec.exe`: the
  0.1.1 wheel metadata declares **CC BY-NC 4.0** (NonCommercial), while the
  current upstream `facebookresearch/encodec` repository is MIT. The shipped
  enhancer imports the vendored `vocos`, not `encodec`, so leave these files
  out of the bundle instead of relying on the upstream relabeling.
- Beatrice official models, standing art and bundled files: redistribution is
  prohibited by their terms, including free distribution. Users fetch them
  from the official distribution after agreeing to the terms
  (`tools/install_character_model.py --help`).
- JVS / JVS-MuSiC audio data and paraphernalia: follow their terms; the voice
  data itself is not redistributed.
- Research models and weights under `vc_models/` other than MeanVC2/LavaSR
  (Seed-VC is GPLv3 code; Vevo2 weights are CC-BY-NC-ND-4.0; EZ-VC weights are
  CC-BY-NC-4.0; FlashSR, MossFormer and VoiceFixer checkpoints carry their own
  terms): never bundle these without per-checkpoint clearance.
- pedalboard (GPLv3) and the legacy Beatrice runtime environment: not part of
  the shipped routes.
- User-registered voices, reference audio, embeddings, recordings and logs:
  private by default. Only the sanitized built-in default voice ships; see
  `PUBLISHING.md`.
- FFmpeg and virtual-audio drivers (VB-CABLE etc.): users install them
  separately; their own redistribution terms apply if you bundle them.

## What distributors must still do

- Ship this file plus `LICENSE` with every distribution, and keep the model
  attribution table in `README.md` in sync with what the bundle actually
  contains.
- Keep upstream copyright/attribution files inside redistributed packages.
- For the frozen Windows build: include the PySide6/Qt license text, state
  where the Qt sources can be obtained, and keep the dynamic linking that lets
  a user relink (LGPLv3).
- Complete the checklist in `PUBLISHING.md` (bundle file list, history scrub
  for source publication, store-specific rules).
