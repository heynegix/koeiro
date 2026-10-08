# LLVC inference subset

Source: https://github.com/KoeAI/LLVC
Pinned commit: 1627c5d358cf9bb2b92b0ccc513d8b36807c923d
License: MIT, retained in LICENSE.

Only model.py and cached_convnet.py are included. The model's two imports were
changed to relative package imports. SpeechBrain's sinusoidal positional encoding
was replaced by a minimal implementation of the same standard formula and `pe`
buffer layout. `load_state_dict(strict=True)` checks full checkpoint compatibility.
No ASR, training, RVC inference, cloud, or model collection is included.

The streaming backend retains encoder, attention, output and prenet context.
Original streaming reference: upstream infer.py `infer_stream`.
