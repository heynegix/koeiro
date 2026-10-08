# Custom Beatrice Clone POC

This folder packages accepted audio and orchestrates the unmodified official
[Beatrice trainer](https://huggingface.co/fierce-cats/beatrice-trainer).
It does not change the realtime application or implement a new trainer.

Upstream revision: `f34836de014b86956096878aecb8d3b17feaaa0b`.
Trainer version: `2.0.0rc0`. Official resume argument: `-r`.
GPU is mandatory. T4 x2 is requested; upstream training uses GPU 0 only.

Build the notebook with `python tools/clone_training/build_notebook.py`.
Upload the accepted-only ZIP as a **private** Kaggle dataset, attach it to the
notebook, enable Internet for installation and official pretrained assets,
then Save & Run All on GPU. Training audio is read from mounted input only.
No automatic download occurs in the realtime application.

The notebook uses batch size 8, generator/discriminator learning rates 5e-5,
and consecutive targets 1000, 2500, 5000. Each later target resumes the latest
checkpoint. `clone_1000`, `clone_2500`, `clone_5000` hold checkpoint, export,
configuration, scalar metrics. GPU usage and stage logs are retained under
`/kaggle/working/beatrice_clone_v1` and must be downloaded from Kaggle Output.

Private notebook: https://www.kaggle.com/code/heynegix/custom-beatrice-clone-v0102
Private input: https://www.kaggle.com/datasets/heynegix/anime-voice-clone-v1-private-poc

Two evaluation WAVs are separate existing normal-voice recordings, not accepted
training clips. Listening against JVS002 +4 and the target voice is required;
successful execution or lower loss alone cannot establish clone quality.
