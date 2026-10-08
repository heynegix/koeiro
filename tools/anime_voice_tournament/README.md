# Anime Voice Architecture Tournament — Offline

Realtime app / settings / installed models are not modified. No audio devices,
ASR, Prosody, EQ, Signalsmith or neural training are opened by these tools.

## First listen, then generate the next stage

1. Freeze mono PCM16 48 kHz sources of 8–15 seconds. `source_manifest.json`
   records original hashes and category provenance. At least 5 independent
   recordings and 3 input categories must eventually support a winner.
2. Stage 1 makes 10-second auditions from the **first source only**. The existing
   model inventory selects 20 spread-out JVS IDs and 4 installed official IDs;
   this is an audition slate, not a female/anime quality ranking.
3. Open `recordings/v010_tournament/blind/stage1_voice/index.html`. Select a first
   impression, optionally score 1–5. Mechanical score means **absence of mechanical
   artifacts** (5 good). Export the JSON. No automatic winner or hidden ratings.
4. Explicitly select the human-reviewed 3–5 voices, then search pitch.
5. Select 2–3 human-reviewed pitch candidates, then search native Formant.
6. Two distinct reviewed speakers from the **same model** can use native Morph.
   Cross-model waveform mixing is never called speaker merge.
7. Compare heavy teachers separately; select a winner only after listening on
   at least 3 verified input categories and against JVS002 +4.

```powershell
.venv\Scripts\python.exe tools/anime_voice_tournament/prepare.py
.venv-ai\Scripts\python.exe tools/anime_voice_tournament/run.py --sources recordings/v010_tournament/metadata/source_manifest.json

# A001 etc below are placeholders, not recommendations. Requires exported human ratings.
.venv\Scripts\python.exe tools/anime_voice_tournament/shortlist.py --ratings <ratings.json> --stage stage1_voice --select A001 A002 A003 --phase pitch --out recordings/v010_tournament/metadata/pitch_plan.json
.venv-ai\Scripts\python.exe tools/anime_voice_tournament/run.py --sources recordings/v010_tournament/metadata/source_manifest.json --recipes recordings/v010_tournament/metadata/pitch_plan.json --stage stage2_pitch_formant

# After pitch listening: use its new rating package and selected IDs, then --phase formant.
# After formant listening: --phase merge accepts exactly two speakers from one model.
# To compare reviewed recipes on every fixed input, pass --all-sources.
```

The installed VST's Formant step is **0.5 st**. Plans therefore use −0.5, 0,
+0.5, +1.0, +1.5, rather than generating aliases at +0.25 / +0.75 / +1.25.
Pitch uses native 0.125 st resolution; initial pitch search uses +1.5 to +5 in
0.5 st steps. Merge targets are 80/20, 70/30, 60/40, 50/50. Actual weights are
recorded after native cursor quantization. Two-speaker Morph uses the official
inverse-distance weighting formula. Native codebook Morph includes randomness,
so new renders are not claimed bitwise reproducible; cached audio is frozen.

One child process renders each candidate, with a 180-second timeout. A native
failure records an error and does not stop the slate. Resume validates cache by
source, settings, actual model/plugin hashes, renderer code and output WAV hash.
Candidate IDs are deterministic for a fixed slate. Changed slates/packages must
be rated again; previous ratings are never silently applied to a different package.

Output uses one RMS-based scalar per file, capped at ±6 dB unless greater peak
attenuation is needed for safety. No compression, limiting, EQ or pumping.
Raw peak / over-one counts, scalar and normalized peak are retained. This is
not calibrated perceptual loudness (LUFS). Audition WAV compensates only the
plugin-reported latency; residual algorithmic delay is not asserted measured.

`metadata/tournament_manifest.json` reveals settings; the blind directory and
HTML do not. Keep metadata closed until ratings are made. Audio/models are local
and Git-excluded. Reports contain empty human score fields until actually rated.

## Teacher references

MioTTS-0.6B existing generated anime WAVs can serve as **one teacher family**.
Several files/styles from it do not count as multiple independent architectures.
Style-Bert-VITS2 and Seed-VC are investigated additional families, but ungenerated
references are not ranked or declared anime-quality. TTS references use text and
have different timing from human-source VC; they belong to a separate comparison.
Do not download new weights or change runtime dependencies merely to fill a table.
Stage 4 actual rendering and its dependencies follow the first human screening.

## Current boundary

Stage 1 is ready for listening. Only 2 existing human recordings were found;
6 category coverage, shortlist, stage 2/3 winners and 2–3 teacher families are
pending. `NO CLEAR WINNER` means **not selected yet**, not that Beatrice was
proven incapable. No Prosody rescue, realtime integration or custom training
is authorized by an unrated slate. The final tournament completion tag is deferred.
