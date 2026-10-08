# Project instructions

## Audio verification policy (user instruction from v0.5 onward)

- Every hardware audio trial must be **15 minutes or less**, including setup/cleanup allowance.
- **30 minutes exactly, 30+ minutes, 60 minutes, one-hour, or multi-hour hardware trials are forbidden.** Do not repeat historical v0.3/v0.4 endurance trials.
- Use short bounded stress, crash/recovery, mode/parameter switching and Start/Stop checks instead.
- Label WAV replay, mock tests, user listening and physical microphone/Discord tests separately. Never report unperformed listening/latency verification as measured.
- Offline dataset feature extraction is computation, not a hardware audio endurance trial. It may finish the full dataset without opening audio devices. Never train ProsodyNet in v0.5.

## Architecture

- Keep Beatrice JVS002 native Base Pitch +4 and the stable WASAPI 48 kHz/256 route.
- Original, Female DSP and AI Voice are parallel slots. No Signalsmith Pitch/Formant in the standard AI route.
- Prosody is an independent bounded analysis path. Audio must never wait for F0; use prior controls for subsequent chunks.
- Prosody failure disables its controls, while AI Voice continues. Keep disk/network/GUI/inference/resampling out of the callback.
- Preserve original input datasets/audio, isolate FCPE dependencies, and use the dataset split manifest rather than blindly consuming all samples.

## Library documentation (Context7)

When a user asks about a library/framework/SDK/API/CLI/cloud service, including syntax, configuration, migration, debugging or setup, fetch current documentation with Context7 even for familiar libraries. Resolve the library ID first (unless an exact /org/project ID is provided), choose the most relevant reputable match, then query one concept at a time using the full question. If Context7 is unavailable, disclose that and use official primary documentation. General programming, code review, business logic, refactoring and scripts from scratch do not require Context7 solely for that reason.
