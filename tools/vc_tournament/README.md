# v0.11 Offline Zero-Shot VC Mega Tournament

Run from the repository root:

```powershell
.\run_v011_mega_tournament.ps1
```

To rebuild the comparison from saved results without setup, downloads, or new
inference (used after repeated unexpected PC restarts):

```powershell
.\run_v011_mega_tournament.ps1 -ReportOnly
```

This verifies saved WAV hashes, restores results from individual journals, and
rebuilds the blind page/report. Missing CPU refresh measurements stay explicitly
UNMEASURED. Existing WAVs and completed measurements are reused.

To import M4A recordings (originals preserved):

```powershell
.\run_v011_mega_tournament.ps1 -M4aDir <folder-with-m4a-files>
```

The runner prepares **all** catalog candidates before inference. Environments are
isolated under `vc_models/<family>/.venv`, with one shared read-only artifact cache.
Amphion shares an environment between Vevo-Timbre, FACodec V2, and Noro. Existing
`.venv*`, app code, settings, WASAPI route and Beatrice configuration are untouched.
No training command, device capture, Prosody, pitch/formant adjustment or EQ.

Model storage, downloads, Python runtimes, dependency caches and temporary files
stay under `vc_models/`. The launcher and subprocess runner
set task-specific cache directories, including uv, Hugging Face and Torch.
Large identical artifacts can share NTFS hardlinks.
The native Windows MeanVC2 executable canonicalizes junction paths before
opening files. Its binary and GGUF files are staged through a physical ASCII directory
(`--native-dir`), which uses the same file data without a second checkpoint copy.
The downloader and subprocess guards keep 2.5 GiB free.

Place `source_normal.wav`, `source_low.wav`, `source_bright.wav` in
`recordings/v011_mega_tournament/source/`, each 5–10 seconds. `-SourceDir` imports
fixed WAVs from elsewhere. A missing/invalid source blocks the conversion slate
but setup continues. Do not infer recording categories from old unannotated audio.
`-ExtractTargetSources` is only for explicitly authorized target-derived holdouts;
it cannot test removal of male voice and is NOT the current source plan.

References 5/10/20/60s use contiguous acoustic-proxy ranking (clipping, silence,
RMS variation); this does not prove clean noise or stable femininity. Full reference
is preserved byte-for-byte unless explicitly extracting held-out source intervals.
The original file is never modified. Reference provenance and selected offsets
are recorded in `metadata/reference_manifest.json`.

Setup uses CPU PyTorch and curated inference dependencies; it intentionally excludes
training-only requirements. Exact environments are frozen in `requirements.lock.txt`.
Upstream sources are pinned by the fetched commit. Download failure, dependency
failure, RAM limit, timeout and conversion failure remain visible in the manifest
and report. Google Drive access, legacy fairseq, and GPU-only implementations may
prevent some candidates. No fabricated output fills these gaps.

Every conversion runs in a bounded child process with sampled process-tree RSS/CPU.
Default 600s inference timeout, 12GiB RSS guard, and 1200s per setup operation are
offline-computation limits, not hardware audio trials. Hardware audio trials are
not performed. CPU percent uses 100%=one core. Peak RAM is sampled, not exact.
Cold end-to-end RTF includes process startup/model/reference loads; generation-only
metrics are separate where the adapter directly instruments them. Speed labels use
instrumented generation RTF, or cold RTF as a conservative fallback; inspect rtf_scope.
Null latency
means unmeasured. Advertised 40ms/120ms chunks are not measured latency.

CPU sampling retains the same psutil process instance between samples. Legacy
invalid CPU percentages are kept as explicitly invalid metadata and displayed as
unmeasured. On resume, a separate matched source/reference inference refreshes
CPU measurements once per successful model only when explicitly requested with
`-RefreshCpu`, preserving its original quality WAV, generation timing and RAM
measurements. These refreshes also resume from cache. New conversions already
measure CPU; cached legacy results with missing CPU stay UNMEASURED by default.
After repeated PC restarts, use `-ReportOnly`; no more CPU refreshes are required
to listen to the existing 42 WAVs.

Successful WAV caches require source/reference, code, dependency lock, upstream
revision, model artifact hashes, and output SHA256. A failed new render never reuses
an old WAV. Cached setup/downloads resume; unchanged failed setup stages are retained.
Use `-RetryDownload` to explicitly retry failed setup/download stages, or change
their dependency specification/upstream revision. Use
`-SkipSetup` to regenerate reports and infer with installed files. `-RetryDownload`
rechecks the download stage; cached valid files are reused.

Source-independent import/model-load failures are checked once per candidate,
then reused for the other sources with `failure_cache_hit=true`; all failures
remain listed. Changed code, reference, dependency lock or artifact inventory
invalidates this failure cache. Unchanged failed tournament conditions are reused
on resume; `-RetryFailed` explicitly retries their inference processes.

Blind conversion subtracts DC and applies ONE constant gain toward 0.1 RMS,
reduced where necessary for 0.98 peak safety. No compressor/limiter/EQ.
The private mapping is in `metadata/blind_manifest.json`. Model names, performance,
and revealing filenames are omitted from the HTML. Ratings autosave to browser
storage and export/import JSON scoped by a package hash. Original voice residue
and mechanical artifacts use 5=more (worse); the other ratings use 5=better.

`READY FOR HUMAN LISTENING` means actual verified non-silent WAV candidates exist.
It does not mean all requested architectures succeeded. No automatic winner.
Read `validation/v011/mega_tournament_report.md` for every success/failure and source.

Official source links are in `catalog.py` and per-model report. Context7 MCP was
unavailable in this session; checked-out official README/inference code was used.
Windows standalone MeanVC2 uses audio.cpp's official CPU release and self-contained
120ms/40ms FP32 and Q4_K GGUF, separately identified from the Python variants.
Its MeanVC2 family supports only the seed request option, so its built-in sampler
is used without the generic CLI's unsupported diffusion-step override.
Seed-VC uses its official pretrained encoder weights in FP32 on CPU; inference
reads only the prepared shared cache. XLS-R uses the checkpoint's legacy weight
normalization names. Raw Seed WAVs are float32, before blind scalar normalization.
Noro's official CUDA/mel-only path remains a failed CPU comparison entry until a
verified WAV reconstruction path is available; it is never silently omitted.
