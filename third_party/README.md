# Vendored DSP sources

Signalsmith Stretch (MIT): https://github.com/Signalsmith-Audio/signalsmith-stretch
Commit: a670068d9aeb64913331d5cc29337b19a457a7df

Signalsmith Linear (MIT): https://github.com/Signalsmith-Audio/linear
Commit: de55e6a50ffcf6f8f43f649692d94691c7025151

Unmodified upstream headers and their LICENSE.txt files are included. Only the
default portable FFT backend is compiled; no additional FFT dependency is used.
The local streaming C ABI, EQ, and limiter are in src/processors/native/.
Run tools/build_native.py with Visual Studio C++ Build Tools to reproduce the DLL.
No Python package or model download is needed at application runtime.
Source SHA256 values normalize CRLF to LF to remain comparable across Git
checkouts; the DLL SHA256 covers exact bytes.
