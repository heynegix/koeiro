"""Export the same LLVC weights/state to fixed-chunk CPU ONNX graphs, offline."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import onnxruntime as ort
from src.vc.llvc import LLVCBackend
from src.vc.onnx_graph import FunctionalLLVC


def main():
    folder = Path('models/research_llvc')
    manifest = {'checkpoint_sha256': json.loads((folder/'metadata.json').read_text())['sha256'], 'graphs': {}}
    for factor in (1, 2, 4):
        backend = LLVCBackend(factor, 1)
        backend.load(folder)
        tensors = (torch.zeros(1, 1, backend.chunk_samples+32), backend.enc, backend.dec, backend.out, backend.prenet)
        graph = FunctionalLLVC(backend.model, factor).eval()
        path = folder/f'llvc-f{factor}.onnx'
        with torch.inference_mode():
            torch.onnx.export(graph, tensors, str(path), input_names=['audio', 'encoder', 'decoder', 'out', 'prenet'],
                              output_names=['converted', 'next_encoder', 'next_decoder', 'next_out', 'next_prenet'],
                              opset_version=17, dynamo=False, external_data=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        # Consecutive non-zero chunks validate cache updates, not just silence.
        names = [i.name for i in session.get_inputs()]
        state = [t.numpy().copy() for t in tensors[1:]]
        backend.reset()
        rng = np.random.default_rng(123)
        max_error = 0.0
        previous = np.zeros(32, dtype=np.float32)
        for _ in range(128):
            audio = rng.normal(0, .03, backend.chunk_samples).astype(np.float32)
            inputs = [np.concatenate((previous, audio)).reshape(1, 1, -1), *state]
            actual, *state = session.run(None, dict(zip(names, inputs)))
            expected = backend.process_chunk(audio)
            max_error = max(max_error, float(np.max(np.abs(actual.reshape(-1)-expected))))
            previous[:] = audio[-32:]
        if max_error > 1e-4:
            raise RuntimeError(f'ONNX differs from PyTorch: {max_error}')
        manifest['graphs'][str(factor)] = dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                             bytes=path.stat().st_size, maximum_equivalence_error=max_error)
        print(json.dumps(manifest['graphs'][str(factor)]), flush=True)
        backend.unload()
    (folder/'onnx.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
