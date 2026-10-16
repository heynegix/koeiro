"""Can the shipped TorchScript traces be moved to another compute device at all?

`torch.jit.freeze` inlines weights as graph constants and a later `.to()` cannot
reach them, which is what issue #6 hit on a GPU. Two more things pin a device: a
device written into the graph as a constant, which `retarget_devices` rewrites, and
an allocation the trace never gave a device, which lands on the process default and
nothing can move. This reads the two traces the app loads, counts each kind, and runs
the app's own retarget pass over them. Exit 0 means load order plus that pass is the
whole job on this side of the boundary.

    python tools/verify_meanvc2_trace_devices.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.vc.meanvc2 import retarget_devices

TRACES = [
    (ROOT / 'vc_models/meanvc2/repo/preprocess/ckpts/fastu2pp_160ms.pt',
     ['m.encoder.embed.pos_enc.pe'], 'graph'),
    (ROOT / 'vc_models/meanvc2/repo/ckpts/vocos/vocos.pt', ['decode'], 'decode'),
]

# Allocations that choose a device for themselves. aten::to is not one of them when
# it only converts a dtype, which is what a missing device argument means here.
FACTORY = ('aten::ones', 'aten::zeros', 'aten::arange', 'aten::full', 'aten::empty',
           'aten::eye', 'aten::scalar_tensor', 'aten::new_ones', 'aten::new_zeros',
           'aten::full_like', 'aten::ones_like', 'aten::zeros_like')


def traced_device(node):
    """The device argument a node was traced with, or None when it has none."""
    for argument in node.inputs():
        producer = argument.node()
        if producer.kind() != 'prim::Constant' or str(producer.output().type()) != 'Device':
            continue
        try:
            return str(producer.s('value'))
        except RuntimeError:
            continue
    return None


def inspect(graph):
    """What pins a device here, split by whether the app can do anything about it."""
    named = []
    stranded = []
    for node in graph.nodes():
        device = traced_device(node)
        if node.kind() in FACTORY:
            (named if device is not None else stranded).append((node.kind(), device))
        elif node.kind() == 'aten::to' and device is not None:
            named.append((node.kind(), device))
    return named, stranded


def main():
    problems = []
    for path, preserved, method in TRACES:
        if not path.is_file():
            print(f'{path.name}: not found. Run tools/setup_release.py first.')
            return 2
        # An open file, not a path: libtorch's own reader fails on a non-ASCII
        # Windows path, and this is how the backend loads the same trace.
        with path.open('rb') as handle:
            module = torch.jit.freeze(torch.jit.load(handle, map_location='cpu').eval(),
                                      preserved_attrs=preserved)
        graph = module.graph if method == 'graph' else module.decode.graph
        named, stranded = inspect(graph)
        on_cpu = retarget_devices([graph], torch.device('cpu'))
        on_gpu = retarget_devices([graph], torch.device('cuda'))
        print(f'{path.name}: {len(named)} device(s) written into the graph '
              f'{sorted(set(named)) or ""}, {len(stranded)} allocation(s) with no device at all')
        print(f'  retarget_devices: {on_cpu} moved on the CPU route (must be 0), '
              f'{on_gpu} moved when asked for a GPU')
        problems.extend(stranded)
        if on_cpu:
            problems.append('the shipped CPU route is not the device these traces name')
    if problems:
        print('More than load order and the retarget pass stands in the way: '
              f'{sorted(set(str(problems)))}')
        return 1
    print('Load order plus retarget_devices covers every device these traces name.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
