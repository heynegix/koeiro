"""Import the user's official local Beatrice archive; never downloads assets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def install(archive, folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'beatrice-manifest.json').read_text(encoding='utf-8'))
    prefix = 'beatrice_'+manifest['version']+'/'
    prepared = []
    with zipfile.ZipFile(archive) as source:
        for name, expected in manifest['files'].items():
            destination = (folder/'beatrice'/name).resolve()
            if not destination.is_relative_to(folder/'beatrice'):
                raise ValueError('Invalid manifest path')
            if name.startswith('beatrice.vst3/'):
                member = prefix+'beatrice_'+manifest['version']+'.vst3/'+name.split('/',1)[1]
            elif name.startswith('model/'):
                member = prefix+'beatrice_paraphernalia_jvs/'+name.split('/',1)[1]
            else:
                member = prefix+name
            info = source.getinfo(member)
            if info.file_size != expected['bytes']:
                raise ValueError('Unexpected official archive file size: '+name)
            data = source.read(info)
            if hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('Unexpected official archive checksum: '+name)
            prepared.append((destination, data))
    # Check all assets before modifying the installation. No archive paths are
    # extracted directly; only bounded, verified manifest destinations are used.
    for destination, data in prepared:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return len(prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[1]/'models/girl_01'
    print(f'Imported {install(args.archive,folder)} verified official assets.')
    print('JVS model: unauthorized commercial use prohibited. See README and model description.')


if __name__=='__main__':
    main()
