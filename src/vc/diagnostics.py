"""Cheap startup checks on the file worker; full checksum validation at load."""
import json
from pathlib import Path
from .client import worker_python


def startup_diagnostics(root,model=None):
    root = Path(root)
    issues = []
    from .models import profile, default_voice_id
    if model is None:
        model = default_voice_id()
        if model is None:
            return dict(ai_available=False, issues=['No voice profile is available'],
                        checksum_validation='Performed in isolated worker on load')
    try:
        worker_python(root,model)
    except (OSError, RuntimeError) as error:
        issues.append(str(error))
    folder = root/'models'/profile(model)['folder']
    try:
        runtime = json.loads((folder/'runtime.json').read_text(encoding='utf-8'))
        if not isinstance(runtime,dict):
            raise ValueError('Invalid Beatrice runtime')
        if runtime.get('backend')=='meanvc2':
            for item in runtime['assets']:
                path=(root/item['path']).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('MeanVC2 asset missing')
            return dict(ai_available=not issues,issues=issues,checksum_validation='Fixed reference and all model assets checked in isolated worker on load')
        if runtime.get('backend') != 'beatrice_vst':
            raise ValueError('Beatrice runtime required')
        assets = (folder/'beatrice').resolve()
        for key in ('plugin','model'):
            path = (folder/runtime[key]).resolve()
            if not path.is_relative_to(assets) or not path.is_file():
                raise ValueError('Beatrice model/plugin missing')
        manifest = json.loads((folder/'beatrice-manifest.json').read_text(encoding='utf-8'))
        if not isinstance(manifest,dict) or not isinstance(manifest.get('files'),dict):
            raise ValueError('Invalid Beatrice manifest')
        for name, expected in manifest['files'].items():
            if not isinstance(expected,dict):
                raise ValueError('Invalid Beatrice asset metadata')
            path = (assets/name).resolve()
            if not path.is_relative_to(assets) or not path.is_file() or path.stat().st_size != expected['bytes']:
                raise ValueError('Beatrice asset missing/invalid')
    except (OSError,ValueError,TypeError,KeyError) as error:
        issues.append(str(error))
    return dict(ai_available=not issues, issues=issues,
                checksum_validation='Performed in isolated worker on load')
