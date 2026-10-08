import hashlib
import json
import zipfile
import pytest
from tools.install_beatrice import install


def fixture(tmp_path, actual=b'official'):
    data=b'official'
    manifest={'version':'2.0.0-rc.3','files':{'model/test.bin':
        {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}}
    (tmp_path/'beatrice-manifest.json').write_text(json.dumps(manifest))
    archive=tmp_path/'test.zip'
    with zipfile.ZipFile(archive,'w') as zip_file:
        zip_file.writestr('beatrice_2.0.0-rc.3/beatrice_paraphernalia_jvs/test.bin',actual)
        zip_file.writestr('../../unselected-file',b'never extracted')
    return archive,manifest


def test_local_archive_imports_only_verified_manifest_files(tmp_path):
    archive,_=fixture(tmp_path)
    assert install(archive,tmp_path)==1
    assert (tmp_path/'beatrice/model/test.bin').read_bytes()==b'official'
    assert not (tmp_path/'unselected-file').exists()


def test_corrupt_archive_leaves_existing_installation_untouched(tmp_path):
    archive,_=fixture(tmp_path,b'corrupt!')
    destination=tmp_path/'beatrice/model/test.bin'
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b'existing')
    with pytest.raises(ValueError,match='checksum'):
        install(archive,tmp_path)
    assert destination.read_bytes()==b'existing'


def test_manifest_path_escape_rejected(tmp_path):
    archive,manifest=fixture(tmp_path)
    manifest['files']['../../escape.bin']=manifest['files'].pop('model/test.bin')
    (tmp_path/'beatrice-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='manifest path'):
        install(archive,tmp_path)
