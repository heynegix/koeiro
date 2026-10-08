import hashlib
import json
import wave
import zipfile

import pytest

from tools.clone_training.prepare import package


def make_dataset(tmp_path, status='accepted', wav_path='wav/001.wav'):
    (tmp_path / 'wav').mkdir()
    with wave.open(str(tmp_path / 'wav/001.wav'), 'wb') as output:
        output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        output.writeframes(b'\x01\x00' * 32000)
    records = [{'id':'001', 'status':status, 'wav_path':wav_path},
               {'id':'002', 'status':'rejected', 'wav_path':'rejected/missing.wav'}]
    (tmp_path / 'metadata.jsonl').write_text('\n'.join(map(json.dumps, records)))
    (tmp_path / 'validation_report.json').write_text('{"passed":true}')
    return tmp_path


def test_package_excludes_rejected_and_preserves_source(tmp_path):
    source = make_dataset(tmp_path)
    before = (source / 'wav/001.wav').read_bytes()
    archive = tmp_path / 'dataset.zip'
    result = package(source, archive, [])
    assert result['clips'] == 1 and result['duration'] == 2
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {'train/own_female/001.wav', 'accepted_manifest.json'}
        record = json.loads(z.read('accepted_manifest.json'))[0]
        assert record['sha256'] == hashlib.sha256(before).hexdigest()
    assert (source / 'wav/001.wav').read_bytes() == before


def test_failed_validator_blocks_upload_package(tmp_path):
    source = make_dataset(tmp_path)
    (source / 'validation_report.json').write_text('{"passed":false}')
    with pytest.raises(ValueError, match='validator failed'):
        package(source, tmp_path / 'dataset.zip', [])


def test_training_path_escape_rejected(tmp_path):
    source = make_dataset(tmp_path, wav_path='../outside.wav')
    with pytest.raises(ValueError, match='outside wav'):
        package(source, tmp_path / 'dataset.zip', [])


def test_empty_accepted_dataset_rejected(tmp_path):
    source = make_dataset(tmp_path, status='rejected')
    with pytest.raises(ValueError, match='empty'):
        package(source, tmp_path / 'dataset.zip', [])


def test_training_audio_cannot_be_evaluation_input(tmp_path):
    source = make_dataset(tmp_path)
    with pytest.raises(ValueError, match='duplicates a training clip'):
        package(source, tmp_path / 'dataset.zip', [source / 'wav/001.wav'])
