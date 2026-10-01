from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from datafest.foundation35 import _make_estimator, _checkpoint_records


def test_explicit_35_selector_without_other_parameter_changes(monkeypatch):
    calls=[]
    version=object()
    def factory(selected, **kwargs):
        calls.append((selected,kwargs))
        return SimpleNamespace(model_path=Path('tabpfn-v3.5-20260909.safetensors'))
    monkeypatch.setitem(sys.modules,'tabpfn',SimpleNamespace(
        TabPFNClassifier=SimpleNamespace(create_default_for_version=factory)))
    monkeypatch.setitem(sys.modules,'tabpfn.constants',SimpleNamespace(
        ModelVersion=SimpleNamespace(V3_5=version)))
    _,context=_make_estimator('tabpfn35','cuda')
    assert calls==[(version,{'device':'cuda'})]
    assert context['selection']=='ModelVersion.V3_5'
    assert context['external_training_subsample'] is False


@pytest.mark.parametrize('name',['tabpfn-v3-classifier.ckpt','tabpfn-v3.5-fast-20260909.safetensors'])
def test_checkpoint_rejects_other_versions(tmp_path,name):
    source=tmp_path/name
    source.write_bytes(b'checkpoint')
    with pytest.raises(RuntimeError,match='explicit TabPFN-3.5'):
        _checkpoint_records(SimpleNamespace(model_path=source),'tabpfn35',tmp_path/'fold')


def test_checkpoint_copies_exact_35_bytes(tmp_path):
    source=tmp_path/'tabpfn-v3.5-20260909.safetensors'
    source.write_bytes(b'exact weights')
    records=_checkpoint_records(SimpleNamespace(model_path=source),'tabpfn35',tmp_path/'fold')
    assert len(records)==1
    assert Path(records[0]['path']).read_bytes()==source.read_bytes()
