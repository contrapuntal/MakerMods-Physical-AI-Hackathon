import sys
import types

import numpy as np
import pytest

from harness import height_offset as ho


def test_drop_shape():
    d = ho.Drop(0.055)
    z = np.array([0.30, 0.28, 0.20, 0.128, 0.12, 0.08, 0.05, 0.03])
    out = d.apply(z)
    assert out[0] == pytest.approx(0.30) and out[1] == pytest.approx(0.28)  # untouched high up
    assert out[3] == pytest.approx(0.128 - 0.055)  # the run's 12.8 cm low point lands at 7.3 cm
    assert (out >= np.minimum(z, 0.04) - 1e-12).all()  # never below the floor unless the model went there
    assert out[-1] == pytest.approx(0.03)  # a target already under the floor is left alone
    fine = np.linspace(0.0, 0.4, 4001)
    assert (np.diff(d.apply(fine)) >= -1e-12).all()  # higher in, never lower out


def test_rejects_unsafe_settings():
    with pytest.raises(ValueError):
        ho.Drop(0.2)
    with pytest.raises(ValueError):
        ho.Drop(0.06, engage_m=0.15, full_m=0.12)


def test_patch_touches_only_the_named_model_and_side(monkeypatch):
    seen = []

    class ClientIK:
        def __init__(self, model):
            self._model_id = model

        def solve(self, action, observation, smoother, rtt_s, commanded):
            seen.append(action)
            return action

    fake = types.ModuleType("dream.client_ik")
    fake.ClientIK = ClientIK
    monkeypatch.setitem(sys.modules, "dream", types.ModuleType("dream"))
    monkeypatch.setitem(sys.modules, "dream.client_ik", fake)
    sys.modules["dream"].client_ik = fake
    ho.uninstall()
    ho.install("ft-cables", ho.Drop(0.05))
    rows = [[0.36, 0.20, 0.13], [0.36, 0.20, 0.30]]
    action = {"per_key": {"action.right_eef_xyz": rows, "action.left_eef_xyz": rows}}
    ClientIK("ft-cables").solve(action, None, None, 0.0, None)
    ClientIK("ft-stack").solve(action, None, None, 0.0, None)
    got, untouched = seen
    assert np.asarray(got["per_key"]["action.right_eef_xyz"])[0, 2] == pytest.approx(0.08)
    assert np.asarray(got["per_key"]["action.right_eef_xyz"])[1, 2] == pytest.approx(0.30)
    assert got["per_key"]["action.left_eef_xyz"] is rows  # other side untouched
    assert untouched is action  # other model untouched
    assert action["per_key"]["action.right_eef_xyz"] is rows  # the caller's dict is not modified
    ho.uninstall()
