import subprocess

import numpy as np
import pytest

from multivar_horner import HornerMultivarPolynomial, c_evaluation
from multivar_horner.classes import horner_poly


def test_native_cache_ignores_legacy_and_foreign_libraries(monkeypatch, tmp_path):
    monkeypatch.setattr(horner_poly, "PATH2CACHE", tmp_path)
    poly = HornerMultivarPolynomial(
        [2.0, 3.0], [[0], [2]], rectify_input=True, store_numpy_recipe=True
    )
    legacy = tmp_path / poly.get_c_file_name(c_evaluation.COMPILED_C_ENDING)
    legacy.write_bytes(b"incompatible legacy library")
    native = poly.c_file_compiled
    with monkeypatch.context() as patch:
        patch.setattr(c_evaluation.platform, "machine", lambda: "foreign-architecture")
        foreign = poly.c_file_compiled
        foreign.write_bytes(b"incompatible foreign library")
    assert native != foreign
    poly._compile_c_file()
    assert poly._eval_c(np.array([4.0])) == 50.0
    assert legacy.read_bytes() == b"incompatible legacy library"
    assert foreign.read_bytes() == b"incompatible foreign library"
    # A second instance can reuse the native cache without a compiler.
    monkeypatch.setattr(horner_poly, "get_compiler", lambda: pytest.fail("cache miss"))
    reused = HornerMultivarPolynomial([2.0, 3.0], [[0], [2]], rectify_input=True)
    assert reused.use_c_eval
    assert reused(np.array([4.0])) == 50.0


@pytest.mark.parametrize(
    "system,machine", [("Darwin", "arm64"), ("Darwin", "x86_64"), ("Linux", "aarch64")]
)
def test_compiler_targets_process_architecture(system, machine, monkeypatch, tmp_path):
    monkeypatch.setattr(c_evaluation.platform, "system", lambda: system)
    monkeypatch.setattr(c_evaluation.platform, "machine", lambda: machine)

    def compile_stub(cmd, **kwargs):
        assert kwargs["check"]
        if system == "Darwin":
            assert cmd[-2:] == ["-arch", machine]
        else:
            assert "-arch" not in cmd
        from pathlib import Path

        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"compiled")

    monkeypatch.setattr(c_evaluation.subprocess, "run", compile_stub)
    output = tmp_path / "evaluation.so"
    c_evaluation.compile_c_file("cc", tmp_path / "evaluation.c", output)
    assert output.read_bytes() == b"compiled"


def test_compiler_failure_does_not_publish_partial_library(monkeypatch, tmp_path):
    def fail(cmd, **kwargs):
        from pathlib import Path

        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, cmd, stderr="unsupported architecture")

    monkeypatch.setattr(c_evaluation.subprocess, "run", fail)
    output = tmp_path / "evaluation.so"
    with pytest.raises(ValueError, match="unsupported architecture"):
        c_evaluation.compile_c_file("cc", tmp_path / "evaluation.c", output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
