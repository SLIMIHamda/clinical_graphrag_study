"""LocalHFClient is a GPU/model path we can't exercise on CPU, but we can pin its
import-safety: importing it (and thus mgr) must not require torch/transformers,
and it must satisfy the generation-client contract used by the executor."""

import pytest

from mgr.clients.local_hf import LocalHFClient


def test_imports_without_torch_and_has_contract():
    assert hasattr(LocalHFClient, "complete_text")
    assert hasattr(LocalHFClient, "complete_text_logprobs")


def test_instantiation_defers_heavy_imports():
    # torch/transformers are absent in the test env; instantiation (not import)
    # is where they load -> proves the lazy-import guard holds.
    with pytest.raises((ImportError, ModuleNotFoundError, OSError)):
        LocalHFClient(model_id="sshleifer/tiny-gpt2")
