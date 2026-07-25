"""Forward-pass smoke test, output-shape contract, and diphone->phone
marginalization sanity."""

import pytest
import torch

from model import SkipDiphoneDecoder

C = 3                    # num_phonemes (kept tiny so num_diphones = C*C = 9)
NUM_DIPHONES = C * C


def _build(variant):
    return SkipDiphoneDecoder(
        input_dim=8, hidden_dim=4, num_layers=1,
        num_phonemes=C, num_diphones=NUM_DIPHONES, num_days=2,
        variant=variant, kernel_len=4, stride_len=2, dropout=0.0,
    )


def _inputs(B=2, T=16, D=8):
    x = torch.randn(B, T, D)
    lengths = torch.tensor([T, T - 4])
    day_ids = torch.tensor([0, 1])
    return x, lengths, day_ids


def test_forward_smoke_and_shapes():
    model = _build("E").eval()
    x, lengths, day_ids = _inputs()
    with torch.no_grad():
        out = model(x, lengths, day_ids, return_logits=True)

    Tp = out["enc_lengths"]  # strided length bookkeeping
    assert (Tp <= (lengths - model.preprocessor.kernel_len)
            // model.preprocessor.stride_len + 1).all()

    for key in ("diphone_log_probs", "skip_log_probs"):
        assert out[key].shape[1:] == (2, NUM_DIPHONES + 1)   # (T', B, classes)
        assert torch.isfinite(out[key]).all()

    # phoneme_probs is a proper distribution over C+1 classes
    pp = out["phoneme_probs"]
    assert pp.shape[-1] == C + 1
    assert torch.allclose(pp.sum(-1), torch.ones_like(pp.sum(-1)), atol=1e-4)
    assert (pp >= 0).all()


def test_log_probs_normalized():
    model = _build("E").eval()
    x, lengths, day_ids = _inputs()
    with torch.no_grad():
        out = model(x, lengths, day_ids)
    probs = out["diphone_log_probs"].exp().sum(-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)


def test_variant_a_only_builds_mono_head():
    model = _build("A")
    assert hasattr(model, "mono_head")
    assert not hasattr(model, "diphone_head")
    with torch.no_grad():
        out = model(*_inputs())
    assert "mono_log_probs" in out
    assert "diphone_log_probs" not in out


def test_marginalize_sums_previous_phoneme():
    """_marginalize should recover phone mass by summing diphone prob over the
    previous-phoneme axis, then renormalize with blank into a distribution."""
    model = _build("E")
    B, T = 1, 2
    # uniform log-probs over all NUM_DIPHONES+1 classes
    lp = torch.full((B, T, NUM_DIPHONES + 1),
                    -torch.log(torch.tensor(float(NUM_DIPHONES + 1))))
    pp = model._marginalize(lp)
    assert pp.shape == (B, T, C + 1)
    assert torch.allclose(pp.sum(-1), torch.ones(B, T), atol=1e-5)
    # each phone gets C diphones' mass -> phones equal, and each phone mass
    # equals C * blank mass before renorm; check phones are equal to each other
    phones = pp[..., :C]
    assert torch.allclose(phones, phones[..., :1].expand_as(phones), atol=1e-5)


def test_decode_marginalize_logsumexp_normalized():
    """WER-path marginalization (log-sum-exp over previous phoneme) must return
    a normalized (phone, blank) log-distribution."""
    pytest.importorskip("editdistance")
    from decode import _marginalize_diphone_logits

    logits = torch.randn(2, 5, NUM_DIPHONES + 1)
    out = _marginalize_diphone_logits(logits, C)
    assert out.shape == (2, 5, C + 1)
    assert torch.allclose(out.exp().sum(-1),
                          torch.ones(2, 5), atol=1e-4)
