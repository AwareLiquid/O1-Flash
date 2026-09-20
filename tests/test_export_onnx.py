"""ONNX export tests: export a trained fixed-schema model and verify the
ONNX runtime reproduces PyTorch outputs (real end-to-end usage)."""

import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from mt_flash.config import o1_flash_tiny
from mt_flash.model import O1Flash
from mt_flash.train import default_questions, make_synthetic_state, train


def test_onnx_export_matches_pytorch(tmp_path):
    import random

    # 1. train a tiny model (single question is enough for the export
    #    contract; joint is covered by test_train)
    model = O1Flash(o1_flash_tiny())
    qs = default_questions()
    train(model, qs, steps=120, batch=16)
    model.eval()

    # 2. export at a fixed sequence length
    seq = 96
    onnx_path = str(tmp_path / "o1_flash_tiny.onnx")
    from mt_flash.export_onnx import export_onnx
    export_onnx(model, qs, onnx_path, seq_len=seq)

    # 3. run the ONNX graph and compare against PyTorch
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rows = [make_synthetic_state(random.Random(i)) for i in range(8)]
    ids = torch.nn.utils.rnn.pad_sequence(
        [model._ids_for(r[0]) for r in rows], batch_first=True,
        padding_value=256)
    ids = ids[:, :seq]
    if ids.shape[1] < seq:
        ids = torch.nn.functional.pad(ids, (0, seq - ids.shape[1]), value=256)

    with torch.no_grad():
        y = model(ids)
        mask = ids != 256
        pt_probs = model.heads.train_forward(y, qs, pad_mask=mask)

    feeds = {"ids": ids.numpy()}
    out = sess.run(None, feeds)
    ort_probs = {q.id: torch.tensor(v) for q, v in zip(qs, out)}

    for q in qs:
        pt = pt_probs[q.id]
        on = ort_probs[q.id]
        assert on.shape == pt.shape
        # Decision-level contract: same argmax everywhere. All three heads
        # use perturbation-robust readouts (logsumexp pooling over
        # positions), so raw probabilities also agree tightly; the
        # generous bound covers cross-platform ONNX runtime float32
        # variation across the unrolled recurrence.
        assert on.argmax(-1).tolist() == pt.argmax(-1).tolist(), q.id
        assert (on - pt).abs().max().item() < 0.15, q.id


def test_exported_model_outputs_are_schema_bounded(tmp_path):
    import random

    model = O1Flash(o1_flash_tiny())
    qs = default_questions()
    train(model, qs, steps=120, batch=16)
    model.eval()

    seq = 96
    onnx_path = str(tmp_path / "m.onnx")
    from mt_flash.export_onnx import export_onnx
    export_onnx(model, qs, onnx_path, seq_len=seq)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rows = [make_synthetic_state(random.Random(50 + i)) for i in range(4)]
    ids = torch.nn.utils.rnn.pad_sequence(
        [model._ids_for(r[0]) for r in rows], batch_first=True,
        padding_value=256)
    ids = ids[:, :seq]
    if ids.shape[1] < seq:
        ids = torch.nn.functional.pad(ids, (0, seq - ids.shape[1]), value=256)

    out = sess.run(None, {"ids": ids.numpy()})
    for v in out:
        assert abs(v.sum(axis=-1) - 1.0).max() < 1e-5
        assert (v >= 0).all() and (v <= 1).all()
