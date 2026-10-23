import tempfile
from pathlib import Path

import numpy as np
import PIL.Image
import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st

from util.loss import fusion_loss
from util.RGB_Depth_dataset import RGB_Depth_dataset
from util.RGB_Disparity_dataset import RGB_Disparity_dataset
from util.util import compute_results

_DATA_ROOT = Path(tempfile.mkdtemp(prefix="inconseg-prop-"))


def _reference_metrics(conf):
    """Precision, recall, IoU, and F1 for one square confusion matrix.

    Rows are ground truth and columns are predictions, including class 0.
    A class whose denominator is zero contributes nan. F1 is also nan when
    precision or recall is zero, which is the rule compute_results implements.
    """
    matrix = np.asarray(conf, dtype=np.float64)
    n_class = matrix.shape[0]
    true_positive = np.diag(matrix).astype(np.float64, copy=True)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    union = actual + predicted - true_positive

    precision = np.full(n_class, np.nan, dtype=np.float64)
    recall = np.full(n_class, np.nan, dtype=np.float64)
    iou = np.full(n_class, np.nan, dtype=np.float64)
    f1 = np.full(n_class, np.nan, dtype=np.float64)

    predicted_ok = predicted != 0
    actual_ok = actual != 0
    union_ok = union != 0
    precision[predicted_ok] = true_positive[predicted_ok] / predicted[predicted_ok]
    recall[actual_ok] = true_positive[actual_ok] / actual[actual_ok]
    iou[union_ok] = true_positive[union_ok] / union[union_ok]

    defined = (
        ~np.isnan(precision)
        & ~np.isnan(recall)
        & (precision != 0)
        & (recall != 0)
    )
    f1[defined] = 2.0 / (1.0 / precision[defined] + 1.0 / recall[defined])
    return precision, recall, iou, f1


@st.composite
def confusion_matrices(draw):
    kind = draw(
        st.sampled_from(
            ["counts", "counts", "counts", "counts", "zero", "nan", "inf", "empty"]
        )
    )
    if kind == "empty":
        return np.zeros((0, 0), dtype=np.float64)
    n_class = draw(st.integers(min_value=1, max_value=5))
    if kind == "zero":
        return np.zeros((n_class, n_class), dtype=np.float64)
    counts = draw(
        st.lists(
            st.integers(min_value=0, max_value=40),
            min_size=n_class * n_class,
            max_size=n_class * n_class,
        )
    )
    matrix = np.asarray(counts, dtype=np.float64).reshape(n_class, n_class)
    if kind in ("nan", "inf"):
        row = draw(st.integers(min_value=0, max_value=n_class - 1))
        col = draw(st.integers(min_value=0, max_value=n_class - 1))
        if kind == "nan":
            matrix[row, col] = np.nan
        else:
            matrix[row, col] = draw(st.sampled_from([np.inf, -np.inf]))
    return matrix


@settings(deadline=None, database=None, max_examples=1000)
@given(matrix=confusion_matrices())
@example(np.zeros((0, 0), dtype=np.float64))
@example(np.zeros((3, 3), dtype=np.float64))
@example(np.full((2, 2), np.nan))
@example(np.full((2, 2), np.inf))
@example(np.array([[np.nan, 0.0], [1.0, np.inf]], dtype=np.float64))
@example(np.array([[40.0, 5.0, 0.0], [10.0, 20.0, 0.0], [2.0, 3.0, 0.0]]))
def test_compute_results_matches_reference(matrix):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        got = compute_results(matrix)
        expected = _reference_metrics(matrix)
    assert len(got) == 4
    for actual, ref in zip(got, expected):
        assert actual.shape == (matrix.shape[0],)
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, ref, rtol=1e-6, atol=1e-6, equal_nan=True)
    if np.isfinite(matrix).all() and np.all(matrix >= 0):
        for metric in got:
            finite = np.isfinite(metric)
            assert np.all(metric[finite] >= 0.0)
            assert np.all(metric[finite] <= 1.0)


def _label_spatial(feat_h, feat_w, mode, alt_h, alt_w):
    """Label (H, W) against logits stored as (N, C, H, W)."""
    label_h, label_w = feat_h, feat_w
    if mode in ("height", "both"):
        label_h = alt_h if alt_h != feat_h else feat_h + 1
    if mode in ("width", "both"):
        label_w = alt_w if alt_w != feat_w else feat_w + 1
    return label_h, label_w


def _nearest_labels(label, out_h, out_w):
    """Nearest resize with the same index map as F.interpolate(..., mode='nearest')."""
    in_h = int(label.shape[-2])
    in_w = int(label.shape[-1])
    if in_h == out_h and in_w == out_w:
        return label
    rows = torch.arange(out_h) * in_h // out_h
    cols = torch.arange(out_w) * in_w // out_w
    return label[:, rows][:, :, cols]


@settings(deadline=None, database=None, max_examples=200)
@given(
    batch=st.integers(min_value=1, max_value=2),
    n_class=st.integers(min_value=2, max_value=4),
    feat_h=st.integers(min_value=1, max_value=5),
    feat_w=st.integers(min_value=1, max_value=5),
    mode=st.sampled_from(["match", "width", "height", "both"]),
    alt_h=st.integers(min_value=1, max_value=5),
    alt_w=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_fusion_loss_properties(
    batch, n_class, feat_h, feat_w, mode, alt_h, alt_w, seed
):
    label_h, label_w = _label_spatial(feat_h, feat_w, mode, alt_h, alt_w)
    if mode == "match":
        assert (label_h, label_w) == (feat_h, feat_w)
    elif mode == "width":
        assert label_h == feat_h and label_w != feat_w
    elif mode == "height":
        assert label_w == feat_w and label_h != feat_h
    else:
        assert label_h != feat_h and label_w != feat_w

    torch.manual_seed(seed)
    logits = torch.randn(batch, n_class, feat_h, feat_w, requires_grad=True)
    depth_add = torch.randn(batch, n_class, feat_h, feat_w, requires_grad=True)
    label = torch.randint(0, n_class, (batch, label_h, label_w))

    loss_pr, loss_add, add_map = fusion_loss(logits, depth_add, label)

    assert loss_pr.ndim == 0 and bool(torch.isfinite(loss_pr))
    assert loss_add.ndim == 0 and bool(torch.isfinite(loss_add))
    assert add_map.shape == (batch, feat_h, feat_w)
    assert add_map.dtype == torch.long
    assert add_map.requires_grad is False
    assert add_map.grad_fn is None

    resized = _nearest_labels(label, feat_h, feat_w)
    prediction = logits.detach().argmax(dim=1)
    expected = torch.where(prediction == resized, torch.zeros_like(resized), resized)
    assert torch.equal(add_map, expected)

    (loss_pr + loss_add).backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert bool(torch.isfinite(logits.grad).all())
    assert float(logits.grad.abs().sum()) > 0
    assert depth_add.grad is not None
    assert bool(torch.isfinite(depth_add.grad).all())
    assert float(depth_add.grad.abs().sum()) > 0
    assert add_map.grad is None


def _write_split(root, folder, head, rgb, label, depth):
    for sub in ("left", "labels", folder):
        (root / sub).mkdir(parents=True, exist_ok=True)
    name = "sample"
    PIL.Image.fromarray(rgb).save(root / "left" / f"left{name}.png")
    PIL.Image.fromarray(label).save(root / "labels" / f"label{name}.png")
    PIL.Image.fromarray(depth).save(root / folder / f"{head}{name}.png")
    (root / "train.txt").write_text(name + "\n")
    return name


def _check_dataset(
    dataset_cls, folder, head, src_h, src_w, input_h, input_w, depth_kind, fill, n_class, seed
):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)
    label = rng.integers(0, n_class, size=(src_h, src_w), dtype=np.uint8)
    if depth_kind == "zero":
        depth = np.zeros((src_h, src_w), dtype=np.uint8)
    elif depth_kind == "constant":
        depth = np.full((src_h, src_w), fill, dtype=np.uint8)
    else:
        depth = rng.integers(0, 256, size=(src_h, src_w), dtype=np.uint8)

    root = _DATA_ROOT / dataset_cls.__name__
    name = _write_split(root, folder, head, rgb, label, depth)
    dataset = dataset_cls(str(root), "train", input_h=input_h, input_w=input_w)
    image, label_t, got_name = dataset[0]

    assert got_name == name
    assert tuple(image.shape) == (4, input_h, input_w)
    assert bool(torch.isfinite(image).all())
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    depth_t = image[3]
    if depth_kind == "zero":
        assert bool(torch.all(depth_t == 0))
    elif depth_kind == "constant":
        assert float(depth_t.max()) == pytest.approx(1.0)
    if bool(torch.all(depth_t == 0)):
        assert float(depth_t.max()) == 0.0
    else:
        assert float(depth_t.max()) == pytest.approx(1.0)

    assert label_t.dtype == torch.int64
    assert tuple(label_t.shape) == (input_h, input_w)
    source_ids = {int(value) for value in np.unique(label)}
    got_ids = {int(value) for value in torch.unique(label_t).tolist()}
    assert got_ids.issubset(source_ids)


@settings(deadline=None, database=None, max_examples=100)
@given(
    src_h=st.integers(min_value=1, max_value=12),
    src_w=st.integers(min_value=1, max_value=12),
    input_h=st.integers(min_value=1, max_value=12),
    input_w=st.integers(min_value=1, max_value=12),
    depth_kind=st.sampled_from(["zero", "constant", "random"]),
    fill=st.integers(min_value=1, max_value=255),
    n_class=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_rgb_depth_dataset_properties(
    src_h, src_w, input_h, input_w, depth_kind, fill, n_class, seed
):
    _check_dataset(
        RGB_Depth_dataset,
        "depth",
        "depth",
        src_h,
        src_w,
        input_h,
        input_w,
        depth_kind,
        fill,
        n_class,
        seed,
    )


@settings(deadline=None, database=None, max_examples=100)
@given(
    src_h=st.integers(min_value=1, max_value=12),
    src_w=st.integers(min_value=1, max_value=12),
    input_h=st.integers(min_value=1, max_value=12),
    input_w=st.integers(min_value=1, max_value=12),
    depth_kind=st.sampled_from(["zero", "constant", "random"]),
    fill=st.integers(min_value=1, max_value=255),
    n_class=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_rgb_disparity_dataset_properties(
    src_h, src_w, input_h, input_w, depth_kind, fill, n_class, seed
):
    _check_dataset(
        RGB_Disparity_dataset,
        "disparity",
        "disparity",
        src_h,
        src_w,
        input_h,
        input_w,
        depth_kind,
        fill,
        n_class,
        seed,
    )
