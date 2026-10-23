import numpy as np

from util.util import compute_results


def test_compute_results_known_values():
    # rows: ground truth, columns: prediction
    conf = np.array([
        [40, 5, 0],
        [10, 20, 0],
        [2, 3, 0],
    ], dtype=np.float64)
    precision, recall, iou, f1 = compute_results(conf)

    assert np.isclose(precision[0], 40 / 52)
    assert np.isclose(recall[0], 40 / 45)
    assert np.isclose(iou[0], 40 / 57)
    assert np.isclose(f1[0], 2 / (52 / 40 + 45 / 40))

    assert np.isclose(precision[1], 20 / 28)
    assert np.isclose(recall[1], 20 / 30)
    assert np.isclose(iou[1], 20 / 38)
    assert np.isclose(f1[1], 2 / (28 / 20 + 30 / 20))

    # class 2 has no predictions, so precision and F1 are nan
    assert np.isnan(precision[2])
    assert np.isclose(recall[2], 0.0)
    assert np.isclose(iou[2], 0.0)
    assert np.isnan(f1[2])


def test_compute_results_empty_class_is_nan():
    conf = np.array([
        [4, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
    ], dtype=np.float64)
    precision, recall, iou, f1 = compute_results(conf)

    assert np.isclose(precision[0], 4 / 5)
    assert np.isclose(recall[0], 1.0)
    assert np.isclose(iou[0], 4 / 5)
    assert np.isclose(f1[0], 2 / (5 / 4 + 1))

    # class 1 is never predicted
    assert np.isnan(precision[1])
    assert np.isclose(recall[1], 0.0)
    assert np.isclose(iou[1], 0.0)
    assert np.isnan(f1[1])

    # class 2 has no ground truth and no predictions
    assert np.isnan(precision[2])
    assert np.isnan(recall[2])
    assert np.isnan(iou[2])
    assert np.isnan(f1[2])
