"""AUROC checks whose answers follow directly from positive/negative pairs."""

import numpy as np
import pytest

from lsngc.evaluate import recovery_performance


def test_auroc_perfect_reverse_and_ties():
    labels = [np.array([0, 0, 1, 1])] * 4
    predictions = [
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.1, 0.5, 0.5, 0.9]),
    ]
    actual = recovery_performance(predictions, labels)
    assert actual.dtype == np.float32
    # Three winning pairs and one tie in the final case: (3 + 0.5) / 4.
    np.testing.assert_array_equal(actual, [1.0, 0.0, 0.5, 0.875])


def test_auroc_preserves_original_diagonal_inclusion():
    # Both diagonal negatives outrank the one positive. Removing the diagonal
    # would give AUROC=1; the original flatten-all-entries convention gives 1/3.
    labels = np.array([[0, 1], [0, 0]])
    affinity = np.array([[0.9, 0.8], [0.1, 0.9]])
    np.testing.assert_array_equal(
        recovery_performance([affinity], [labels]), np.array([1 / 3], dtype=np.float32)
    )


@pytest.mark.parametrize(
    "predictions,labels,message",
    [
        ([[1, 2]], [], "same number"),
        ([[1, 2]], [[0, 1, 0]], "shapes must match"),
        ([[1, np.nan]], [[0, 1]], "finite"),
        ([[1, 2]], [[0, 0]], "two label classes"),
    ],
)
def test_auroc_rejects_undefined_inputs(predictions, labels, message):
    with pytest.raises(ValueError, match=message):
        recovery_performance(predictions, labels)
