"""K-means compatibility for the published implementation's 2020 defaults.

The NumPy initializer adapts scikit-learn 0.23.1's k-means++ initializer;
see ``THIRD_PARTY_NOTICES`` for its BSD-3-Clause notice. Modern scikit-learn
performs clustering through its public ``KMeans`` API. The seed schedule,
ten restarts, and stopping convention preserve the reference implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

FloatArray = NDArray[np.float64]


def _initial_centers(
    states: FloatArray, n_clusters: int, random_state: np.random.RandomState
) -> FloatArray:
    """Seed centers with the greedy k-means++ procedure used in 0.23.1.

    The first index uses ``randint``; each subsequent center is selected from
    ``2 + floor(log(k))`` candidates sampled proportionally to squared distance
    from the closest center. This preserves the historical random draw order.
    """
    trials = 2 + int(np.log(n_clusters))
    norms = np.einsum("ij,ij->i", states, states)

    def squared_distances(points: FloatArray) -> FloatArray:
        distances = -2 * np.dot(points, states.T)
        distances += np.einsum("ij,ij->i", points, points)[:, None]
        distances += norms[None, :]
        np.maximum(distances, 0, out=distances)
        return distances

    centers = np.empty((n_clusters, states.shape[1]), dtype=np.float64)
    centers[0] = states[random_state.randint(len(states))]
    closest = squared_distances(centers[:1])
    potential = closest.sum()
    for index in range(1, n_clusters):
        values = random_state.random_sample(trials) * potential
        candidates = np.searchsorted(np.cumsum(closest, dtype=np.float64), values)
        np.clip(candidates, None, closest.size - 1, out=candidates)
        proposed = squared_distances(states[candidates])
        np.minimum(closest, proposed, out=proposed)
        potentials = proposed.sum(axis=1)
        best = np.argmin(potentials)
        centers[index] = states[candidates[best]]
        closest = proposed[best]
        potential = potentials[best]
    return centers


def cluster_centers(
    states: NDArray[np.floating], n_clusters: int, seed: int
) -> FloatArray:
    """Fit K-means with the original seed, restart, and stopping conventions.

    This is the clustering step in Methods, "Nonlinear transformation using
    generalized radial basis function," Eqs. (4)-(5), and Supplement section 2
    of Wismüller et al., Scientific Reports 11, 7817 (2021).

    The original float32 tensor input was converted to float64 by scikit-learn.
    Version 0.23.1 drew ten independent child seeds and returned the previous
    centers when their squared shift first fell below ``mean(var(X))*1e-4``.
    Current KMeans returns the updated centers. Replaying each run through
    its preceding iteration restores the original convention without relying
    on private scikit-learn APIs. A run that reaches 100 iterations without
    satisfying the tolerance retains its last updated centers.
    """
    data = np.asarray(states, dtype=np.float64, order="C")
    tolerance = np.var(data, axis=0).mean() * 1e-4
    seeds = np.random.RandomState(seed).randint(np.iinfo(np.int32).max, size=10)
    algorithm = "lloyd" if n_clusters == 1 else "elkan"
    best_centers: FloatArray | None = None
    best_inertia = np.inf

    for child_seed in seeds:
        initial: list[FloatArray] = []

        def capture_initial(
            values: FloatArray,
            count: int,
            random_state: np.random.RandomState,
        ) -> FloatArray:
            centers = _initial_centers(values, count, random_state)
            initial.append(centers.copy())
            return centers

        fit = KMeans(
            n_clusters=n_clusters,
            max_iter=100,
            random_state=int(child_seed),
            n_init=1,
            algorithm=algorithm,
            init=capture_initial,
        ).fit(data)
        centers = fit.cluster_centers_
        inertia = fit.inertia_

        if fit.n_iter_ > 1:
            previous = KMeans(
                n_clusters=n_clusters,
                max_iter=fit.n_iter_ - 1,
                tol=0,
                random_state=int(child_seed),
                n_init=1,
                algorithm=algorithm,
                init=_initial_centers,
            ).fit(data)
            shift = np.sum((centers - previous.cluster_centers_) ** 2)
            if fit.n_iter_ < 100 or shift <= tolerance:
                centers = previous.cluster_centers_
                inertia = previous.inertia_
        else:
            # Immediate tolerance convergence keeps the initialization.
            mean = data.mean(axis=0)
            centers = initial[0] + mean
            centered = data - mean
            inertia = sum(
                min(
                    sum(float(a - b) ** 2 for a, b in zip(row, center))
                    for center in initial[0]
                )
                for row in centered
            )

        if inertia < best_inertia:
            best_centers = centers.copy()
            best_inertia = inertia

    assert best_centers is not None
    return best_centers
