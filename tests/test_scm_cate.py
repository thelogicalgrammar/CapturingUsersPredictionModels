import numpy as np
import pytest

import scm


@pytest.mark.parametrize(
    'model,X_o',
    [
        (scm.SCM_WithConstants(N_E=1.25), 80.0),
        (
            scm.MultivariateSCM(
                n_covariates=3,
                N_E=1.25),
            np.array([80.0, 80.0, 80.0]),
        ),
    ])
def test_cate_is_conditional_mean_contrast(model, X_o):
    samples, mean, std = model.sample_CATE(
        X_o, 20, 10, draws=7, rng=4)

    assert samples.shape == (7,)
    assert np.array_equal(samples, np.full(7, 12.5))
    assert mean == 12.5
    assert std == 0


def test_cate_keeps_batch_shape():
    model = scm.MultivariateSCM(
        n_covariates=2,
        N_E=-0.5)
    X = np.array([
        [70.0, 80.0],
        [80.0, 90.0],
        [90.0, 100.0],
    ])

    samples, mean, std = model.sample_CATE(
        X, 20, 10, draws=5, rng=4)

    assert samples.shape == (3,)
    assert np.array_equal(samples, np.full(3, -5.0))
    assert mean == -5.0
    assert std == 0


def test_cate_does_not_advance_outcome_rng():
    model = scm.SCM_WithConstants()
    rng = np.random.default_rng(4)

    model.sample_CATE(80.0, 20, 10, draws=7, rng=rng)
    outcomes_after_cate, *_ = model.sample_do_A(
        20, 80.0, draws=7, rng=rng)
    expected_outcomes, *_ = model.sample_do_A(
        20, 80.0, draws=7, rng=np.random.default_rng(4))

    assert np.array_equal(outcomes_after_cate, expected_outcomes)
