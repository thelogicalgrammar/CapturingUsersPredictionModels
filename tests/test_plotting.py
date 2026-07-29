import numpy as np
import pandas as pd
import pytest

from utils import _mixture_moments


def test_mixture_moments_include_within_and_between_patient_variance():
    data = pd.DataFrame({
        'prior': [0, 0, 1],
        'mean': [1.0, 3.0, 4.0],
        'std': [2.0, 2.0, 0.5],
    })

    summary = _mixture_moments(data, 'prior', 'mean', 'std')

    assert summary.loc[0, 'mean'] == pytest.approx(2.0)
    assert summary.loc[0, 'std'] == pytest.approx(np.sqrt(5.0))
    assert summary.loc[1, 'mean'] == pytest.approx(4.0)
    assert summary.loc[1, 'std'] == pytest.approx(0.5)
