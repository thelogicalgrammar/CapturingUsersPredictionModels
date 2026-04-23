"""
Smoke test for the dedicated scalability experiment driver.

The test keeps the configuration intentionally tiny. It validates the
CLI-facing experiment code path, output file generation, and diagnostics
serialization for one run.
"""
from pathlib import Path
import tempfile

import numpy as np

import scalability_experiment as se


print("Testing only that the scalability experiment driver runs end-to-end.")


def test_scalability_experiment_variant_helpers():
    variants = (
        'full_no_intercept',
        'reduced_no_intercept',
        'full_intercept',
        'reduced_intercept',
    )
    expected_uses_intercept = {
        'full_no_intercept': False,
        'reduced_no_intercept': False,
        'full_intercept': True,
        'reduced_intercept': True,
    }
    expected_uses_reduction = {
        'full_no_intercept': False,
        'reduced_no_intercept': True,
        'full_intercept': False,
        'reduced_intercept': True,
    }
    for variant in variants:
        model = se.build_agent_model(
            model_variant=variant,
            agent_priors=se.build_agent_priors(1),
            scm_constants=se.build_scm_constants(1, 'inverse_k'),
            plate_size=5,
        )
        assert model is not None
        assert se.model_uses_intercept(variant) == expected_uses_intercept[variant]
        assert se.model_uses_reduction(variant) == expected_uses_reduction[variant]
        trace_var_names = se.build_trace_var_names(variant)
        assert 'phi' in trace_var_names
        assert 'N_b' in trace_var_names
        assert 'N_d' in trace_var_names
        if se.model_uses_intercept(variant):
            assert 'alpha' in trace_var_names
        else:
            assert 'alpha' not in trace_var_names


def test_scalability_experiment_driver():
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        summary, diagnostics_rows, output_paths = se.run_scalability_experiment(
            n_covariates=1,
            plate_size=5,
            draws=5,
            tune=5,
            chains=2,
            target_accept=.9,
            max_treedepth=10,
            init='jitter+adapt_diag',
            model_variant='full_no_intercept',
            coefficient_scaling='inverse_k',
            x_o_mode='mean',
            seed=4,
            output_dir=output_dir,
            run_name='test_scalability_driver_full_no_intercept',
        )

        assert summary['model_variant'] == 'full_no_intercept'
        assert summary['n_covariates'] == 1
        assert summary['plate_size'] == 5
        assert np.isfinite(summary['total_runtime'])
        assert 'n_divergences' in summary
        assert 'max_rhat' in summary
        assert 'min_ess_bulk' in summary
        assert 'min_ess_tail' in summary
        assert len(diagnostics_rows) > 0
        assert output_paths['summary_json'].exists()
        assert output_paths['summary_csv'].exists()
        assert output_paths['diagnostics_csv'].exists()

    print("Scalability experiment driver smoke test passed.")


if __name__ == '__main__':
    test_scalability_experiment_driver()
