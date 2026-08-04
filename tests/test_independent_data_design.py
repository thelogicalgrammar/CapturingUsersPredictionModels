import json

import numpy as np

import experiment_randomness as randomness
from slurm.generate_independent_data_parameter_file import (
    DESIGNS,
    generate_manifest,
)


def _scientific_rows():
    for experiment_design, design in DESIGNS.items():
        for param_name in design['centers']:
            for deviation in design['deviations']:
                for k in design['n_covariates']:
                    for replicate in range(50):
                        yield (
                            experiment_design,
                            param_name,
                            deviation,
                            k,
                            replicate,
                            design['scaling'],
                        )


def test_all_registered_stream_keys_are_unique():
    rows = list(_scientific_rows())
    assert len(rows) == 35200
    keys = {
        randomness.run_stream_key(domain, *row)
        for row in rows
        for domain in randomness.DOMAIN_CODES
    }
    assert len(keys) == 3 * len(rows)


def test_streams_are_reproducible_and_domain_separated():
    row = (
        'multivariate_intercept', 'mu_A', -20, 32, 7, 'sqrtk')
    first = {
        domain: randomness.run_rng(
            randomness.BASE_SEED, domain, *row).normal(size=8)
        for domain in randomness.DOMAIN_CODES
    }
    second = {
        domain: randomness.run_rng(
            randomness.BASE_SEED, domain, *row).normal(size=8)
        for domain in randomness.DOMAIN_CODES
    }
    for domain in randomness.DOMAIN_CODES:
        np.testing.assert_array_equal(first[domain], second[domain])
    assert not np.array_equal(
        first['historical_data'], first['observation'])
    assert not np.array_equal(first['observation'], first['sampler'])


def test_manifest_counts_and_settings(tmp_path):
    expected_counts = {
        'univariate_no_intercept': 12200,
        'univariate_intercept': 12200,
        'multivariate_intercept': 10800,
    }
    for experiment_design, expected_count in expected_counts.items():
        output = tmp_path / '{}.txt'.format(experiment_design)
        generate_manifest(experiment_design, output)
        lines = output.read_text().splitlines()
        sidecar = json.loads(
            output.with_suffix('.design.json').read_text())
        assert len(lines) == expected_count
        assert sidecar['line_count'] == expected_count
        assert sidecar['experiment_design'] == experiment_design
        assert sidecar['base_seed'] == randomness.BASE_SEED
        assert sidecar['result_schema_version'] == \
            randomness.RESULT_SCHEMA_VERSION
        assert sidecar['sampler'] == {
            'chains': 4,
            'draws': 1000,
            'init': 'jitter+adapt_diag',
            'max_treedepth': 15,
            'target_accept': 0.85,
            'tune': 1000,
        }


def test_manifest_extensions_preserve_existing_rows(tmp_path):
    full = tmp_path / 'full.txt'
    extension = tmp_path / 'extension.txt'
    generate_manifest(
        'multivariate_intercept', full, replicates=range(50))
    generate_manifest(
        'multivariate_intercept', extension, replicates=range(50, 60))
    full_rows = set(full.read_text().splitlines())
    extension_rows = set(extension.read_text().splitlines())
    assert full_rows.isdisjoint(extension_rows)

    repeated = tmp_path / 'repeated.txt'
    generate_manifest(
        'multivariate_intercept', repeated, replicates=range(50))
    assert full.read_bytes() == repeated.read_bytes()
