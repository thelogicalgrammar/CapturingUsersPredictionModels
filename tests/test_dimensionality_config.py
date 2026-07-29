import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import scm
from dimensionality_config import (
    A0_TREATMENT,
    A1_TREATMENT,
    DECISION_THRESHOLD,
    DEFAULT_SCALING,
    DIMENSIONALITY_EXPERIMENT_DESIGN,
    DIMENSIONALITY_PAIRING,
    FIT_INTERCEPT,
    FIXED_RNG_SEED,
    N_CHAINS,
    N_COVARIATES,
    N_DRAWS,
    N_REPLICATES,
    N_TUNE,
    SCALING_CHOICES,
    SCAN_CENTERS,
    SCAN_DEVIATIONS,
    TARGET_ACCEPT,
    TRUE_SCM,
    run_stream_key,
)
from experiment_dimensionality import (
    _array_fingerprint,
    _prepare_run_randomness,
    _publish_without_overwrite,
)
from slurm.generate_parameter_file_dimensionality import (
    generate_slurm_parameter_file,
    parse_integer_selection,
)
from slurm.generate_parameter_file_univariate_intercept import (
    _load_observations,
    generate_univariate_intercept_manifest,
)


EXPECTED_LINE_COUNT = (
    len(SCAN_CENTERS)
    * len(SCAN_DEVIATIONS)
    * len(N_COVARIATES)
    * N_REPLICATES
)


def _training_features(k):
    data_scm = scm.MultivariateSCM(
        n_covariates=k,
        alpha_mu_X=np.full(k, TRUE_SCM['alpha_mu_X']),
        alpha_sigma_X=np.full(k, TRUE_SCM['alpha_sigma_X']),
        alpha_mu_A=TRUE_SCM['alpha_mu_A'],
        alpha_sigma_A=TRUE_SCM['alpha_sigma_A'],
        alpha_mu_Y=TRUE_SCM['alpha_mu_Y'],
        alpha_sigma_Y=TRUE_SCM['alpha_sigma_Y'],
        N_E=TRUE_SCM['N_E'],
        N_b=np.full(k, TRUE_SCM['N_b'] / k),
        N_d=np.full(k, TRUE_SCM['N_d'] / k),
    )
    features, *_ = data_scm.sample(
        draws=1000, rng=np.random.default_rng(FIXED_RNG_SEED))
    return features


def _patient(param, value, k, replicate, scaling='1k'):
    sequence = np.random.SeedSequence(
        FIXED_RNG_SEED,
        spawn_key=run_stream_key(
            'observation', param, value, k, replicate, scaling))
    patient, *_ = _prepare_run_randomness(
        k, np.random.default_rng(0),
        observation_rng=np.random.default_rng(sequence),
        data_rng=np.random.default_rng(FIXED_RNG_SEED))
    return patient


def test_dimensionality_grid_configuration():
    assert TRUE_SCM['alpha_mu_Y'] == 12.0
    assert SCAN_CENTERS == {
        'mu_A': (2.0, 1.0),
        'mu_X': (80.0, 10.0),
        'mu_Y': (12.0, 0.1),
        'N_E_mu': (1.0, 0.05),
    }
    assert SCAN_DEVIATIONS == (-10, -3, -1, 0, 1, 3, 10)
    assert N_COVARIATES == (1, 2, 4, 8, 16, 32)
    assert N_REPLICATES == 50
    assert (N_DRAWS, N_TUNE, N_CHAINS, TARGET_ACCEPT) == (
        1000, 1000, 4, 0.85)
    assert (A1_TREATMENT, A0_TREATMENT, DECISION_THRESHOLD) == (20, 10, 5)
    assert FIT_INTERCEPT is True
    assert SCALING_CHOICES == {
        '1k': '1/k',
        'sqrtk': '1/sqrt(k)',
    }
    assert DEFAULT_SCALING == '1k'


def test_generated_manifests_preserve_the_registered_grid(tmp_path):
    output_1k = tmp_path / 'parameters_dimensionality.txt'
    output_sqrtk = tmp_path / 'parameters_dimensionality_sqrtk.txt'

    generate_slurm_parameter_file(output_1k, sigma_scaling='1k')
    generate_slurm_parameter_file(output_sqrtk, sigma_scaling='sqrtk')

    lines_1k = output_1k.read_text().splitlines()
    lines_sqrtk = output_sqrtk.read_text().splitlines()

    assert len(lines_1k) == EXPECTED_LINE_COUNT == 8400
    assert len(lines_sqrtk) == EXPECTED_LINE_COUNT
    assert lines_1k[0] == 'mu_A -8.000000 1 0'
    assert lines_1k[-1] == 'N_E_mu 1.500000 32 49'
    assert lines_sqrtk[0] == 'mu_A -8.000000 1 0 sqrtk'
    assert lines_sqrtk[-1] == 'N_E_mu 1.500000 32 49 sqrtk'
    assert lines_sqrtk == [line + ' sqrtk' for line in lines_1k]

    design = json.loads(
        output_1k.with_name(
            'parameters_dimensionality.design.json').read_text())
    assert design['manifest_sha256'] == hashlib.sha256(
        output_1k.read_bytes()).hexdigest()
    assert design['replicates'] == list(range(50))
    assert design['design_version'] == 2
    assert design['experiment_design'] == DIMENSIONALITY_EXPERIMENT_DESIGN
    assert design['pairing'] == DIMENSIONALITY_PAIRING
    assert design['sampler'] == {
        'chains': 4,
        'draws': 1000,
        'target_accept': 0.85,
        'tune': 1000,
    }
    assert design['experiment'] == {
        'a0_treatment': 10,
        'a1_treatment': 20,
        'decision_threshold': 5,
        'fit_intercept': True,
        'plate_size': 1000,
    }
    assert not (tmp_path / 'original_parameters_dimensionality.txt').exists()


def test_manifest_batches_are_exactly_composable(tmp_path):
    full = tmp_path / 'full.txt'
    first = tmp_path / 'first.txt'
    second = tmp_path / 'second.txt'
    kwargs = {
        'sigma_scaling': 'sqrtk',
        'deviations': (-10, 0, 10),
        'n_covariates': (1, 8, 32),
    }
    generate_slurm_parameter_file(full, replicates=(0, 1, 2, 3), **kwargs)
    generate_slurm_parameter_file(first, replicates=(0, 1), **kwargs)
    generate_slurm_parameter_file(second, replicates=(2, 3), **kwargs)

    assert full.read_text() == first.read_text() + second.read_text()
    assert len(full.read_text().splitlines()) == 4 * 3 * 3 * 4


def test_only_registered_grid_extensions_are_generated(tmp_path):
    extension = tmp_path / 'extension.txt'
    generate_slurm_parameter_file(
        extension,
        deviations=(-30, -25, -20, 20, 25, 30),
        n_covariates=(1, 32),
        replicates=(0,),
    )
    assert len(extension.read_text().splitlines()) == 4 * 6 * 2
    with pytest.raises(ValueError, match='registered'):
        generate_slurm_parameter_file(
            tmp_path / 'unregistered.txt',
            deviations=(0,),
            n_covariates=(3,),
            replicates=(0,),
        )


def test_patient_streams_are_stable_independent_and_out_of_sample():
    runs = [
        ('mu_A', -8.0, 32, replicate, scaling)
        for scaling in ('1k', 'sqrtk')
        for replicate in range(50)
    ]
    patients = [_patient(*run) for run in runs]
    assert len({tuple(patient) for patient in patients}) == len(runs)
    assert np.array_equal(
        _patient('mu_A', -8.0, 32, 7, '1k'),
        _patient('mu_A', -8.0, 32, 7, '1k'))
    assert not np.array_equal(
        _patient('mu_A', -8.0, 32, 7, '1k'),
        _patient('mu_A', -8.0, 32, 7, 'sqrtk'))
    assert not np.array_equal(
        _patient('mu_A', -8.0, 32, 7, '1k'),
        _patient('mu_X', -20.0, 32, 7, '1k'))
    assert not np.array_equal(
        _patient('mu_A', -8.0, 32, 7, '1k')[:8],
        _patient('mu_A', -8.0, 8, 7, '1k'))
    training = _training_features(32)
    assert all(
        not np.any(np.all(training == patient, axis=1))
        for patient in patients
    )


def test_new_rows_do_not_change_existing_random_streams(tmp_path):
    original = _patient('mu_A', -8.0, 32, 7, 'sqrtk')
    generate_slurm_parameter_file(
        tmp_path / 'extended.txt',
        sigma_scaling='sqrtk',
        replicates=(7, 50, 79),
        deviations=(-25, -10, 0, 10, 25),
        n_covariates=(1, 32),
    )
    assert np.array_equal(
        original, _patient('mu_A', -8.0, 32, 7, 'sqrtk'))
    assert not np.array_equal(
        original, _patient('mu_A', -8.0, 32, 50, 'sqrtk'))


def test_fixed_univariate_dataset_and_intercept_model_match_reference():
    data_scm = scm.MultivariateSCM(
        n_covariates=1,
        alpha_mu_X=np.asarray([TRUE_SCM['alpha_mu_X']]),
        alpha_sigma_X=np.asarray([TRUE_SCM['alpha_sigma_X']]),
        alpha_mu_A=TRUE_SCM['alpha_mu_A'],
        alpha_sigma_A=TRUE_SCM['alpha_sigma_A'],
        alpha_mu_Y=TRUE_SCM['alpha_mu_Y'],
        alpha_sigma_Y=TRUE_SCM['alpha_sigma_Y'],
        N_E=TRUE_SCM['N_E'],
        N_b=np.asarray([TRUE_SCM['N_b']]),
        N_d=np.asarray([TRUE_SCM['N_d']]),
    )
    features, treatment, outcome, *_ = data_scm.sample(
        draws=1000, rng=np.random.default_rng(FIXED_RNG_SEED))
    model = LinearRegression(fit_intercept=True).fit(features, outcome)

    repeated = data_scm.sample(
        draws=1000, rng=np.random.default_rng(FIXED_RNG_SEED))
    assert _array_fingerprint(features, treatment, outcome) == \
        _array_fingerprint(*repeated[:3])
    assert float(model.intercept_) == pytest.approx(
        13.923777627998746, rel=0.0, abs=1e-12)
    assert float(model.coef_[0]) == pytest.approx(
        0.026233446791786606, rel=0.0, abs=1e-12)


def test_integer_selection_is_stable_for_later_batches():
    assert parse_integer_selection('0-49') == tuple(range(50))
    assert parse_integer_selection('25-49') == tuple(range(25, 50))
    assert parse_integer_selection('2,7,51') == (2, 7, 51)


def test_result_publication_never_overwrites(tmp_path):
    final = tmp_path / 'result.csv'
    first = tmp_path / 'first.tmp'
    first.write_text('first')
    _publish_without_overwrite(first, final)
    assert final.read_text() == 'first'
    assert not first.exists()

    second = tmp_path / 'second.tmp'
    second.write_text('second')
    with pytest.raises(FileExistsError, match='refusing to overwrite'):
        _publish_without_overwrite(second, final)
    assert final.read_text() == 'first'
    assert not second.exists()


def test_default_manifest_names_depend_on_scaling(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    output_1k = generate_slurm_parameter_file(sigma_scaling='1k')
    output_sqrtk = generate_slurm_parameter_file(sigma_scaling='sqrtk')

    assert output_1k == Path('slurm/tmp/parameters_dimensionality.txt')
    assert output_sqrtk == Path(
        'slurm/tmp/parameters_dimensionality_sqrtk.txt')
    assert output_1k.exists()
    assert output_sqrtk.exists()


def test_dense_univariate_manifest_matches_archived_observations(tmp_path):
    output = tmp_path / 'univariate.txt'
    source = Path(__file__).resolve().parents[1] / 'results/wide_results.csv'
    generate_univariate_intercept_manifest(
        source_file=source, parameters_file=output)

    lines = output.read_text().splitlines()
    assert len(lines) == 12200
    assert lines[0].startswith('mu_A -28.000000 0 ')
    assert lines[-1].startswith('N_E_mu 2.500000 49 ')
    mu_y_values = {
        float(line.split()[1])
        for line in lines if line.startswith('mu_Y ')
    }
    assert min(mu_y_values) == 9.0
    assert max(mu_y_values) == 15.0

    design = json.loads(
        output.with_name('univariate.design.json').read_text())
    assert design['source_sha256'] == hashlib.sha256(
        source.read_bytes()).hexdigest()
    assert design['source_sha256'] == (
        'af83e70b7b92ba1d4b00eb6876b76ac007c52bc125fb321d1a40d7de4cb902df')
    assert design['line_count'] == 12200
    assert design['sampler']['draws'] == 1000


def test_dense_univariate_batches_are_exact_set_additions(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'results/wide_results.csv'
    full = tmp_path / 'full.txt'
    first = tmp_path / 'first.txt'
    second = tmp_path / 'second.txt'
    generate_univariate_intercept_manifest(
        source, full, replicates=tuple(range(50)))
    generate_univariate_intercept_manifest(
        source, first, replicates=tuple(range(25)))
    generate_univariate_intercept_manifest(
        source, second, replicates=tuple(range(25, 50)))
    full_lines = set(full.read_text().splitlines())
    first_lines = set(first.read_text().splitlines())
    second_lines = set(second.read_text().splitlines())
    assert first_lines.isdisjoint(second_lines)
    assert first_lines | second_lines == full_lines


def test_dense_univariate_generator_runs_as_a_script():
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable,
         str(repository / 'slurm/generate_parameter_file_univariate_intercept.py'),
         '--help'],
        cwd=repository,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_dense_univariate_manifest_rejects_nonfinite_source(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'results/wide_results.csv'
    malformed = tmp_path / 'malformed.csv'
    with source.open(newline='') as input_stream, \
            malformed.open('w', newline='') as output_stream:
        reader = csv.DictReader(input_stream)
        writer = csv.DictWriter(output_stream, fieldnames=reader.fieldnames)
        writer.writeheader()
        for index, row in enumerate(reader):
            if index == 0:
                row['X_o'] = 'nan'
            writer.writerow(row)
    with pytest.raises(ValueError, match='non-finite'):
        _load_observations(malformed)


def test_result_schema_includes_provenance_and_sign_probability():
    from utils import RESULT_COLUMNS

    assert RESULT_COLUMNS.count((
        'posterior_negative_sign_probability',
        'Posterior P(N_E < 0)',
    )) == 1
    keys = {key for key, _display in RESULT_COLUMNS}
    assert {
        'result_schema_version', 'experiment_design', 'replicate_id',
        'patient_id', 'data_seed', 'sampler_seed', 'sampler_stream_key',
        'n_draws', 'n_tune', 'n_chains', 'target_accept',
    } <= keys


def test_result_writer_rejects_an_existing_different_schema(tmp_path):
    from utils import write_results_to_file

    output = tmp_path / 'legacy.csv'
    output.write_text('"Experiment ID","X_o"\n"old","80.0"\n')
    with pytest.raises(ValueError, match='different schema'):
        write_results_to_file(
            {'experiment_id': 'new', 'X_o': 81.0},
            file_name=output.name,
            path=str(tmp_path) + '/',
        )
    assert output.read_text() == '"Experiment ID","X_o"\n"old","80.0"\n'
