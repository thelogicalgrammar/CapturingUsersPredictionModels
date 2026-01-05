# Capturing users prediction models

- experiment:
    - Contains the experiments reported in the paper
    - This can also be used as a basic walk-through of how to use the package
- agent_model:
    - Contains AgentModel class, which encodes the agent's prior beliefs 
    (A_hist in the paper)
    - The Bayesian model that the agent is using to update their beliefs, i.e., the pymc code for the agent-internal model
    - Methods for predictive sampling
- scm:
    - The model in the paper to implement hist and the A_inf model
    - Note that the AgentModel includes an object of class scm.SCM for the A_inf model.
- tests folder:
    - Files to run the code tests for the various parts of the model

## Quickstart

Run `python experiment.py N_E_mu 17 80` to draw posterior samples given that the
agent has default priors with the exception of N_E_mu which takes value 17, and
X_o takes value 80.

## Documentation

Navigate to ./docs/_build/html/index.html locally to access documentation.

## Use Case
- Inputs: true underlying distributions, Agent's beliefs, patient characteristics
- Outputs: AI-prediction for patient, treatment decisions based on prior (no decision
  support) and posterior beliefs (with decision support). Actual outcome for patient
  under the different treatments.
- We are interested in knowing for what kind of inputs the outputs change, and when
  they change for the better or for the worse.

## Trace
If you want to work with trace directly, access it with `model.trace`.

## Tests
To run code tests, use the following commands:
```
python -m tests.test_scm
python -m tests.test_agent_models
```

## Slurm
Pre-compile code to speed up runs with 
```
python -m compileall .
```
Define parameters to scan over in generate_parameter_file.py and run
```
python generate_parameter_file.py
```
OR populate ```./slurm/tmp/parameters.txt``` with desired parameter names and values.

Set ```#SBATCH --array=0-99``` to a range that starts at 0 and ends at 1/4 of the number
of lines in parameters.txt rounded up.

Run by submitting on cluster with the command
```
sbatch slurm/array_job.sh
```


