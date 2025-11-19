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
