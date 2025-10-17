# Capturing users prediction models
 
- basic_example
    - Contains a basic walk-through of the functions
- agent_models:
    - Contains AgentModel class, which implements A_hist in the paper
    - The Bayesian model that the agent is using to update their beliefs, i.e., the pymc code for the agent-internal model
- scm:
    - The model in the paper to implement hist and the A_inf model
- tests folder:
    - Files to run the sanity checks / code tests for the various parts of the model
- experiment
    - Contains the experiments reported in the paper