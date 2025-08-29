import arviz as az
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import agent_models as am
import historical_scm



# Set number of observations used to train prediction model.
# This is also the plate size used later.
PLATE_SIZE = 100

# 1. Instantiate SCM to generate the "historical" data. Currently
#    this is thought of as the true underlying SCM.
scm = historical_scm.SCM(
        alpha_x_mu   = 80,
        alpha_x_sigma= 10,
        alpha_a_mu   = 10,
        alpha_y_sigma= 0.1,
        n_e          = 20,
        A_type       = 'continuous'
)

# 2. Generate data from historical model
X, A, Y = scm.sample(n=PLATE_SIZE)

# 2.1 Plot features against outcomes. Treatment dosage color coded:
plt.scatter(X, Y, c=A)
plt.xlabel("Weight (X)")
plt.ylabel("Outcome (Y)")
plt.title("Effect of dosage (A) on outcome for different patients (X)")
plt.colorbar(label="A")
plt.savefig('./fig/basic_example_historical_data.png')
plt.clf()


# 3. Train prediction model on historical data. 
pred_model = LinearRegression(fit_intercept=False).fit(X.reshape(-1, 1), Y)


# 4. The prediction model is now used for decision support. The physician
#    inputs the data of the patient and the prediction model produces a prediction.
X_o = np.asarray(80)  # The observed patient. x_o simulates a patient weighing 80kg.
pred_o = pred_model.predict(X_o.reshape(-1, 1))  # The observed prediction
print("The predicted outcome for this patient weighing {}kg is {}.".format(X_o, pred_o[0]))


# 5. Define agent model. The agent's priors are set at initialization of class
#    by passing constants and prior_parameters dictionaries.
agent_model = am.AgentModel(plate_size=PLATE_SIZE)  # Using default values for priors and constants.

# 6. The agent's belief is updated / the posterior given the observed data is estimated.
trace = agent_model.sample(X_o=np.asarray(X_o), pred_o=np.asarray(pred_o[0]), draws=1000)

# Alternatively access the model direcltly and adjust sampling parameters:
if False:
    with agent_model.model:
        trace = pm.sample(
            draws=1000, 
            target_accept=.99, 
            chains=4, 
            random_seed=42,
            max_treedepth=15,
            init="jitter+adapt_diag"
        )

# This is super inefficient. Let's not switch to dataframe.
# posterior_df = trace.posterior.to_dataframe()


# What is the posterior distribution of the agent's belief about N_E?
az.plot_posterior(trace, var_names=['N_E'])
plt.title("Posterior distribution of agent's belief about N_E")
plt.savefig('./fig/basic_example_posterior_NE.png')
plt.clf()

# Started with N_E ~ Normal(20,1) prior.
# -Could compare prior with posterior.
# -Could also estimate how the prediction for patient with X_o changes with
# some intervention level A. Model is linear, so higher is better, but
# we can track the change for some specific level. 
# -Could track change in uncertainty of outcome for patient.