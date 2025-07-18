import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import agent_models
import historical_scm


# This is a basic example of a model with continuous treatments.


# 1. Instantiate SCM that has generated the historical data. Currently
#    this is thought of as the true underlying SCM.
scm = historical_scm.SCM(
        alpha_x_mu   = 80,
        alpha_x_sigma= 10,
        alpha_a_mu   = 10,
        alpha_y_sigma= 0.1,
        n_e          = 20,
        A_type       = 'continuous'
)

# Low value of n to run tests quickly:
n_mc = 40
seed = 7


# 2. Generate data
X, A, Y = scm.sample(n=n_mc, random_state=seed)

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
x_o    = np.asarray(80)  # The observed patient. x_o simulates a patient weighing 80kg.
pred_o = pred_model.predict(x_o.reshape(-1, 1))  # The observed prediction
print("The predicted outcome for this patient weighing {}kg is {}.".format(x_o, pred_o[0]))


# 5. Define agent model. The agent's priors are set in define_model_1() and
#    additionally the data and prediction of the observed patient is passed
#    to the function (for estimating the posterior).
agent_model = agent_models.define_model_1(x_o, pred_o, n=7, tiny=1e-3)


# 6. Maybe feed the same x_o to the updated agent model and see how the
#    decision of the physician changed as consequence of having decision
#    support.




# 7. The agent's belief is updated / the posterior given the observed data is estimated.
latent_sample_size = 5  # Size of plate
with agent_model:
    trace = pm.sample(
        latent_sample_size, 
        target_accept=.99, 
        chains=4, 
        random_seed=42,
        max_treedepth=15,
        init="jitter+adapt_diag"
    )

tmp = trace.posterior.to_dataframe()
