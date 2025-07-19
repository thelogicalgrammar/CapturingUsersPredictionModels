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
n_mc = 1000
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
agent_model = agent_models.define_model_1(x_o, pred_o, n=1000, tiny=1e-3)


# 6. The agent's belief is updated / the posterior given the observed data is estimated.
latent_sample_size = 100  # Size of plate
with agent_model:
    trace = pm.sample(
        latent_sample_size, 
        target_accept=.99, 
        chains=4, 
        random_seed=42,
        max_treedepth=15,
        init="jitter+adapt_diag"
    )

posterior_df = trace.posterior.to_dataframe()


# 7. Estimate physician's CATE with updated beliefs. Perhaps compare
#    to true model and to what the physician would have done without
#    the updated knowledge.
posterior_CATE_samples = [historical_scm.sample_CATE(tmp_params, x_o, a1_fixed=20, a0_fixed=10) for _, tmp_params in posterior_df.iterrows()]
posterior_CATE_samples = np.concatenate(posterior_CATE_samples)
plt.hist(posterior_CATE_samples, label="Samples from agent's posterior distribution", color="#FF000090", density=True, bins=10)
posterior_CATE_mu = np.mean(posterior_CATE_samples)

# 7.1 Compare to TRUE MODEL, not agent's prior...
true_Ydo1_samples = scm.sample_do_A(a_fixed=20, X=x_o, n=400)
true_Ydo0_samples = scm.sample_do_A(a_fixed=10, X=x_o, n=400)
true_CATE_samples = [item1 - item0 for item0, item1 in zip(true_Ydo0_samples, true_Ydo1_samples)]
true_CATE_mu = np.mean(true_CATE_samples)
plt.hist(true_CATE_samples, label='Samples from true distribution', color="#0000FF90", density=True, bins=10)  # HOW DO I INCLUDE A LABEL FOR THE TWO COLORED PLOTS?
plt.legend()
plt.title("True CATE vs. CATE estimated from physicians posterior.")
plt.savefig('./fig/basic_example_true_and_posterior_CATE.png')
plt.clf()

print("The agent's posterior CATE for *this patient is {} whereas the true CATE is {}.".format(posterior_CATE_mu, true_CATE_mu))

# 7.2 Compare to physicians prior beliefs
#     Do we have a model for the physician's prior beliefs encoded?
