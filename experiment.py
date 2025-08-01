import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import agent_models
import historical_scm

# Flag for quicker testing of code.
TEST_CODE = False

# Generating the historical data, the prediction model and the 
# prediction on a specific new data point.
# We begin by defining an SCM and generating 100 data points.
# This is the historical SCM so the parameters are fixed
Hist = historical_scm.BinaryA_SCM(
        alpha_x_mu   = 2,
        alpha_x_sigma= 1,
        alpha_a_mu   = 0,
        alpha_y_sigma= 0.1,
        n_e          = -20,
)

X, A, Y = Hist.sample(n=100)
# we check what is the mean outcome in this data
print(np.mean(Y))  #-15.930996903688122
df = pd.DataFrame({'X': X, 'A': A, 'Y': Y})

# fig, ax = plt.subplots()
# scatter = ax.scatter(df["X"], df["Y"], c=df["A"])
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# plt.colorbar(scatter, label="A")  # optional but handy
# plt.show()
# plt.clf()


# These data points are used to train a linear model
pred_model = LinearRegression(fit_intercept=False).fit(X.reshape(-1, 1), Y)
# given a new data point X^o, the model gives a prediction Pred^o for the user
x_obs    = np.asarray(1)  # X° you actually saw
pred_obs = pred_model.predict(x_obs.reshape(-1, 1))  # Pred° you actually saw
print(pred_obs)  # [-6.73087294]


# Define internal model of the agent and do Bayesian update
# ---------- constants ----------
# latent sample size
n         = 100
# NOTE: If this value is too small, the model will have a lot of divergences.
tiny      = 5e-2
Xo_obs    = 1.37
Predo_obs = 2.11
m = agent_models.define_model_1(Xo_obs, Predo_obs, intercept = 12, coefficient_X = 0.1, n=n,
                                tiny=tiny)
if TEST_CODE:
    # For testing purposes, we use a smaller sample size
    # to speed up the sampling.
    sample_size = 7
else:
    sample_size = 1000
with m:
    prior_preds = pm.sample_prior_predictive()

# az.plot_pair(
#     prior_preds,
#     group="prior",
#     # Only plot meaningful parameters
#     var_names=[
#         "NE",
#         "N_E",
#         "phi",
#         "phi_hat",
#         "αA",
#         "βA",
#         "αAμ",
#         "α_Aμ",
#         "αXμ",
#         "αXσ",
#         "α_Xμ",
#         "α_Xσ",
#         "αYσ",
#         "α_Yσ"
#     ]
# )
# plt.show()
# plt.clf()

with m:
    trace = pm.sample(
        sample_size,
        target_accept=.99,
        chains=4,
        random_seed=42,
        max_treedepth=15,
        init="jitter+adapt_diag"
    )
#
# pm.model_to_graphviz(m)
# plt.show()
#
# az.plot_posterior(
#     trace,
# )
#
# az.plot_pair(trace)
# plt.show()
# plt.clf()
#
# # one predictive sample per posterior draw
# rng = np.random.default_rng(123)  # FIXED VALUE RNG SEED!!
# Xnew_samples   = rng.normal(loc=trace.posterior['αXμ'],
#                             scale=trace.posterior['αXσ'])
#
# phi_samples = trace.posterior['phi']
# Prednew_samples = phi_samples * Xnew_samples
#
# # Shape (chains, draws); flatten if you wish
# pred_vec = np.asarray(Prednew_samples).ravel()
# print("Shape of prediction vector: "  + str(Prednew_samples.shape))
#
# plt.plot(Xnew_samples.flatten(), pred_vec)
#
# trace.posterior.keys()
#
#
# if False:
#     df = trace.posterior['NE', ].to_dataframe()  # Where is NE supposed to come from??
#
# NE = df.reset_index(drop=True)
# if False:
#     Y = 0.5 * X + NE * A + N_Y  # N_Y is undefined at this point. ??
#
# # From update to agent decision
# # -Analytic (exact, fastest)
# # -Save samples from the posterior in a dataframe
# df_params = trace.posterior.to_dataframe()
# df_params["cate"] = df_params.apply(historical_scm.estimate_cate_from_row, axis=1)
#
# cate_vector = df_params["cate"].to_numpy()
# print(cate_vector)
#
# plt.hist(cate_vector)
# plt.show()
# plt.clf()
#
# # What happens to the distribution of Y if different copies of the same agent act on different
# # patients? For example if 100 agent defined as above act on 100 new data points.
#
# # What happens if prior of the agent on N_E are way off?
#
# # Version with binary A but X, Y continuous
#
# ##########################################
# ### Model 2
# ##########################################
# # ------------------------------------------------------------------
# #  FILL-IN: your observed values and (optionally) how many training
# #  points you want the model to posit as latent.
# # ------------------------------------------------------------------
# n_train = 30  # size of the (unseen) training sample
# eps_pred = 0.01  # tiny measurement noise for Pred°
# eps_phi = 1e-9  # for safe division
# rng = np.random.default_rng(0)  # for initialization of Z_X
#
# scm_nc = agent_models.define_model_2(x_obs, pred_obs, n_train, eps_pred, eps_phi, rng)
#
# with scm_nc:
#     idata = pm.sample(3000, tune=3000, init="jitter+adapt_diag_grad",
#                       chains = 4, mp_ctx="spawn",
#                       target_accept=0.99, random_seed=42)
#
# with scm_nc:
#     prior_preds = pm.sample_prior_predictive()
#
# plt.scatter(
#     prior_preds.prior_predictive.data_vars['Pred_o'].to_numpy().flatten(),
#     prior_preds.prior_predictive.data_vars['X_o'].to_numpy().flatten()
# )
# plt.xlabel('Pred_o')
# plt.ylabel('X_o')
#
#
# # ------------------------------------------------------------------
# #  Quick look at the updated posteriors
# # ------------------------------------------------------------------
# print(
#     az.summary(
#         idata,
#         var_names=[
#             "α_Xμ",
#             "α_Xσ",
#             "α_Aμ",
#             "α_Yσ",
#             "N_E",
#             "phi_hat",
#         ],
#         round_to=2,
#     )
# )
#
# idata  # print?
#
# az.plot_posterior(idata, var_names=[
#     "α_Xμ",
#     "α_Xσ",
#     "α_Aμ",
#     "α_Yσ",
#     "N_E"
# ])
#
# az.plot_pair(idata, var_names=[
#     "α_Xμ",
#     "α_Xσ",
#     "α_Aμ",
#     "α_Yσ",
#     "N_E"
# ])
#
# ##########################################
# ### Model 3
# ##########################################
# # constants
# if TEST_CODE:
#     n = 7
# else:
#     n = 8000
# Xo_obs    = 1.37
# Predo_obs = 2.11
# tiny      = 1e-3
#
# model_3 = agent_models.define_model_3(Xo_obs, Predo_obs, n, tiny)
# with model_3:
#     trace = pm.sample(2000, target_accept=.99, chains=4)
#
# trace
#
#
# ##########################################
# ### Model 4
# ##########################################
#
# if TEST_CODE:
#     n = 7  # smaller sample size for testing
# else:
#     n = 500
# Xo_obs    = 1.37
# Predo_obs = 2.11
# sigma_P   = 0.05          # <- not 1e-3 !
# rng       = np.random.default_rng(1)
#
# m = agent_models.define_model_4(Xo_obs, Predo_obs, n, sigma_P, rng)
#
# with m:
#     trace = pm.sample(2_000,
#                       target_accept=0.95,
#                       # max_treedepth=15,
#                       init='jitter+adapt_diag',
#                       random_seed=42)
#
# trace
#
# az.plot_posterior(trace)
