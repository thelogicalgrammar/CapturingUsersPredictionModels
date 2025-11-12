import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import agent_models
import scm


print("Testing only that the code runs, not validating results.")


def test_binary_scm():
    # Create an instance of the BinaryA_SCM class
    binary_scm = scm.BinaryA_SCM(
        alpha_x_mu=0.0,
        alpha_x_sigma=1.0,
        alpha_a_mu=0.0,
        alpha_y_sigma=1.0,
        n_e=-0.5
        )

    # Sample from the SCM
    X, A, Y = binary_scm.sample(n=10, random_state=42)
    print("X:", X)
    print("A:", A)
    print("Y:", Y)

    # Intervene on A
    a_fixed = 1  # Intervening on A to be 1
    Y_do = binary_scm.sample_do_A(a_fixed, X, n=10, random_state=42)
    print("Y | do(A=1):", Y_do)

    tmp = scm.BinaryA_SCM(
        alpha_x_mu=0.0,
        alpha_x_sigma=1.0,
        alpha_a_mu=0.0,
        alpha_y_sigma=1.0,
        n_e=-0.5
    )

    tmp2 = scm.BinaryA_SCM.sample(tmp, n=10, random_state=42)
    print(tmp2)
    return



hist_model = scm.SCM(
        alpha_x_mu   = 80,
        alpha_x_sigma= 10,
        alpha_a_mu   = 10, 
        alpha_y_sigma= 1,
        A_type = 'continuous',
)

X, A, Y = hist_model.sample(n=7)
# we check what is the mean outcome in this data
print(np.mean(Y))  #-15.930996903688122
df = pd.DataFrame({'X': X, 'A': A, 'Y': Y})

fig, ax = plt.subplots()
scatter = ax.scatter(df["X"], df["Y"], c=df["A"])
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.colorbar(scatter, label="A")  # optional but handy
#plt.show()
#plt.clf()
plt.savefig("./tests/tmp_test_historical_scm.png")

# These data points are used to train a linear model
pred_model = LinearRegression(fit_intercept=False).fit(X.reshape(-1, 1), Y)
# given a new data point X^o, the model gives a prediction Pred^o for the user
Xo_obs    = np.asarray(72.0)  # A person that would weight 72 kg.
Predo_obs = pred_model.predict(Xo_obs.reshape(-1, 1))  # Pred° you actually saw
print(Predo_obs)

sigma_P   = 0.05          # <- not 1e-3 !
rng       = np.random.default_rng(1)

n = 30

agent_model = agent_models.define_model_1(Xo_obs, Predo_obs, n, tiny=sigma_P)

with agent_model:
    trace = pm.sample(20,
                      target_accept=0.95,
                      # max_treedepth=15,
                      init='jitter+adapt_diag',
                      random_seed=42)

trace

az.plot_posterior(trace)
plt.savefig("./tests/tmp_test_agent_model_1_posterior.png")

print("Tests passed for continuous SCM.")

print("Now testing binary SCM...")
test_binary_scm()