import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import agent_model
import scm


print("Testing only that the code runs, not validating results.")


def test_binary_scm():
    # Create an instance of the BinaryA_SCM class
    binary_scm = scm.BinaryA_SCM(
        alpha_mu_X=0.0,
        alpha_sigma_X=1.0,
        alpha_mu_A=0.0,
        alpha_sigma_Y=1.0,
        N_E=-0.5
        )

    # Sample from the SCM
    X, A, Y = binary_scm.sample(draws=10, rng=42)
    print("X:", X)
    print("A:", A)
    print("Y:", Y)

    # Intervene on A
    a_fixed = 1  # Intervening on A to be 1
    Y_do = binary_scm.sample_do_A(a_fixed, X, draws=10, rng=42)
    print("Y | do(A=1):", Y_do)

    tmp = scm.BinaryA_SCM(
        alpha_mu_X=0.0,
        alpha_sigma_X=1.0,
        alpha_mu_A=0.0,
        alpha_sigma_Y=1.0,
        N_E=-0.5
    )

    tmp2 = scm.BinaryA_SCM.sample(tmp, draws=10, rng=42)
    print(tmp2)
    return



hist_model = scm.SCM(
        alpha_mu_X   = 80,
        alpha_sigma_X= 10,
        alpha_mu_A   = 10, 
        alpha_sigma_Y= 1,
)

X, A, Y, *_ = hist_model.sample(draws=7)
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


print("Tests passed for continuous SCM.")

print("Now testing binary SCM...")
test_binary_scm()