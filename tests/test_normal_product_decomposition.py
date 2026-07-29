r"""
Test the normal-product decomposition
    \sum(x_i e_i) = \tau g \sqrt(\sum(x_i^2))
where x_i ~ N(a, b), e_i ~ N(0, \tau^2), and g ~ N(0, 1).
"""
import random
import math
import matplotlib.pyplot as plt


def generate_data(n=1000, x_mu=0.0, x_sigma=1.0, e_sigma=1.0):
    x_i = [random.normalvariate(mu=x_mu, sigma=x_sigma) for i in range(n)]  # sigma is the standard deviation!
    e_i = [random.normalvariate(mu=0.0, sigma=e_sigma) for i in range(n)]
    g = random.normalvariate(mu=0.0, sigma=1.0)  # This one always samples on with standard parameters
    sqrt_den = math.sqrt(sum([item**2 for item in x_i]))
    lhs = sum([x_item * e_item for x_item, e_item in zip(x_i, e_i)])  # Direct sum of products
    rhs = e_sigma * g * sqrt_den  # The other expression for the same.
    return lhs, rhs


def generate_plots():
    # Randomize parameters
    e_sigma = random.expovariate(lambd=100)
    x_sigma = random.expovariate(lambd=200)
    x_mu = random.normalvariate(mu=0, sigma=100)
    samples_lhs = []
    samples_rhs = []
    for i in range(10000):
        tmp = generate_data()
        samples_lhs.append(tmp[0])
        samples_rhs.append(tmp[1])
    plt.hist(samples_lhs, alpha=.3, bins=100)
    plt.hist(samples_rhs, alpha=.3, bins=100)
    plt.show()
    plt.clf()


if __name__ == '__main__':
    generate_plots()
