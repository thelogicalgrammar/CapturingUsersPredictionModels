"""
Statistical tests for testing whether two groups of observations can
be considered "meaningfully similar." "Meaningfully similar" is
parameterized by max_diff and p_mass where the first one defines what
is an acceptable level of difference in the parameters and the second
how probable it has to be that the parameters for the two
distributions are within max_diff.

BAYESIAN RANK TEST FOR SIMILARITY (implemented)
Tests whether the pairwise rank differs from what
would be expected if the observations were from
identical distributions.

BAYESIAN TEST FOR SIMILARITY OF MEAN (implemented)
Based on CLT. Checks whether the means of the two
groups differs significantly.

BOOTSTRAP TEST FOR MEANS, VARIANCE, AND SKEWNESS (implemented)
Using bootstrap to estimate spread of sample mean estimates,
sample variance estimates, and sample skewness estimates, and
compares whether these clearly different.

WASSERSTEIN DISTANCE (implemented)
a.k.a Earth-mover's distance
A normalized version is included here.

KOLMOGOROW-SMIRNOV TEST (skip, maybe future)
Largest distance between the cumulative distribution functions of
the two groups.
The test estimates a p-value but that value is highly dependent
on the number of observations in the groups. The test cannot reject
similarity, but it can in rare cases prove it (statistically 
significant result).

KL-DIVERGENCE (skip)
As the variables take continuous values, we would have to
estimate the continuous distributions somehow, which introduced
another source of uncertainty.

LIKELIHOOD (future)
Likelihood of some data should be the same.
-Include here?

TEST FOR MODELING SIMILARITY (future)
Dependencies between variables in one group can be modeled
using e.g. logistic regression and similarity can be evaluated
by comparing e.g. AUC-ROC on observations from both groups.

MULTI-DIMENSIONAL RANK TEST (future)
Could eventually look into multi-dimensional extensions of below.
Does CLT extend to multiple dimensions?
Rank similarity should be possible to extend using a Dirichlet-distribution
and L2-distance. Although the Dirichlet-distribution does not consider
dependence between the variables.
-Maybe a stratified version?
"""

from scipy.stats import binom, beta, binom, skew, wasserstein_distance
import numpy as np
import warnings

# Number of observations to sample for comparison of uncertainty distributions:
MC_SAMPLES = 10000
BOOTSTRAP_SAMPLES = 1000  # Increasing this slows down the test considerably.
VERBOSE = False


def print_verbose(text, verbose=VERBOSE):
    if verbose:
        print(text)


def bayesian_test_for_rank_similarity(observations_a,
                                      observations_b,
                                      max_diff=0.01,
                                      p_mass=0.95):
    """
    This pair-wise rank test estimates whether observations in two
    groups are meaningfully similar to each other in terms of rank.
    The test also estimates whether such similarity can be reasonably
    expected in cases where the generating processes are identical
    given the amount of observations.

    The pair-wise rank comparisons between observations from two groups
    or random variables from two unknown distributions is described by
    a Binomial distribution where the observations in one group (here
    observations_a) being larger than in the other (observations_b)
    considered "successes" (and vice versa).
    Using the number of successes as alpha and the number of trials
    subtracted by the number of successes as beta, the uncertainty of 
    the parameter p of the Binomial distribution can be characterized
    using a Beta-distribution. Using a uniform uninformative prior 
    Beta(1, 1), the uncertainty of p is then characterized by
        p ~ Beta(successes + 1, n_trials - successes + 1).
    Given this formulation, we estimate how close the true parameter p
    (given the data) is to 0.5 with p_mass probability. If that distance
    is small enough for the application at hand, the two distributions
    are for the purpose at meaningfully similar in terms of rank.

    This function also estimates whether the number of observations in
    the two groups is sufficient recognize meaninful similarity with
    high probability.

    -If the latter test indicates that there is not enough observations, no
    conclusions can be done.
    -If the latter test shows that the amount of data should be sufficient
    to recognize a difference that is smaller than max_diff and the former
    test shows that the difference is larger, this is clear evidence to
    reject the similarity hypothesis.
    -If the former test shows that the two distributions are within max_diff,
    then this is strong evidence that the two distributions are functionally
    similar in terms of rank.

    Args:
    observations_a (list or numpy-like): 1-dim. array of observations from one process
    observations_b (list or numpy-like): 1-dim. array of observations from another process
    max_diff (float): The maximum difference to test for. Should be deduced from the
     use case where the difference takes meaning.
    p_mass (float): The desired confidence level, in ]0,1[.
    """
    # Both groups assumed to contain an equal amount of observations:
    text = "Unequal amount of observations in group a and b."
    assert len(observations_a) == len(observations_b), text

    # Counting pairwise differences.
    N_equal = np.sum(observations_a == observations_b)
    N_a_larger = np.sum(observations_a > observations_b)
    N_b_larger = np.sum(observations_a < observations_b)  # Should be N - N_a_larger (-N_equal)
    N_obs = len(observations_b)

    # Reference point:
    # How well should we expect to find similarity with N_obs observations
    # when the true parameter is 0.5?
    uncertainty_p_theoretical = np.zeros(shape=(MC_SAMPLES,))
    for i in range(MC_SAMPLES):
        # Sample one observation from a binomial distribution with N_obs trials.
        alpha_i = binom.rvs(N_obs, 0.5, size=1)  # #successes if true p=0.5
        beta_i = N_obs - alpha_i
        # Then sample from a beta-distribution characterized by the number of
        # successes (and a uniform uninformative prior, Beta(1,1)).
        # This results in one observation from the distribution characterizing
        # the uncertainty of the empiric estimate for p when the 
        # true value is 0.5.
        uncertainty_p_theoretical[i] = beta.rvs(a=alpha_i + 1, b=beta_i + 1, size=1)[0]
    # Given the generated observations, estimate the 95% credible interval:
    ci_lower, ci_upper = np.percentile(uncertainty_p_theoretical,
                                       [100*(1-p_mass)/2, 100*(p_mass + (1-p_mass)/2)])
    # Should be symmetric, wont be exactly due to numeric approach.
    # Calculate how close to 0.5 the result should be in 95% of cases:
    theoretical_p_mass_diff = np.mean([0.5 - ci_lower, ci_upper - 0.5])

    # Next, the observations provided are compared pairwise. If the generating
    # distributions are identical, the difference in rank should be tiny.
    # This does not guarantee that they are identical in other ways, though.
    uncertainty_of_p_empirical = beta.rvs(a=N_a_larger + N_equal / 2 + 1,
                                b=N_b_larger + N_equal / 2 + 1, size=MC_SAMPLES)
    # Given the samples above, estimate how often they are further away from 0.5
    # than max_diff:
    ci_empiric_lower, ci_empiric_upper = np.percentile(uncertainty_of_p_empirical,
                                                       [100*(1-p_mass)/2, 100*(p_mass + (1-p_mass)/2)])
    # Estimating the probability of the true parameter being less than max_diff from 0.5.
    # Note that the empirical p is not centered around 0.5.
    empirical_p_mass_similarity = np.mean(np.abs(0.5 - uncertainty_of_p_empirical) <= max_diff)

    # The two groups rank differently in less than x cases with 95% probability.
    # The probability that the two groups ranks more differently than 0.1 
    # is {}
    print("== PAIR-WISE RANK TEST ==")
    print_verbose("Comparing RVs pair-wise generates an RV following a Binomial distribution.")
    print_verbose("This test estimates whether the parameter for that Binomial distribution")
    print_verbose("is meaningfully different from 0.5 with high probability.")

    # print("The interval in which the empirical estimate of the parameter")
    # print("is found with {:.0%} probability when sampling {} observations".format(p_mass, N_obs))
    # print("\t0.5 \u00B1{:.3}".format(p_mass, theoretical_p_mass_diff))

    print_verbose("Theoretical {:.0%}-credible interval of empirical estimates of p".format(p_mass))
    print_verbose(u"from {} observations (when the true value is 0.5): [{:.3}, {:.3}]".format(N_obs, ci_lower, ci_upper))
    #print("\t Theoretical {:.0%}-credible interval of p = 0.5 \u00B1{:.3}".format(p_mass, theoretical_p_mass_diff))

    print("{:.0%}-credible interval of p given the observations: [{:.3}, {:.3}]".format(p_mass, ci_empiric_lower, ci_empiric_upper))
    # The interval in which we expect the true parameter to be given the data
    # 95%-credible interval in which we expect the true parameter to be given the data
    # p(p_empirical = 0.5 +- some diff) = 95%.

    success = False
    if theoretical_p_mass_diff > max_diff:
        # The test has little to no chance of discovering a difference as small
        text = "This test cannot discover a difference as small as {}".format(max_diff)
        text += "\nwith only {} observations in both groups.".format(N_obs)
        warnings.warn(text)
    elif empirical_p_mass_similarity < p_mass:
        # The ratio of ranks: (0.5 - empiric_mean_diff):(empiric_mean_diff + 0.5)
        text = "The probability that the true value of p differ by more than {:.3}".format(max_diff)
        text += "\nis larger than {:.0%}. The test does not support similarity.".format(1-p_mass)
        warnings.warn(text)
    elif empirical_p_mass_similarity > p_mass:
        success = True
        print(u"The probability of the true mean being in 0.5 \u00B1{:.3}: {:.0%}".format(max_diff, empirical_p_mass_similarity))
        print("STRONG EVIDENCE FOUND FOR MEANINGFUL SIMILARITY OF RANK!")
    return {'success': success,
            'empirical_similarity_mass': empirical_p_mass_similarity}


def bayesian_test_for_similarity_of_means(observations_a, observations_b,
                                          max_diff=0.01, p_mass=0.95):
    """
    Assuming CLT, the means will approximately follow normal distributions.
    -This assumes that the distributions behave "nicely", i.e. they are such
    that sample means and averages can be estimated.
    We are literally assuming that if we sample the same underlying distributions
    many times, in one sampling event the sample mean will be based on the sum
    of many observations which following CLT should approach a normal distribution.
    That sum divided by a constant N is still normally distributed.
    The variance of the first is (or std?)
        var (unknown?)
    but it is related to the variance of the sub-samples.
    Given our data, the sample average is our best estimate for the center of the normal
    distribution. Our best estimate for its variance, though is... 
    std_mean_A = std(A)/sqrt(n_A)
    std_mean_B = std(B)/sqrt(n_B)
    # Pooled standard deviation:
    std_diff = sqrt(std_mean_A**2/n_A + std_mean_B**2/n_B)

    Args:
    observations_a (list or numpy-like): 1-dim. array of observations from one process
    observations_b (list or numpy-like): 1-dim. array of observations from another process
    max_diff (float): Estimating how probable it is that the means of the two
    distributions are within +-max_diff from each other. The choice of max_diff is rather
    arbitrary.
    """
    print("== BAYESIAN TEST FOR SIMILARITY OF MEANS (BASED ON CLT) ==")
    n_a = len(observations_a)
    n_b = len(observations_b)

    # Means    
    mean_a = np.mean(observations_a)
    mean_b = np.mean(observations_b)
    mean_diff = mean_a - mean_b
    # Standard deviation of difference
    std_a = np.std(observations_a, ddof=1)
    std_b = np.std(observations_b, ddof=1)
    std_diff = np.sqrt(std_a**2/n_a + std_b**2/n_b)

    # Sample from the difference distribution:
    # Difference between two normals is normal
    # Note np.random.normal is defined with mean and standard deviation (not variance!).
    diff_samples = np.random.normal(mean_diff, std_diff, size=MC_SAMPLES)
    similar_n = np.sum(np.abs(diff_samples) <= max_diff)
    similarity_mass = similar_n / MC_SAMPLES

    # If the distributions of the means of the two distributions are the same,
    # the mean should equal zero.
    ci_lower, ci_upper = np.percentile(diff_samples, [100*(1-p_mass)/2, 100*(p_mass + (1-p_mass)/2)])
    # Is there a reason to assume that the difference actually is centered? Nope.
    ci_diff = np.mean(np.abs([mean_diff - ci_lower, ci_upper - mean_diff]))
    print(u"{:.0%}-credible interval for the mean: {:.3} \u00B1{:.3}".format(p_mass, mean_diff, ci_diff))
    print("Probability that the difference of the true means is smaller than {:.3}: {:.0%}".format(max_diff, similarity_mass))
    if similarity_mass > p_mass:
        # Test succeeded.
        success = True
        print("STRONG EVIDENCE FOUND FOR MEANINGFUL SIMILARITY OF MEAN!")
    else:
        success = False
        text = "The difference in means is with more than {:.0%} probability".format(1-p_mass)
        text += "\nlarger than {:.3}. This does not support similarity.".format(max_diff)
        warnings.warn(text)

    return {'success': success,
            'similarity_mass': similarity_mass}


def bootstrap_test_for_similarity(observations_a, observations_b, 
                                  max_diff=0.05, p_mass=0.95):
    """
    Bootstrap approach to estimate mean, variance, and skewness of distributions and
    further estimate how likely it is that the parameters of the two are within
    max_diff distance of each other.

    Args:
    observations_a (list or numpy-like): 1-dim. array of observations from one process
    observations_b (list or numpy-like): 1-dim. array of observations from another process
    max_diff (float): Estimating how probable it is that the means, the variances,
     and the skewnesses of the two distributions are within +-max_diff from each other.
     The choice of max_diff is rather arbitrary.
    """
    print("== BOOTSTRAP TEST FOR SIMILARITY OF MEANS, VARIANCES, AND SKEWNESS ==")
    n_obs = len(observations_a)
    # "Counters"
    bootstrap_means_a = np.zeros(shape=(BOOTSTRAP_SAMPLES,))
    bootstrap_means_b = np.zeros(shape=(BOOTSTRAP_SAMPLES,))
    bootstrap_variance_a = np.zeros(shape=(BOOTSTRAP_SAMPLES,))
    bootstrap_variance_b = np.zeros(shape=(BOOTSTRAP_SAMPLES,))
    bootstrap_skewness_a = np.zeros(shape=(BOOTSTRAP_SAMPLES,))
    bootstrap_skewness_b = np.zeros(shape=(BOOTSTRAP_SAMPLES,))
    for i in range(BOOTSTRAP_SAMPLES):
        # Resample without replacement as we are interested
        # in mean, variance, and skewness
        tmp_a = np.random.choice(observations_a, size=n_obs, replace=True)
        tmp_b = np.random.choice(observations_b, size=n_obs, replace=True)
        bootstrap_means_a[i] = np.mean(tmp_a)
        bootstrap_means_b[i] = np.mean(tmp_b)
        bootstrap_variance_a[i] = np.var(tmp_a, ddof=1)  # Unbiased estimate
        bootstrap_variance_b[i] = np.var(tmp_b, ddof=1)  # Unbiased estimate
        bootstrap_skewness_a[i] = skew(tmp_a)
        bootstrap_skewness_b[i] = skew(tmp_b)
    # Everything stored. Compare.
    # In how large fraction of the cases is the difference in means larger than max_diff:
    p_means_within_diff = np.mean(np.abs(bootstrap_means_a - bootstrap_means_b) < max_diff)
    p_var_within_diff = np.mean(np.abs(bootstrap_variance_a - bootstrap_variance_b) < max_diff)
    p_skew_within_diff = np.mean(np.abs(bootstrap_skewness_a - bootstrap_skewness_b) < max_diff)

    ci_means = np.percentile(bootstrap_means_a-bootstrap_means_b, [100*(1-p_mass)/2, 100*(p_mass + (1-p_mass)/2)])
    ci_var = np.percentile(bootstrap_variance_a - bootstrap_variance_b, [100*(1-p_mass)/2, 100*(p_mass + (1-p_mass)/2)])
    ci_skew = np.percentile(bootstrap_skewness_a - bootstrap_skewness_b, [100*(1-p_mass)/2, 100*(p_mass + (1-p_mass)/2)])

    # print(u"The means are with {:.0%} probability within \u00B1{:.3}.".format(p_means_within_diff, max_diff))
    # print(u"The variances are with {:.0%} probability within \u00B1{:.3}.".format(p_var_within_diff, max_diff))
    # print(u"The skewnesses are with {:.0%} probability within \u00B1{:.3}.".format(p_skew_within_diff, max_diff))

    print(u"The means are with {:.0%} probability within \u00B1{:.3} ({:.0%}-CI: [{:.3}, {:.3}]).".format(p_means_within_diff, max_diff, p_mass, ci_means[0], ci_means[1]))
    print(u"The variances are with {:.0%} probability within \u00B1{:.3} ({:.0%}-CI: [{:.3}, {:.3}]).".format(p_var_within_diff, max_diff, p_mass, ci_var[0], ci_var[1]))
    print(u"The skewness is with {:.0%} probability within \u00B1{:.3} ({:.0%}-CI: [{:.3}, {:.3}]).".format(p_skew_within_diff, max_diff, p_mass, ci_skew[0], ci_skew[1]))

    # Set binary flags:
    mean_diff_small = p_means_within_diff > p_mass
    var_diff_small = p_var_within_diff > p_mass
    skew_diff_small = p_skew_within_diff > p_mass
    success = False
    if mean_diff_small and var_diff_small and skew_diff_small:
        print("STRONG EVIDENCE FOUND FOR MEANINGFUL SIMILARITY OF MEAN, VARIANCE, AND SKEWNESS!")
        success = True
    elif mean_diff_small and var_diff_small and not skew_diff_small:
        text = "The means and variances are similar, but the SKEWNESS IS NOT."
        warnings.warn(text)
    elif mean_diff_small and not var_diff_small:
        text = "The means are similar, but the VARIANCES ARE NOT WITHIN SET BOUNDS."
        warnings.warn(text)
    return {'success': success, 
            'p_mean_diff_small': p_means_within_diff,
            'p_var_diff_small': p_var_within_diff,
            'p_skew_diff_small': p_skew_within_diff}


def normalized_wasserstein_distance(observations_a, observations_b,
                                    max_diff=0.01):
    """
    Wasserstein distance, a.k.a earth mover's distance
    An estimate of how much probability mass needs to be shifted times
    the distance to make one distribution identical to the other.
    This implementation adds normalization by pooled standard deviation
    on top of the scipy-implementation.

    The empirical estimate for this distance is calculated from
    the two groups of observations by calculating the distance
    between pairs created by matching the ordered samples in one
    group to the ordered samples in the other. The distance
    considered is the L1-distance (p=1).

    Args:
    observations_a (list or numpy-like): 1-dim. array of observations from one process
    observations_b (list or numpy-like): 1-dim. array of observations from another process
    max_diff (float): Estimating how probable it is that the means, the variances,
     and the skewnesses of the two distributions are within +-max_diff from each other.
     The choice of max_diff is rather arbitrary.
    """
    print("== WASSERSTEIN DISTANCE ==")
    wass_dist = wasserstein_distance(observations_a, observations_b)
    print("Wasserstein distance: {:.3}".format(wass_dist))
    pooled_std = np.sqrt((np.var(observations_a) + np.var(observations_b)) / 2)
    normalized_wass_dist = wass_dist / pooled_std
    print("Normalized Wasserstein distance: {:.3}".format(normalized_wass_dist))
    if normalized_wass_dist > max_diff:
        # Raise warning. It seems the two are not identical.
        # Claude (AI) suggest4ed that <0.1 is essentially identical, and 0.1-0.5 are "very close."
        text = "The Wasserstein distance is more than {:.3} pooled standard deviations.".format(max_diff)
        text += "\nThis indicates that the observations in the two groups are not from the same distributions."
        warnings.warn(text)
        success = False
    else:
        success = True
        print("STRONG EVIDENCE FOUND FOR MEANINGFUL SIMILARITY!")
    return {'success': success,
            'wasserstein_distance': wass_dist,
            'normalized_wasserstein_distance': normalized_wass_dist}
