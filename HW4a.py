"""
This script simulates drawing rock diameters from a truncated log-normal distribution.
It calculates the probability density function (PDF) of the distribution, and then uses numerical methods
from the SciPy library to determine random samples and analyze their means and variances.

The main entry point runs a loop to create multiple samples of rock diameters,
prints the mean and variance for each sample, and then outputs statistics of the sampling distribution.

Detailed comments and docstrings are provided for beginners, with step-by-step instructions.
"""

import math  # Provides mathematical functions such as exp, log, sqrt, and constants like pi
from random import random as rnd  # Used to generate uniformly distributed random numbers
from scipy.integrate import quad  # Integrates a function numerically over a given interval
from scipy.optimize import fsolve  # Finds roots of equations by numerical methods
from copy import deepcopy as dc  # Used if you need to create independent copies of objects


def ln_PDF(D, mu, sig):
    """
    Computes the value of the standard log-normal PDF at a given diameter D.

    Step-by-step:
    1. Check if D is 0. If yes, return 0 to avoid log(0).
    2. Calculate the leading factor: 1 / (D * sig * sqrt(2*pi)).
    3. Compute the exponent: -((ln(D) - mu)^2) / (2 * sig^2).
    4. Return the product of the leading factor and exp(exponent).

    Parameters:
    -----------
    D : float
        The diameter value at which we want to compute the PDF.
    mu : float
        The mean of the natural logarithm of the diameter.
    sig : float
        The standard deviation of the natural logarithm of the diameter.

    Returns:
    --------
    float
        The value of the log-normal PDF at diameter D.
    """
    if D == 0.0:
        return 0.0
    p = 1 / (D * sig * math.sqrt(2 * math.pi))
    _exp = -((math.log(D) - mu) ** 2) / (2 * sig ** 2)
    return p * math.exp(_exp)


def F_tlnpdf(D, mu, sig, D_Min, D_Max, F_DMax, F_DMin):
    """
    Computes the cumulative distribution function (CDF) of the truncated log-normal distribution
    from the lower bound D_Min up to the current value D.

    Step-by-step:
    1. If D is outside [D_Min, D_Max], return 0, because the truncated distribution is 0 there.
    2. Use quad() to integrate the truncated PDF from D_Min to D.
       The truncated PDF = ln_PDF(D) / (F_DMax - F_DMin).
    3. Return the cumulative probability P.

    Parameters:
    -----------
    D : float
        The diameter up to which we want the truncated CDF.
    mu : float
        The mean of ln(D) for the log-normal distribution.
    sig : float
        The standard deviation of ln(D) for the log-normal distribution.
    D_Min : float
        The lower bound of the truncated distribution.
    D_Max : float
        The upper bound of the truncated distribution.
    F_DMax : float
        CDF of the un-truncated log-normal at D_Max.
    F_DMin : float
        CDF of the un-truncated log-normal at D_Min.

    Returns:
    --------
    float
        The truncated CDF value between D_Min and D.
    """
    if D > D_Max or D < D_Min:
        return 0
    P, _ = quad(lambda x: ln_PDF(x, mu, sig) / (F_DMax - F_DMin), D_Min, D)
    return P


def makeSample(ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin, N=100):
    """
    Generates a sample of rock diameters from a truncated log-normal distribution.

    Step-by-step:
    1. Create a list of N random probabilities in [0, 1].
    2. For each probability p, find the diameter D such that the truncated CDF = p.
       - This is done by using a helper function safe_fsolve() which calls fsolve on F_tlnpdf(D) - p = 0.
    3. Collect these diameters into a list and return.

    Parameters:
    -----------
    ln_Mean : float
        The mean of ln(D) for the log-normal distribution.
    ln_sig : float
        The standard deviation of ln(D) for the log-normal distribution.
    D_Min : float
        The lower bound of the truncated distribution.
    D_Max : float
        The upper bound of the truncated distribution.
    F_DMax : float
        CDF of the un-truncated log-normal at D_Max.
    F_DMin : float
        CDF of the un-truncated log-normal at D_Min.
    N : int, optional
        The number of samples (default is 100).

    Returns:
    --------
    list
        A list of length N containing sampled diameters from the truncated distribution.
    """
    probs = [rnd() for _ in range(N)]

    def safe_fsolve(p):
        """
        Finds the diameter D that corresponds to the truncated CDF = p.

        Step-by-step:
        1. Begin with an initial guess = (D_Min + D_Max) / 2.
        2. Call fsolve to solve F_tlnpdf(D) - p = 0.
        3. If the returned solution is out of [D_Min, D_Max], try again with D_Max.
        4. Return the final diameter value.

        Parameters:
        -----------
        p : float
            A probability in [0, 1] to invert the truncated distribution.
        """
        guess = (D_Min + D_Max) / 2
        result = fsolve(lambda x: F_tlnpdf(x, ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin) - p, guess)
        if not (D_Min <= result[0] <= D_Max):
            result = fsolve(lambda x: F_tlnpdf(x, ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin) - p, D_Max)
        return result[0]

    d_s = [safe_fsolve(p) for p in probs]
    return d_s


def sampleStats(D):
    """
    Calculates the sample mean and sample variance of a list of values.

    Step-by-step:
    1. Let N be the size of the list.
    2. Calculate the mean = sum of values / N.
    3. Calculate the sample variance with denominator (N-1) for an unbiased estimate.
    4. Return a tuple (mean, variance).

    Parameters:
    -----------
    D : list
        A list of numeric values (in this context, rock diameters).

    Returns:
    --------
    tuple
        A tuple of (mean, variance) of the given list.
    """
    N = len(D)
    mean = sum(D) / N
    var = sum((d - mean) ** 2 for d in D) / (N - 1)
    return mean, var


def getFDMaxFDMin(mean_ln, sig_ln, D_Min, D_Max):
    """
    Computes the un-truncated log-normal CDF values at D_Min and D_Max by integrating
    the standard log-normal PDF with SciPy's quad function.

    Step-by-step:
    1. Integrate ln_PDF(x, mean_ln, sig_ln) from x=0 to x=D_Max, store as F_DMax.
    2. Integrate ln_PDF(x, mean_ln, sig_ln) from x=0 to x=D_Min, store as F_DMin.
    3. Return (F_DMin, F_DMax).

    Parameters:
    -----------
    mean_ln : float
        The mean of ln(D) for the log-normal distribution.
    sig_ln : float
        The standard deviation of ln(D) for the log-normal distribution.
    D_Min : float
        The lower bound of the truncated distribution.
    D_Max : float
        The upper bound of the truncated distribution.

    Returns:
    --------
    tuple
        (F_DMin, F_DMax) where both are floats representing the un-truncated log-normal CDF
        at D_Min and D_Max respectively.
    """
    F_DMax, _ = quad(ln_PDF, 0, D_Max, args=(mean_ln, sig_ln))
    F_DMin, _ = quad(ln_PDF, 0, D_Min, args=(mean_ln, sig_ln))
    return F_DMin, F_DMax


def main():
    """
    The main driver function. It initializes default parameters for the log-normal distribution
    and truncated bounds, computes relevant CDF values, then draws multiple samples.
    Finally, it prints out the mean and variance of each sample, and of the sampling distribution.

    Step-by-step:
    1. Set default values for mean_ln (log(2)), sig_ln=1, D_Max=1, D_Min=3/8, N_samples=11, N_sampleSize=100.
    2. Use getFDMaxFDMin() to compute F_DMin, F_DMax for the un-truncated distribution.
    3. Create empty lists Samples, Means to store data.
    4. Loop over N_samples times:
       a. Generate a sample using makeSample().
       b. Compute mean and variance with sampleStats().
       c. Print those stats and store mean in Means.
    5. After the loop, compute the mean and variance of Means.
    6. Print those final statistics.

    Returns:
    --------
    None
        Prints results to the console.
    """
    mean_ln = math.log(2)
    sig_ln = 1
    D_Max = 1
    D_Min = 3.0 / 8.0
    N_samples = 11
    N_sampleSize = 100

    F_DMin, F_DMax = getFDMaxFDMin(mean_ln, sig_ln, D_Min, D_Max)

    Samples = []
    Means = []

    for _ in range(N_samples):
        sample = makeSample(mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize)
        Samples.append(sample)
        mean, var = sampleStats(sample)
        Means.append(mean)
        print(f"Sample mean = {mean:.3f}, variance = {var:.3f}")

    mean_of_means, var_of_means = sampleStats(Means)
    print(f"Mean of sampling mean: {mean_of_means:.3f}")
    print(f"Variance of sampling mean: {var_of_means:.6f}")

# The standard entry point in Python.
if __name__ == '__main__':
    main()
