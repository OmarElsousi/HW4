"""
This script demonstrates how to generate random rock diameters from a truncated log-normal distribution,
calculate their sample statistics, and examine the distribution of those statistics across multiple samples.

Key features of this script:
1. It accepts user inputs for distribution parameters and sample sizes, but falls back to default values if no input is given.

Dependencies:
- math: For various mathematical functions (exp, log, sqrt, etc.)
- random.random: For generating uniform random numbers in [0, 1)
- scipy.integrate.quad: For numerically integrating functions (PDF to get CDF values)
- scipy.optimize.fsolve: For numerically solving equations (inverse CDF)
- copy.deepcopy: For creating independent copies of objects if needed
"""

import math
from random import random as rnd
from scipy.integrate import quad
from scipy.optimize import fsolve
from copy import deepcopy as dc


def ln_PDF(D, mu, sig):
    """
    Compute the value of the standard log-normal PDF at a given diameter D.

    Parameters
    ----------
    D : float
        The diameter at which to evaluate the log-normal PDF.
    mu : float
        The mean of ln(D). This corresponds to the location parameter of the log-normal distribution.
    sig : float
        The standard deviation of ln(D). This corresponds to the scale parameter of the log-normal distribution.

    Returns
    -------
    float
        The value of the log-normal PDF at diameter D. If D = 0.0, returns 0.0 to avoid a singularity.

    Notes
    -----
    The log-normal PDF for a random variable D (D > 0) is given by:

        f(D) = (1 / [D * sig * sqrt(2π)]) * exp( - [ln(D) - mu]^2 / [2 * sig^2] )

    This function implements that formula directly.
    """
    if D == 0.0:
        # The PDF is not defined at D=0 in a strict sense, so return 0.0 to avoid numerical issues.
        return 0.0

    # Coefficient outside the exponential
    p = 1 / (D * sig * math.sqrt(2 * math.pi))

    # Exponential part
    exponent = -((math.log(D) - mu) ** 2) / (2 * sig ** 2)

    return p * math.exp(exponent)


def F_tlnpdf(D, mu, sig, D_Min, D_Max, F_DMax, F_DMin):
    """
    Compute the truncated CDF of the log-normal distribution at diameter D.

    Parameters
    ----------
    D : float
        The diameter at which to compute the truncated CDF.
    mu : float
        The mean of ln(D).
    sig : float
        The standard deviation of ln(D).
    D_Min : float
        The lower bound of the truncation.
    D_Max : float
        The upper bound of the truncation.
    F_DMax : float
        The value of the untruncated log-normal CDF evaluated at D_Max.
    F_DMin : float
        The value of the untruncated log-normal CDF evaluated at D_Min.

    Returns
    -------
    float
        The truncated CDF value at diameter D. If D is outside [D_Min, D_Max], returns 0.

    Notes
    -----
    The truncated CDF F_trunc(D) is defined as:

        F_trunc(D) = [F(D) - F(D_Min)] / [F(D_Max) - F(D_Min)]

    where F(D) is the untruncated log-normal CDF. Numerically, we approximate F(D) by integrating
    the PDF from 0 to D. Here, for efficiency, we only integrate from D_Min to D (since we know
    how to scale by the difference [F(D_Max) - F(D_Min)]).
    """
    # If D is outside the truncation range, the truncated CDF is 0 by definition in this script.
    if D > D_Max or D < D_Min:
        return 0

    # Numerically integrate the PDF over [D_Min, D], then normalize by the total mass in [D_Min, D_Max].
    # The factor (F_DMax - F_DMin) is the total untruncated CDF mass in the range [D_Min, D_Max].
    # We divide the integrand ln_PDF by that factor to scale it properly.
    P, _ = quad(lambda x: ln_PDF(x, mu, sig) / (F_DMax - F_DMin), D_Min, D)

    return P


def makeSample(ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin, N=100):
    """
    Generate a sample of rock diameters from a truncated log-normal distribution.

    Parameters
    ----------
    ln_Mean : float
        The mean of ln(D).
    ln_sig : float
        The standard deviation of ln(D).
    D_Min : float
        The lower bound of the truncation.
    D_Max : float
        The upper bound of the truncation.
    F_DMax : float
        The untruncated log-normal CDF value at D_Max.
    F_DMin : float
        The untruncated log-normal CDF value at D_Min.
    N : int, optional
        The number of samples (diameters) to generate, by default 100.

    Returns
    -------
    list of float
        A list of rock diameters (float) that respect the truncated log-normal distribution.

    Notes
    -----
    This function first generates N uniform random numbers in [0, 1). For each random number p,
    we invert the truncated CDF to find the corresponding diameter. Numerically, the inversion is done
    via fsolve on the function F_tlnpdf(D) - p = 0.
    """

    # Generate N uniform probabilities in [0, 1)
    probs = [rnd() for _ in range(N)]

    def safe_fsolve(p):
        """
        Find the diameter D that corresponds to the truncated CDF = p.

        Parameters
        ----------
        p : float
            A probability in [0, 1) to invert.

        Returns
        -------
        float
            The diameter D for which F_tlnpdf(D) equals p, if it exists in [D_Min, D_Max].
        """
        # Initial guess for diameter is the midpoint of [D_Min, D_Max].
        guess = (D_Min + D_Max) / 2

        # Attempt to solve F_tlnpdf(D) = p
        result = fsolve(lambda x: F_tlnpdf(x, ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin) - p, guess)

        # If the solution is out of bounds, try again with a different initial guess
        if not (D_Min <= result[0] <= D_Max):
            result = fsolve(lambda x: F_tlnpdf(x, ln_Mean, ln_sig, D_Min, D_Max, F_DMax, F_DMin) - p, D_Max)

        return result[0]

    # For each uniform probability p, invert to find the corresponding diameter
    diameters = [safe_fsolve(p) for p in probs]
    return diameters


def sampleStats(D):
    """
    Calculate the sample mean and sample variance of a list of values.

    Parameters
    ----------
    D : list of float
        The data sample (e.g., rock diameters) for which to compute statistics.

    Returns
    -------
    tuple
        (mean, variance) of the sample.

    Notes
    -----
    The mean is computed as the arithmetic average of the elements in D.
    The variance is computed using the sample variance formula:

        variance = [1 / (N - 1)] * Σ (xi - mean)^2

    where N is the sample size.
    """
    N = len(D)
    # Compute sample mean
    mean = sum(D) / N

    # Compute sample variance
    var = sum((d - mean) ** 2 for d in D) / (N - 1)

    return mean, var


def getFDMaxFDMin(mean_ln, sig_ln, D_Min, D_Max):
    """
    Compute the un-truncated log-normal CDF values at D_Min and D_Max by integrating the PDF.

    Parameters
    ----------
    mean_ln : float
        The mean of ln(D).
    sig_ln : float
        The standard deviation of ln(D).
    D_Min : float
        The lower diameter bound of interest.
    D_Max : float
        The upper diameter bound of interest.

    Returns
    -------
    tuple
        (F_DMin, F_DMax), where F_DMin is the untruncated CDF at D_Min, and F_DMax is the untruncated CDF at D_Max.

    Notes
    -----
    We integrate from 0 to D_Min and 0 to D_Max to get the untruncated log-normal CDF values.
    These values are then used to help define the truncated distribution in F_tlnpdf.
    """
    # Integrate the PDF from 0 to D_Max
    F_DMax, _ = quad(ln_PDF, 0, D_Max, args=(mean_ln, sig_ln))

    # Integrate the PDF from 0 to D_Min
    F_DMin, _ = quad(ln_PDF, 0, D_Min, args=(mean_ln, sig_ln))

    return F_DMin, F_DMax


def main():
    """
    The main driver function.

    Steps:
    1. Initialize default distribution parameters (mean_ln, sig_ln) and truncation bounds (D_Min, D_Max).
    2. Prompt the user for inputs, providing defaults if no input is given.
    3. Compute F_DMin and F_DMax for the untruncated log-normal distribution.
    4. Generate N_samples, each of size N_sampleSize, from the truncated log-normal distribution.
    5. For each sample, compute and print its mean and variance.
    6. Finally, compute and print the mean of all sample means and the variance of all sample means.

    This function demonstrates how repeated sampling can help you understand the behavior of
    sample statistics (like the mean and variance) drawn from a truncated distribution.
    """
    # Default parameters
    mean_ln = math.log(2)  # ln(2)
    sig_ln = 1             # Standard deviation of ln(D)
    D_Max = 1              # Upper truncation bound
    D_Min = 3.0 / 8.0      # Lower truncation bound
    N_samples = 11         # Number of samples to draw
    N_sampleSize = 100     # Size of each sample

    # Prompt user for inputs with defaults
    mean_ln = float(input(f"Mean of ln(D) for the pre-sieved rocks? (Ln(2.0)=0.693, where D is in inches): ") or mean_ln)
    sig_ln = float(input(f"Standard deviation of ln(D) for the pre-sieved rocks? (1.000): ") or sig_ln)
    D_Max = float(input(f"Large aperture size? (1.000): ") or D_Max)
    D_Min = float(input(f"Small aperture size? (0.375): ") or D_Min)
    N_samples = int(input(f"How many samples? (11): ") or N_samples)
    N_sampleSize = int(input(f"How many items in each sample? (100): ") or N_sampleSize)

    # Compute untruncated log-normal CDF at D_Min and D_Max
    F_DMin, F_DMax = getFDMaxFDMin(mean_ln, sig_ln, D_Min, D_Max)

    # Lists to hold samples and means
    Samples = []
    Means = []

    # Generate each sample, compute statistics, and store results
    for i in range(N_samples):
        sample = makeSample(mean_ln, sig_ln, D_Min, D_Max, F_DMax, F_DMin, N_sampleSize)
        Samples.append(sample)

        mean, var = sampleStats(sample)
        Means.append(mean)

        # Print sample-specific statistics
        print(f"Sample {i+1} -> mean = {mean:.3f}, variance = {var:.3f}")

    # Analyze the distribution of the sample means themselves
    mean_of_means, var_of_means = sampleStats(Means)
    print(f"\nMean of sampling means: {mean_of_means:.3f}")
    print(f"Variance of sampling means: {var_of_means:.6f}")


# Standard Python entry point: calls main() if the script is run directly.
if __name__ == '__main__':
    main()
