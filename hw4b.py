"""
File name: hw4b.py

Description:
    This script illustrates how to plot the PDF (probability density function) and CDF
    (cumulative distribution function) for a truncated log-normal distribution over a
    specified domain [dmin, dmax]. The code parallels the approach in HW4a (Gaussian case)
    but replaces the normal distribution with a log-normal that has been truncated to
    [dmin, dmax].

    We:
      1) Prompt (or hard-code) the user for log-normal parameters mu, sigma,
         and the bounds dmin, dmax.
      2) Construct the truncated log-normal PDF and CDF using custom functions.
      3) Choose a cutoff point D_cut = dmin + 0.75*(dmax - dmin) and compute the probability
         P(D < D_cut).
      4) Plot the PDF (upper panel) and CDF (lower panel) in a two-panel figure,
         shading the area up to D_cut on the PDF plot and highlighting the corresponding
         point on the CDF.

Author:
    Your Name, Date

Usage:
    python hw4b.py

Dependencies:
    - numpy
    - matplotlib
    - scipy.stats

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def truncated_lognorm_pdf(x, mu, sigma, dmin, dmax):
    """
    Compute the truncated log-normal PDF over [dmin, dmax].

    The standard log-normal PDF for random variable D is:
        f_LN(D) = (1 / (D * sigma * sqrt(2 pi))) * exp( - (ln(D) - mu)^2 / (2 sigma^2) ),
        for D > 0.

    The truncated version on [dmin, dmax] is:
        f_TLN(D) = f_LN(D) / Z   if  dmin <= D <= dmax,
                    0           otherwise,
    where
        Z = F_LN(dmax) - F_LN(dmin)
    is the normalization factor (the difference between the untruncated CDF at dmax
    and dmin).

    Parameters
    ----------
    x : np.ndarray
        Array of D-values at which to evaluate the truncated PDF.
    mu : float
        Mean (mu) of the underlying log(D) distribution.
    sigma : float
        Standard deviation (sigma) of the underlying log(D) distribution.
    dmin : float
        Lower bound of truncation.
    dmax : float
        Upper bound of truncation.

    Returns
    -------
    np.ndarray
        Array of truncated log-normal PDF values at the points in x.
    """

    ln_dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    Z = ln_dist.cdf(dmax) - ln_dist.cdf(dmin)
    pdf_untrunc = ln_dist.pdf(x)
    pdf_untrunc[(x < dmin) | (x > dmax)] = 0.0
    return pdf_untrunc / Z

def truncated_lognorm_cdf(x, mu, sigma, dmin, dmax):
    """
    Compute the truncated log-normal CDF over [dmin, dmax].

    The untruncated log-normal CDF is:
        F_LN(D) = P(X <= D) for a log-normal random variable X.

    The truncated version on [dmin, dmax] is:
        F_TLN(D) = [F_LN(D) - F_LN(dmin)] / [F_LN(dmax) - F_LN(dmin)],
                    for dmin <= D <= dmax,
                  0  for D < dmin,
                  1  for D > dmax.

    Parameters
    ----------
    x : np.ndarray
        Array of D-values at which to evaluate the truncated CDF.
    mu : float
        Mean (mu) of the underlying log(D) distribution.
    sigma : float
        Standard deviation (sigma) of the underlying log(D) distribution.
    dmin : float
        Lower bound of truncation.
    dmax : float
        Upper bound of truncation.

    Returns
    -------
    np.ndarray
        Array of truncated log-normal CDF values at the points in x.
    """

    ln_dist = stats.lognorm(s=sigma, scale=np.exp(mu))
    Z = ln_dist.cdf(dmax) - ln_dist.cdf(dmin)
    cdf_untrunc = ln_dist.cdf(x)
    cdf_trunc = (cdf_untrunc - ln_dist.cdf(dmin)) / Z
    cdf_trunc[x < dmin] = 0.0
    cdf_trunc[x > dmax] = 1.0
    return cdf_trunc

def main():
    """
    Main driver function that:
      1) Specifies (or prompts for) mu, sigma, dmin, dmax.
      2) Chooses D_cut as the 75% span between dmin and dmax.
      3) Computes truncated log-normal PDF and CDF arrays for plotting.
      4) Fills the region up to D_cut in the PDF plot.
      5) Marks the corresponding point in the CDF plot.
      6) Shows the final figure with two stacked subplots.

    Returns
    -------
    None
    """

    mu = 0.69
    sigma = 1.00
    dmin = 0.047
    dmax = 0.244
    D_cut = dmin + 0.75 * (dmax - dmin)

    Dvals = np.linspace(dmin, dmax, 500)
    fvals = truncated_lognorm_pdf(Dvals, mu, sigma, dmin, dmax)
    Fvals = truncated_lognorm_cdf(Dvals, mu, sigma, dmin, dmax)
    p_val = truncated_lognorm_cdf(np.array([D_cut]), mu, sigma, dmin, dmax)[0]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax_pdf = axs[0]
    ax_cdf = axs[1]

    ax_pdf.plot(Dvals, fvals, color='blue')
    ax_pdf.set_ylabel('f(D)', size=12)
    ax_pdf.set_xlim(dmin, dmax)
    ax_pdf.set_ylim(0, fvals.max() * 1.1)

    fill_mask = (Dvals >= dmin) & (Dvals <= D_cut)
    ax_pdf.fill_between(Dvals[fill_mask], fvals[fill_mask], color='grey', alpha=0.4)

    text_x = dmin + 0.05 * (dmax - dmin)
    text_y = 0.6 * fvals.max()
    ax_pdf.text(
        text_x, text_y,
        r'$f(D) = \frac{1}{D \sigma \sqrt{2\pi}} \exp \left[-\frac{(\ln D - \mu)^2}{2\sigma^2}\right] / \text{(norm. const.)}$'
    )

    arrow_x = dmin + 0.5 * (D_cut - dmin)
    arrow_y = 0.5 * truncated_lognorm_pdf(np.array([arrow_x]), mu, sigma, dmin, dmax)[0]
    annotation_str = f'P(D < {D_cut:.3f} | TLN) = {p_val:.2f}'
    ax_pdf.annotate(
        annotation_str,
        xy=(arrow_x, arrow_y),
        xytext=(text_x, 0.4 * text_y),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3'),
        size=8
    )

    ax_cdf.plot(Dvals, Fvals, color='blue')
    ax_cdf.set_ylabel(r'$\Phi(D)=\int_{D_{\min}}^{\,D} f(D)\,dD$', size=12)
    ax_cdf.set_xlabel('D')
    ax_cdf.set_ylim(0, 1)

    ax_cdf.plot(D_cut, p_val, 'o', markerfacecolor='white', markeredgecolor='red')
    ax_cdf.hlines(p_val, dmin, D_cut, color='black', linewidth=1)
    ax_cdf.vlines(D_cut, 0, p_val, color='black', linewidth=1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
