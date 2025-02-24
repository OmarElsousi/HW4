"""
This script generates graphs for the Truncated Log-Normal Probability Density Function (PDF)
and Cumulative Distribution Function (CDF). It illustrates the relationship between the
PDF and CDF, highlighting the area under the curve corresponding to a specific probability.

# region Imports
# Import necessary libraries for numerical operations and plotting
import numpy as np
import matplotlib.pyplot as plt
from math import log
import sys

# Add the directory containing numericalMethods and X1SP25_1 to the Python path
sys.path.append('/mnt/data')

# Import functions from X1SP25_1.py for truncated log-normal calculations
from X1SP25_1 import getFDMaxFDMin, tln_PDF, F_tlnpdf


# endregion

# region Input Function

def get_input(prompt, default):
    """
    Prompts the user for input with a default value. Accepts up to three decimal places.

    Step-by-Step:
    1. Display a prompt message to the user, showing the default value.
    2. If the user provides input, try to convert it to a float and round to three decimal places.
    3. If the input is invalid or left blank, use the default value.
    4. Return the validated user input or default value.

    Parameters:
    -----------
    prompt : str
        The message to display to the user.
    default : float
        The default value to use if no input is provided or if the input is invalid.

    Returns:
    --------
    float
        The user input value rounded to three decimal places or the default value.

    Example:
    --------
    >>> mean_ln = get_input("Enter mean (default 0.690): ", 0.690)
    >>> # If user inputs 0.75, the returned value will be 0.750
    >>> # If user presses Enter, the returned value will be 0.690
    """
    while True:
        try:
            # Prompt user for input and round to three decimal places
            value = input(prompt)
            # If valid input is provided, convert to float and round
            # If input is empty, use the default value
            return round(float(value), 3) if value else default
        except ValueError:
            # Handle invalid input and show error message
            print("Invalid input. Please enter a valid number.")


# endregion

# region User Inputs
"""
This section handles user inputs for the parameters of the pre-sieved log-normal 
distribution and the sieved truncated log-normal distribution.

Parameters:
- mean_ln: Mean of the natural logarithm of the rock diameters.
- sig_ln: Standard deviation of the natural logarithm of the rock diameters.
- D_Min: Minimum diameter corresponding to the small aperture size.
- D_Max: Maximum diameter corresponding to the large aperture size.

Defaults:
- mean_ln = 0.690
- sig_ln = 1.000
- D_Min = 0.375
- D_Max = 1.000

These defaults are chosen based on standard examples for truncated log-normal distributions.
"""
# Get user inputs with default values
mean_ln = get_input("Enter mean of ln(D) for pre-sieved rocks (default 0.693): ", 0.693)
sig_ln = get_input("Enter standard deviation of ln(D) (default 1.000): ", 1.000)
D_Min = get_input("Enter small aperture size D_Min (default .375): ", 0.375)
D_Max = get_input("Enter large aperture size D_Max (default 1): ", 1.000)
# endregion

# region Calculations
"""
This section calculates the necessary values for plotting the Truncated Log-Normal 
Probability Density Function (PDF) and Cumulative Distribution Function (CDF).

Calculations include:
1. F_DMin and F_DMax: These are the cumulative distribution function values at D_Min 
   and D_Max, which normalize the truncated log-normal distribution.
2. PDF and CDF: Using numpy arrays, the truncated log-normal PDF and CDF are calculated
   for a range of diameters between D_Min and D_Max.
3. Shading Area: The shaded area corresponds to the cumulative probability up to
   D_fill, which is calculated as 75% of the range between D_Min and D_Max.
"""
# Calculate F_DMin and F_DMax using the imported function
F_DMin, F_DMax = getFDMaxFDMin((mean_ln, sig_ln, D_Min, D_Max))

# Define the range for D using numpy's linspace function
D = np.linspace(D_Min, D_Max, 500)

# Calculate the truncated log-normal PDF and CDF for each value in D
pdf = np.array([tln_PDF((d, mean_ln, sig_ln, F_DMin, F_DMax)) for d in D])
cdf = np.array([F_tlnpdf((mean_ln, sig_ln, D_Min, D_Max, d, F_DMax, F_DMin)) for d in D])

# Determine the point of integration for shading (75% of the way between D_Min and D_Max)
D_fill = D_Min + (D_Max - D_Min) * 0.75
D_shade = D[D <= D_fill]
pdf_shade = pdf[D <= D_fill]
F_fill = F_tlnpdf((mean_ln, sig_ln, D_Min, D_Max, D_fill, F_DMax, F_DMin))
# endregion

# region Plotting
"""
This section creates the plots for:
1. Truncated Log-Normal PDF with shaded area showing cumulative probability.
2. Truncated Log-Normal CDF with markers and lines showing the relationship to the PDF.

The graphs include:
- Axis labels
- Grid lines for better visualization
- Annotations explaining the shaded area and mathematical equations
"""
# Create the plot with two subplots (PDF and CDF)
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Plot the PDF
ax[0].plot(D, pdf, color='blue', label='Truncated Log-Normal PDF')
ax[0].fill_between(D_shade, pdf_shade, color='grey', alpha=0.7)
ax[0].set_ylabel('f(D)', fontsize=12)
ax[0].set_ylim(0, max(pdf) * 1.1)

# Display the equation for the log-normal PDF
ax[0].text(D_Min + 0.05, max(pdf) * 0.75,
           r'$f(D) = \frac{1}{D\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{\ln(D)-\mu}{\sigma}\right)^2}$',
           fontsize=12)

# Annotation for shaded area showing cumulative probability
ax[0].annotate(
    r'$P(D<%0.3f|TLN(%0.3f,%0.3f,%0.3f, %0.3f))=%0.3f$' % (D_fill, mean_ln, sig_ln, F_DMin, F_DMax, F_fill),
    xy=(D_fill, max(pdf) * 0.5), xytext=(D_Min + 0.1, max(pdf) * 0.6),
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    fontsize=10
)

ax[0].grid(True)

# Plot the CDF
ax[1].plot(D, cdf, label='Truncated Log-Normal CDF', color='blue')
ax[1].set_xlabel('x', fontsize=12)
ax[1].set_ylabel(r'$\Phi(x)=\int_{D_{min}}^{x} f(D) dD$', fontsize=12)
ax[1].plot(D_fill, F_fill, 'o', markerfacecolor='white', markeredgecolor='red')
ax[1].hlines(F_fill, D_Min, D_fill, color='black', linewidth=1)
ax[1].vlines(D_fill, 0, F_fill, color='black', linewidth=1)
ax[1].grid(True)

plt.tight_layout()
plt.show()
# endregion