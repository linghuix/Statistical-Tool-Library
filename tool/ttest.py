
# The `environment.yml` file is used to manage and reproduce conda environments.
# 
# To create this file from your current conda environment:
# 1. Run `conda env export > environment.yml` in your terminal. This command exports
#    the specifications of the active conda environment to an `environment.yml` file.
# 
# To recreate the environment from an `environment.yml` file:
# 1. Run `conda env create -f environment.yml`. This will create a new conda environment
#    with the exact dependencies listed in the file.
# 
# Note: If the environment already exists and you want to update it, use:
#    `conda env update -f environment.yml`.


import numpy as np
from scipy.stats import ttest_rel, shapiro, wilcoxon
from statsmodels.stats.power import TTestPower, NormalIndPower
from scipy.stats import rankdata

def one_sample_t_test(data, test_value=0, alpha=0.05):
    """
    Perform a one-sample t-test to check if the sample mean is equal to a given value (test_value),
    and calculate the statistical power of the test. Also checks for normality of the data.

    Parameters:
    - data: Sample data (array-like).
    - test_value: The value to compare the sample mean to (default is 0).
    - alpha: Significance level (default is 0.05).

    Returns:
    - dict: A dictionary containing the t-statistic, p-value, reject_null, power, and normality test result.
    """
    # Calculate the sample mean and standard deviation
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    n = len(data)
    
    # Perform one-sample t-test (default is two-tailed)
    t_stat, p_value = stats.ttest_1samp(data, test_value)
    
    # Determine if we reject the null hypothesis
    reject_null = p_value < alpha
    
    # Calculate the effect size (Cohen's d)
    effect_size = (sample_mean - test_value) / sample_std
    
    # Perform power analysis using statsmodels
    power_analysis = TTestPower()
    power = power_analysis.solve_power(effect_size=effect_size, nobs=n, alpha=alpha, alternative='two-sided')
    
    # Perform normality check using Shapiro-Wilk test
    normality_stat, normality_p_value = stats.shapiro(data)
    normality_passed = normality_p_value > alpha  # Normality assumption is passed if p > alpha
    
    # Return results as a dictionary
    results = {
        "t_stat": t_stat,
        "p_value": p_value,
        "reject_null": reject_null,
        "power": power,
        "normality_stat": normality_stat,
        "normality_p_value": normality_p_value,
        "normality_passed": normality_passed
    }
    
    return results


def paired_t_test(x, y, alpha=0.05):
    """
    Perform a paired t-test with power analysis and check the assumptions.

    Parameters:
        x (array-like): First set of paired observations.
        y (array-like): Second set of paired observations.
        alpha (float): Significance level (default is 0.05).

    Returns:
        dict: Results of the paired t-test, power analysis, and suggestions if assumptions are violated.
    """
    # Convert inputs to numpy arrays for calculation
    x = np.array(x)
    y = np.array(y)

    # Check that the two inputs have the same length
    if len(x) != len(y):
        return {"error": "Input arrays must have the same length."}

    # Check that the inputs are not empty
    if len(x) == 0:
        return {"error": "Input arrays must not be empty."}

    # Compute the differences
    differences = x - y

    # Check if the data is continuous
    if not (np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number)):
        return {"error": "Inputs must be continuous numerical data."}

    # Check normality assumption using Shapiro-Wilk test
    stat, p_normality = shapiro(differences)
    normality_passed = p_normality > alpha

    # Paired t-test
    t_stat, p_value = ttest_rel(x, y)

    # Calculate the power of the test
    diff_mean = np.mean(differences)
    diff_std = np.std(differences, ddof=1)
    effect_size = diff_mean / diff_std if diff_std != 0 else 0  # Cohen's d for paired samples
    n = len(differences)

    # Use statsmodels for power analysis
    power_analysis = TTestPower()
    power = power_analysis.solve_power(effect_size=effect_size, nobs=n, alpha=alpha, alternative="two-sided")

    results = {
        "t_stat": t_stat,
        "p_value": p_value,
        "p_normality": p_normality,
        "normality_passed": normality_passed,
        "power": power,
        "effect_size": effect_size,
        "suggestion": None,
    }

    # If normality assumption is not met, suggest Wilcoxon signed-rank test
    if not normality_passed:
        results["suggestion"] = {
            "alternative_test": "Wilcoxon signed-rank test",
            "reason": "The differences are not normally distributed.",
        }

        # Perform Wilcoxon signed-rank test as an alternative
        wilcoxon_stat, wilcoxon_p = wilcoxon(differences)
        results["wilcoxon_test"] = {
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p_value": wilcoxon_p,
        }

    else:
        # Check for outliers (simple method using z-scores)
        outliers = np.where(
            (differences < diff_mean - 3 * diff_std) | 
            (differences > diff_mean + 3 * diff_std)
        )[0]

        if np.any(outliers):
            return {
                "error": "Extreme outliers detected in the differences.",
                "outliers": outliers,
                "suggestion": "Consider robust statistical methods, or handle outliers appropriately (e.g., using Winsorization).",
                "differences": differences,
            }

        results["suggestion"] = "The paired t-test assumptions are met."

    return results


def compare_twodata(data, name):
    """

    Compares the 'First' and 'Last' columns of multiple cases in the input data.
    Performs normality tests, variance equality tests, and selects the appropriate statistical test.
    
    Parameters:
        data (dict): Dictionary where each key is a case name and the values are dictionaries
                     with 'First' and 'Last' columns as lists of numeric values.
                     
    Returns:
        dict: Results of statistical tests for each case (Shapiro-Wilk, Levene, t-test/Wilcoxon).
    """

    print("-----------------", name, "-----------------")

    # Perform analysis for each case
    for case, values in data.items():
        first = np.array(values["First"])
        last = np.array(values["Last"])

        res = paired_t_test(first, last)

        print(res)

    return {
        'case': name,
        'res':res,
    }


Midstance = {
    "exoMidstanceMoreAffected": {
        "First": [37.0, 10.6, 28.3, 28.0, 21.9, 17.9, 3.9],
        "Last": [29.1, 11.3, 23.1, 22.9, 14.2, 13.2, -3.8]
    },
    "exoMidstanceLessAffected": {
        "First": [31.0, 2.8, 14.8, 24.2, 15.3, -1.4, 2.4],
        "Last": [33.4, -3.4, 18.5, 19.3, 11.5, 5.1, 0.4]
    },
    "baseMidstanceMoreAffected": {
        "First": [31.4, 17.8, 26.5, 28.3, 42.6, 30.8, 23.8],
        "Last": [29.7, 19.4, 31.4, 30.7, 25.2, 26.5, 35.8]
    },
    "baseMidstanceLessAffected": {
        "First": [28.6, 6.1, 17.4, 25.2, 29.8, -2.6, 7.2],
        "Last": [27.2, 1.8, 24.1, 27.4, 20.1, 2.5, 12.5]
    }
}

# Call the function with the data
results = compare_twodata(Midstance, 'Midstance')


Initial = {
    "exoInitialMoreAffected": {
        "First": [39.7, 25.0, 34.1, 33.8, 35.4, 32.9, 33.6],
        "Last": [31.2, 22.8, 33.5, 30.1, 27.5, 29.1, 41.1]
    },
    "exoInitialLessAffected": {
        "First": [38.7, 8.7, 26.5, 37.6, 33.4, 24.8, 27.7],
        "Last": [32.1, 11.1, 31.2, 26.2, 30.5, 34.5, 16.4]
    },
    "baseInitialMoreAffected": {
        "First": [30.7, 29.0, 30.5, 29.2, 51.7, 49.7, 48.9],
        "Last": [31.6, 30.1, 36.8, 33.7, 38.2, 46.1, 50.7]
    },
    "baseInitialLessAffected": {
        "First": [31.0, 21.0, 25.2, 26.7, 44.2, 27.6, 26.8],
        "Last": [32.5, 21.3, 33.9, 31.1, 28.4, 37.2, 27.8]
    }
}

# Call the function with the data
results = compare_twodata(Initial, 'Initial')

stepLength = {
    "case1": {
        "First": [0.05, 0.45, 0.18, 0.25, 0.22, 0.41, 0.28],
        "Last": [0.09, 0.43, 0.35, 0.45, 0.28, 0.49, 0.34]
    },
    "case2": {
        "First": [0.19, 0.39, 0.34, 0.47, 0.25, 0.46, 0.29],
        "Last": [0.35, 0.43, 0.39, 0.57, 0.29, 0.49, 0.17]
    },
    "case3": {
        "First": [0.30, 0.50, 0.30, 0.40, 0.30, 0.40, 0.40],
        "Last": [0.30, 0.60, 0.30, 0.50, 0.30, 0.30, 0.30]
    },
    "case4": {
        "First": [0.40, 0.50, 0.40, 0.60, 0.30, 0.40, 0.40],
        "Last": [0.40, 0.50, 0.40, 0.60, 0.30, 0.40, 0.30]
    }
}
results = compare_twodata(stepLength, 'stepLength')


gaitSpeed = {
    "case1": {
        "First": [0.16, 0.66, 0.48, 0.78, 0.42, 0.88, 0.52],
        "Last": [0.37, 0.75, 0.73, 1.04, 0.57, 1.28, 0.40]
    },
    "case2": {
        "First": [0.60, 1.00, 0.60, 1.00, 0.50, 0.80, 0.70],
        "Last": [0.60, 0.90, 0.60, 1.10, 0.60, 0.80, 0.50]
    }
}
results = compare_twodata(gaitSpeed, 'gaitSpeed')


