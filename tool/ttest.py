
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
from scipy import stats
from scipy.stats import ttest_rel, shapiro, wilcoxon
from statsmodels.stats.power import TTestPower, TTestIndPower
from scipy.stats import ttest_ind, levene, shapiro, mannwhitneyu
import testTool

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


def independent_t_test(x, y, alpha=0.05):
    """
    Perform an independent two-sample t-test with power analysis and check the assumptions.

    Parameters:
        x (array-like): First sample of observations.
        y (array-like): Second sample of observations.
        alpha (float): Significance level (default is 0.05).

    Returns:
        dict: Results of the independent t-test, power analysis, and suggestions if assumptions are violated.
    """
    # Convert inputs to numpy arrays for calculation
    x = np.array(x)
    y = np.array(y)

    # Check that inputs are not empty
    if len(x) == 0 or len(y) == 0:
        return {"error": "Input arrays must not be empty."}

    # Check if the data is continuous
    if not (np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number)):
        return {"error": "Inputs must be continuous numerical data."}

    # Normality checks using Shapiro-Wilk test
    stat_x, p_normality_x = shapiro(x)
    stat_y, p_normality_y = shapiro(y)
    normality_x_passed = p_normality_x > alpha
    normality_y_passed = p_normality_y > alpha

    # Equal variance check using Levene's test
    stat_levene, p_levene = levene(x, y)
    equal_variance = p_levene > alpha
    # print('p_levene', p_levene, equal_variance)

    # Perform the independent t-test
    t_stat, p_value = ttest_ind(x, y, equal_var=equal_variance)

    # Calculate effect size (Cohen's d)
    mean_diff = np.mean(x) - np.mean(y)
    pooled_std = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
    effect_size = mean_diff / pooled_std if pooled_std != 0 else 0   # keep sign

    # Power analysis
    n1, n2 = len(x), len(y)
    power_analysis = TTestIndPower()
    power = power_analysis.solve_power(effect_size=effect_size, nobs1=n1, ratio=n2 / n1, alpha=alpha, alternative="two-sided")

    results = {
        "t_stat": t_stat,
        "p_value": p_value,
        "p_normality_x": p_normality_x,
        "p_normality_y": p_normality_y,
        "normality_x_passed": normality_x_passed,
        "normality_y_passed": normality_y_passed,
        "p_levene": p_levene,
        "equal_variance": equal_variance,
        "power": power,
        "effect_size": effect_size,
        "suggestion": None,
    }

    # If normality is not met for either group, suggest Mann-Whitney U test
    if not (normality_x_passed and normality_y_passed):
        results["suggestion"] = {
            "alternative_test": "Mann-Whitney U test",
            "reason": "At least one of the samples is not normally distributed.",
        }

        # Perform Mann-Whitney U test
        mw_stat, mw_p = mannwhitneyu(x, y, alternative="two-sided")
        results["mann_whitney_test"] = {
            "mw_stat": mw_stat,
            "mw_p_value": mw_p,
        }

    return results


def test_independent_t_test():
    """
    Test the independent_t_test function with various scenarios.
    """
    # Test case 1: Normal data with equal variances
    group1 = [2.1, 2.5, 3.6, 3.8, 2.9]
    group2 = [3.1, 3.4, 4.0, 4.1, 3.8]
    results = independent_t_test(group1, group2)
    print(results["normality_x_passed"])
    assert "t_stat" in results, "Missing t_stat in results"
    assert "p_value" in results, "Missing p_value in results"
    assert "equal_variance" in results, "Missing equal_variance in results"
    assert results["normality_x_passed"] is True, "Normality check failed for group1"
    assert results["normality_y_passed"] is True, "Normality check failed for group2"

    print("Test case 1 passed.")

    # Test case 2: Data with unequal variances
    group3 = [2.1, 2.2, 2.3, 2.4, 2.5]
    group4 = [11, 12, 13, 14, 15]
    results = independent_t_test(group3, group4)
    print("Debugging equal_variance:", results["effect_size"] )
    assert bool(results["equal_variance"]) is False, "Levene's test failed to detect unequal variances"
    assert abs(results["effect_size"]) > 5, "Effect size should be large for large differences"

    print("Test case 2 passed.")

    # Test case 3: Normality assumption violated
    group5 = [1, 1, 1, 1, 50]  # Highly skewed data
    group6 = [2, 2, 2, 2, 2]
    results = independent_t_test(group5, group6)

    assert "suggestion" in results, "Suggestion for non-normal data is missing"
    assert results["suggestion"]["alternative_test"] == "Mann-Whitney U test", \
        "Alternative test suggestion should be Mann-Whitney U test"

    print("Test case 3 passed.")

    # Test case 4: Empty input arrays
    group7 = []
    group8 = [1, 2, 3]
    results = independent_t_test(group7, group8)

    assert "error" in results, "Empty input arrays should return an error"
    assert results["error"] == "Input arrays must not be empty.", "Error message for empty arrays is incorrect"

    print("Test case 4 passed.")

    # Test case 5: Non-numerical data
    group9 = ["a", "b", "c"]
    group10 = [1, 2, 3]
    results = independent_t_test(group9, group10)

    assert "error" in results, "Non-numerical data should return an error"
    assert results["error"] == "Inputs must be continuous numerical data.", "Error message for non-numerical data is incorrect"

    print("Test case 5 passed.")

    d1 = [0.5, 8.0, 8.2, 9.3, 11.0, 19.5, 36.7]      # Stance & Swing most affected
    d2 = [0.4,7.3,3.3,5.4,10.7,8.5,11.7]               # Stance & Swing most affected
    results = independent_t_test(d1, d2)
    print(results)
              

    print("All test cases passed!")


if __name__ == "__main__":
    # Run the test function
    test_independent_t_test()


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
    print(results)


    # # Perform one-sample t-test
    stanceswing_midstance = [0.5, 8.0, 8.2, 9.3, 11.0, 19.5, 36.7]      # Stance & Swing most affected
    stanceswing_Initial = [0.4,7.3,3.3,5.4,10.7,8.5,11.7]               # Stance & Swing most affected

    res = one_sample_t_test(stanceswing_midstance, test_value=0, alpha=0.05)
    results = testTool.normality_test(stanceswing_midstance, method="all")
    print(res, '===', results)
