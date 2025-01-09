

def normality_test(data, method="all"):
    """
    Perform normality tests on the given data.

    Parameters:
    data: A 1D array or list of numerical data.
    method: A string specifying which normality test to run. Options are:
            "shapiro" - Shapiro-Wilk Test
            "anderson" - Anderson-Darling Test
            "jarque_bera" - Jarque-Bera Test
            "dagostino" - D'Agostino-Pearson Test
            "all" - Run all available tests (default).

    Returns:
    results: A dictionary containing test statistics and p-values for the selected normality test(s).
    """
    from scipy.stats import shapiro, anderson, jarque_bera, normaltest

    results = {}

    # Validate the method input
    valid_methods = {"shapiro", "anderson", "jarque_bera", "dagostino", "all"}
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Choose from {valid_methods}.")

    # Check for numeric data
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("All elements in the data must be numeric.")

    # Check for sufficient data size
    n = len(data)
    if n < 3:
        raise ValueError("Normality tests require at least 3 data points.")

    # Shapiro-Wilk Test
    if method in {"shapiro", "all"}:
        if n <= 5000:  # Shapiro-Wilk is valid for n <= 5000
            stat_shapiro, p_value_shapiro = shapiro(data)
            results['Shapiro-Wilk Test'] = {'Statistic': stat_shapiro, 'P-value': p_value_shapiro}
        else:
            results['Shapiro-Wilk Test'] = "Not performed (requires n <= 5000)."
            
    # Anderson-Darling Test
    if method in {"anderson", "all"}:
        result_anderson = anderson(data)
        results['Anderson-Darling Test'] = {
            'Statistic': result_anderson.statistic,
            'Critical Values': result_anderson.critical_values,
            'Significance Level': result_anderson.significance_level
        }

    # Jarque-Bera Test
    if method in {"jarque_bera", "all"}:
        if n >= 5:  # Jarque-Bera works well for larger sample sizes
            stat_jarque_bera, p_value_jarque_bera = jarque_bera(data)
            results['Jarque-Bera Test'] = {
                'Statistic': stat_jarque_bera,
                'P-value': p_value_jarque_bera,
            }
        else:
            results['Jarque-Bera Test'] = "Not performed (requires n >= 5)."

    # D'Agostino and Pearson's Normality Test (requires more than 7 data points)
    if method in {"dagostino", "all"}:
        if n > 20:
            stat_dagostino, p_value_dagostino = normaltest(data)
            results["D'Agostino-Pearson Test"] = {'Statistic': stat_dagostino, 'P-value': p_value_dagostino}
        else:
            results["D'Agostino-Pearson Test"] = "Not reliable (requires more than 20 data points)."

    return results

# equality of variances.
def levene_test(*groups, alpha = 0.05):
    """
    Perform Levene's test for equality of variances.

    Parameters:
    *groups: Two or more arrays of data points (representing different groups).

    Returns:
    statistic: Levene test statistic.
    p_value: P-value associated with the test.
    """
    # Perform the Levene test
    statistic, p_value = levene(*groups)
    print('statistic, p_value = ', statistic, p_value)

    # Decision based on P-value
    if p_value < alpha:
        print("Reject the null hypothesis: The variances are not equal (heteroscedastic).")
    else:
        print("Fail to reject the null hypothesis: The variances are equal (homoscedastic).")

    # Return the statistic and p-value
    return statistic, p_value

def bartlett_test(*groups, alpha = 0.05):
    """
    Perform Bartlett's test for equality of variances.

    Parameters:
    *groups: Two or more arrays of data points (representing different groups).

    Returns:
    statistic: Bartlett test statistic.
    p_value: P-value associated with the test.
    """
    from scipy.stats import bartlett

    # Perform the Bartlett test
    statistic, p_value = bartlett(*groups)
    print('statistic, p_value = ', statistic, p_value)

    # Decision based on P-value
    if p_value < alpha:
        print("Reject the null hypothesis: The variances are not equal (heteroscedastic).")
    else:
        print("Fail to reject the null hypothesis: The variances are equal (homoscedastic).")

    # Return the statistic and p-value
    return statistic, p_value


if __name__ == "__main__":
    data = [1.2, 2.3, 3.1, 2.9, 3.3, 2.7, 3.8, 2.1]

    # Run all tests
    results_all = normality_test(data, method="all")

    # Run only the Shapiro-Wilk test
    results_shapiro = normality_test(data, method="shapiro")
    print(results_shapiro)

    # Run only the Anderson-Darling test
    results_anderson = normality_test(data, method="anderson")
    print(results_anderson)