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
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera
from scipy.stats import levene
from statsmodels.stats.power import TTestPower

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

def normality_test(data):
    """
    Perform normality tests on the given data.

    Parameters:
    data: A 1D array or list of numerical data.

    Returns:
    results: A dictionary containing test statistics and p-values for multiple normality tests.
    """
    from scipy.stats import shapiro, anderson, normaltest
    
    results = {}

    # Shapiro-Wilk Test
    stat_shapiro, p_value_shapiro = shapiro(data)
    results['Shapiro-Wilk Test'] = {'Statistic': stat_shapiro, 'P-value': p_value_shapiro}

    # Anderson-Darling Test
    result_anderson = anderson(data)
    results['Anderson-Darling Test'] = {'Statistic': result_anderson.statistic, 'Critical Values': result_anderson.critical_values, 'Significance Level': result_anderson.significance_level}

    # jarque_bera Test
    stat, p_value = jarque_bera(data)
    results['jarque_bera Test'] = {'Statistic': stat, 'P-value': p_value}

    # D'Agostino and Pearson's Normality Test
    if len(data)>7:
        stat_dagostino, p_value_dagostino = normaltest(data)
        results["D'Agostino-Pearson Test"] = {'Statistic': stat_dagostino, 'P-value': p_value_dagostino}

    return results

def anova_with_posthoc(*groups):
    """
    Perform a one-way ANOVA test followed by Tukey's HSD post hoc test if the ANOVA is significant.
    
    Parameters:
    *groups: Two or more arrays of data points (representing different groups).
    
    Returns:
    result: A dictionary containing the ANOVA results and post hoc Tukey's HSD results (if applicable).
    """

    import numpy as np
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Perform the one-way ANOVA
    statistic, p_value = f_oneway(*groups)
    
    result = {
        'ANOVA': {
            'statistic': statistic,
            'p-value': p_value
        }
    }
    
    # If the ANOVA result is significant (p-value < 0.05), proceed with post hoc Tukey's HSD test
    alpha = 0.05
    if p_value < alpha:
        # Combine the data into a single array for Tukey's HSD test
        data = np.concatenate(groups)
        labels = [f'Group {i+1}' for i in range(len(groups)) for _ in groups[i]]
        
        # Tukey's HSD test
        tukey_result = pairwise_tukeyhsd(data, labels, alpha=alpha)
        
        # Store the Tukey's HSD results in the result dictionary
        result['Tukey HSD'] = tukey_result.summary().as_text()
    
    return result

def kruskal_wallis_with_posthoc(*groups):
    """
    Perform Kruskal-Wallis H test followed by Dunn's Test if Kruskal-Wallis is significant.
    
    Parameters:
    *groups: Two or more arrays of data points (representing different groups).
    
    Returns:
    result: A dictionary containing Kruskal-Wallis results and Dunn's test results if applicable.
    """

    import numpy as np
    from scipy.stats import kruskal
    import scikit_posthocs as sp

    # Perform Kruskal-Wallis H test
    statistic, p_value = kruskal(*groups)
    
    result = {
        'Kruskal-Wallis': {
            'statistic': statistic,
            'p-value': p_value
        }
    }
    
    # If Kruskal-Wallis is significant (p-value < 0.05), perform Dunn's test for pairwise comparisons
    alpha = 0.05
    # if p_value < alpha:
    #     # Combine the data into a single array for Dunn's test
    #     data = np.concatenate(groups)
    #     labels = [f'Group {i+1}' for i in range(len(groups)) for _ in groups[i]]

    #     data = np.array(data)
    #     labels = np.array(labels)
    #     print(data, labels)
    #     # Dunn's test using scikit-posthocs
    #     dunn_result = sp.posthoc_dunn(data, labels, p_adjust='bonferroni')
        
    #     # Store the Dunn's test results in the result dictionary
    #     result['Dunn\'s Test'] = dunn_result
    
    return result


def perform_t_test(group_1, group_2, paired=False):
    """
    Perform a t-test between two groups.

    Parameters:
    - group_1: list or array-like, first group of data
    - group_2: list or array-like, second group of data
    - paired: bool, default=False. If True, performs a paired t-test.
    
    Returns:
    - t_stat: float, the t-statistic
    - p_value: float, the p-value
    """
    from scipy.stats import ttest_ind, ttest_rel

    if paired:
        # Paired t-test
        if len(group_1) != len(group_2):
            raise ValueError("For a paired t-test, both groups must have the same length.")
        t_stat, p_value = ttest_rel(group_1, group_2)
    else:
        # Independent t-test
        t_stat, p_value = ttest_ind(group_1, group_2, equal_var=False)  # Welch's t-test
    
    return t_stat, p_value


def perform_ANOVA_tests(*groups):
    """
    Perform various statistical tests on the given datasets.
    
    Parameters:
    - *groups: Variable number of groups of data (e.g., group1, group2, group3, ...)
    
    Prints the results of normality tests, Levene's test, Bartlett's test, ANOVA with post hoc, and Kruskal-Wallis with post hoc.
    """

    import pingouin as pg
    import pandas as pd
    import scipy.stats as stats
    
    print('-----------------------------------------')
    # Normality tests for each group
    for i, group in enumerate(groups, 1):
        print(f'## Normality Test for Group {i}')
        print(f'Normality test for group {i}')
        results = normality_test(group)
        print(results)
    
    # Levene's test for equality of variances
    print('## Levene Test for Equality of Variance')
    print('Levene test ')
    statistic, p_value = levene_test(*groups)
    print(f'Levene Test Statistic: {statistic}, p-value: {p_value}')
    
    # Bartlett's test for equality of variances
    print('## Bartlett Test for Equality of Variance')
    print('Bartlett test ')
    statistic, p_value = bartlett_test(*groups)
    print(f'Bartlett Test Statistic: {statistic}, p-value: {p_value}')
    
    # ANOVA test with post hoc analysis (only if more than two groups)
    if len(groups) > 2:
        print('## ANOVA with Post Hoc Test')
        print('ANOVA test ')
        result = anova_with_posthoc(*groups)
        print(result)
    
    # Kruskal-Wallis test with Dunn's post hoc test (non-parametric, used when data is not normal)
    print('## Kruskal-Wallis Test with Dunn\'s Post Hoc')
    print('Kruskal-Wallis test ')
    result = kruskal_wallis_with_posthoc(*groups)
    print(result)


    # Repeated Measures ANOVA and Post Hoc for the given groups
    print('## Repeated Measures ANOVA with Post Hoc Comparisons')
    
    # Ensure all groups are of the same length (number of subjects)
    num_subjects = len(groups[0])  # Assuming all groups have the same number of subjects
    for i, group in enumerate(groups, 1):
        if len(group) != num_subjects:
            print(f"Warning: Group {i} does not have the same number of subjects. Skipping repeated measures ANOVA.")
            return

    # Create a long-format DataFrame for the repeated measures ANOVA
    subjects = list(range(1, num_subjects + 1))  # Subject identifiers (1 to n)
    conditions = [f'Condition_{i}' for i in range(1, len(groups) + 1)]  # Create condition names (Condition_1, Condition_2, etc.)

    # Creating a long-format DataFrame where each row is a subject's score under a condition
    data = pd.DataFrame({
        'Subject': subjects * len(groups),  # Repeating subjects for each condition
        'Condition': [condition for condition in conditions for _ in range(num_subjects)],  # Repeating conditions for each subject
        'Score': [score for group in groups for score in group]  # Flatten the groups into a single list
    })

    # Perform Repeated Measures ANOVA using pingouin
    anova = pg.rm_anova(dv='Score', within='Condition', subject='Subject', data=data, detailed=True)

    # Test for sphericity
    mauchly = pg.sphericity(data=data, dv='Score', within='Condition', subject='Subject')
    print("Test for sphericity:")
    print(mauchly)

    # Print ANOVA results
    print("ANOVA Results:")
    print(anova)

    # Perform post hoc comparisons (pairwise t-tests) with Bonferroni correction
    post_hoc = pg.pairwise_tests(dv='Score', within='Condition', subject='Subject', data=data, padjust='bonferroni')

    # Print post hoc comparison results
    print("\nPost Hoc Comparisons (with Bonferroni correction):")
    print(post_hoc)

    print("-----------------------------------------")
    print("-----------------------------------------")


import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import shapiro
from statsmodels.stats.anova import AnovaRM
from pingouin import sphericity
from pingouin import sphericity, pairwise_ttests, power_rm_anova

def one_factor_repeated_anova(*groups):
    """
    Perform a one-factor repeated measures ANOVA with assumption checks.

    Parameters:
        *groups: Variable number of arrays or lists representing repeated measures data.

    Returns:
        Dictionary with results of assumption checks and ANOVA summary.
    """
    # Check input consistency
    if len(groups) < 2:
        raise ValueError("At least two groups are required for repeated measures ANOVA.")

    n_subjects = len(groups[0])
    if not all(len(group) == n_subjects for group in groups):
        raise ValueError("All groups must have the same number of observations (balanced design).")

    # Convert data into a long-form DataFrame
    data = pd.DataFrame({f"Group_{i+1}": group for i, group in enumerate(groups)})
    data["Subject"] = np.arange(1, n_subjects + 1)
    data_long = data.melt(id_vars="Subject", var_name="Condition", value_name="Score")

    # Check for normality (Shapiro-Wilk test for each group)
    normality_results = {col: shapiro(data[col]).pvalue for col in data.columns if col != "Subject"}
    normality_passed = all(p > 0.05 for p in normality_results.values())

    # Check for sphericity (Mauchly's test using Pingouin)
    sphericity_res = sphericity(data_long, dv="Score", subject="Subject", within="Condition")
    sphericity_passed = sphericity_res.pval > 0.05;

    # Perform repeated measures ANOVA
    anova_results = pg.rm_anova(data=data_long, dv="Score", within="Condition", subject="Subject")

    # Perform post hoc analysis (pairwise comparisons)
    post_hoc_results = pg.pairwise_ttests(
        data=data_long,
        dv="Score",
        within="Condition",
        subject="Subject",
        padjust="bonferroni"  # Use Bonferroni correction for multiple comparisons
    )

    # Power analysis for each post hoc comparison
    power_results = []
    power_analysis = TTestPower()

    for index, row in post_hoc_results.iterrows():
        # Calculate Cohen's d for each comparison using t-statistic and sample size
        t_stat = row['T']
        effect_size_d = t_stat / np.sqrt(n_subjects)
        
        # Perform power analysis using the Cohen's d for each pairwise comparison
        power = power_analysis.solve_power(effect_size=effect_size_d, nobs=n_subjects, alpha=0.05)
        
        power_results.append({
            "Contrast": f"{row['A']} vs {row['B']}",
            "T-statistic": t_stat,
            "Effect Size (Cohen's d)": effect_size_d,
            "Power": power
        })

    # Return results
    return {
        "Normality Results (Shapiro-Wilk)": normality_results,
        "Normality Passed": normality_passed,
        "Sphericity Test (Mauchly)": {
            "p-value": sphericity_res.pval,
            "Passed": sphericity_passed
        },
        "ANOVA Summary": anova_results,
        "Post Hoc Results": post_hoc_results,
        "power_results": power_results,
    }

### Defining the "More Affected Midstance" values for each mode (Stance & Swing, Stance, Swing)
print('## repeated ANOVA assistance mode effect on knee extension angle of More Affected side in Midstance')
stanceswing = [0.5, 8.0, 8.2, 9.3, 11.0, 19.5, 36.7]    # Stance & Swing
stance = [-5.3, 1.4, 8.1, 8.9, 6.2, 17.1, 18.4]         # Stance
swing = [-8.8, 2.5, 5.7, 1.4, -2.5, 5.1, 7.6]           # Swing
# perform_ANOVA_tests(stanceswing, stance, swing)

res = one_factor_repeated_anova(stanceswing, stance, swing)
print(res)

# # Reduction in Crouch: θ Initial Contact (°) More Affected
print('## assistance mode effect on knee extension angle of More Affected side in Initial Contact')
stanceswing = [0.4,7.3,3.3,5.4,10.7,8.5,11.7]   # Stance & Swing
stance = [-1.9,2.8,1.8,0.9,-0.5,6.0,0.3]        # Stance
swing = [-12.5,11.8,-2.6,3.9,5.7,15.6,10.6]     # Swing
#perform_ANOVA_tests(stanceswing, stance, swing)
res = one_factor_repeated_anova(stanceswing, stance, swing)
print(res)

# # Reduction in Crouch: θ Initial Contact (°) less affected
print('## assistance mode effect on knee extension angle of Less Affected side in Initial Contact')
stanceswing = [0.5,10.2,2.8,6.7,-2.1,8.8,19.3]   # Stance & Swing
stance = [-7.3,-4.3,-3.3,1.9,-11.0,1.9,-0.7]        # Stance
swing = [-6.7,6.1,3.8,10.8,-0.5,5.6,10.9,]     # Swing
# perform_ANOVA_tests(stanceswing, stance, swing)
res = one_factor_repeated_anova(stanceswing, stance, swing)
print(res)


# # Reduction in Crouch: θ Midstance (°) less affected
print('## assistance mode effect on knee extension angle of Less Affected side in Midstance')
stanceswing = [-6.2,5.2,5.6,8.9,8.6,6.9,11.5]   # Stance & Swing
stance = [-7.6,2.0,6.0,15.0,7.1,5.0,15.3]        # Stance
swing = [-12.4,5.2,-7.6,3.5,-2.3,-0.4,2.7]     # Swing
# perform_ANOVA_tests(stanceswing, stance, swing)
res = one_factor_repeated_anova(stanceswing, stance, swing)
print(res)

### check the effect of exo assistance
stanceswing_midstance = [0.5, 8.0, 8.2, 9.3, 11.0, 19.5, 36.7]      # Stance & Swing most affected
stanceswing_Initial = [0.4,7.3,3.3,5.4,10.7,8.5,11.7]               # Stance & Swing most affected

# # Perform one-sample t-test
res = one_sample_t_test(stanceswing_midstance, test_value=0, alpha=0.05)
results = normality_test(stanceswing_midstance)
print(res)

res = one_sample_t_test(stanceswing_Initial, test_value=0, alpha=0.05)
results = normality_test(stanceswing_Initial)
print(res)

stanceswing_midstance_less = [-6.2,5.2,5.6,8.9,8.6,6.9,11.5]    # Stance & Swing less affected
stanceswing_Initial_less = [0.5,10.2,2.8,6.7,-2.1,8.8,19.3]   # Stance & Swing less affected
res = one_sample_t_test(stanceswing_midstance_less, test_value=0, alpha=0.05)
results = normality_test(stanceswing_midstance_less)
print(res)
# # If p-value is less than 0.05, data is not normally distributed
print("Data is not normally distributed. Performing the Wilcoxon Signed-Rank Test.")
# Perform Wilcoxon Signed-Rank Test (comparing against a population mean of 0)
stat, p_value = stats.wilcoxon(np.array(stanceswing_midstance_less) - 0)  # 0 is the hypothesized median
print(f"Wilcoxon Signed-Rank Test p-value: {p_value}")


res = one_sample_t_test(stanceswing_Initial_less, test_value=0, alpha=0.05)
results = normality_test(stanceswing_Initial_less)
print(res)


# ## ===========================================
# print('## ===========================================')
# print('## ===========================================')
# print('## ===========================================')
# import numpy as np
# from scipy.stats import shapiro, levene, ttest_rel, wilcoxon

# def compare_twodata(data):
#     """

#     Compares the 'First' and 'Last' columns of multiple cases in the input data.
#     Performs normality tests, variance equality tests, and selects the appropriate statistical test.
    
#     Parameters:
#         data (dict): Dictionary where each key is a case name and the values are dictionaries
#                      with 'First' and 'Last' columns as lists of numeric values.
                     
#     Returns:
#         dict: Results of statistical tests for each case (Shapiro-Wilk, Levene, t-test/Wilcoxon).
#     """

#     results = {}

#     # Perform analysis for each case
#     for case, values in data.items():
#         first = np.array(values["First"])
#         last = np.array(values["Last"])

#         # Check normality using Shapiro-Wilk test
#         shapiro_first = shapiro(first).pvalue
#         shapiro_last = shapiro(last).pvalue

#         # Check variance equality using Levene's test if normality is assumed
#         if shapiro_first > 0.05 and shapiro_last > 0.05:
#             levene_pvalue = levene(first, last).pvalue
#             normal = True
#         else:
#             levene_pvalue = None
#             normal = False

#         # Perform the appropriate test
#         if normal and levene_pvalue and levene_pvalue > 0.05:
#             # If both are normal and variances are equal, use paired t-test
#             test_stat, pvalue = ttest_rel(first, last)
#             test_used = "t-test"
#         else:
#             # If normality or variance equality is not met, use Wilcoxon signed-rank test
#             test_stat, pvalue = wilcoxon(first, last)
#             test_used = "Wilcoxon"

#         # Store results for each case
#         results[case] = {
#             "Shapiro First (p)": shapiro_first,
#             "Shapiro Last (p)": shapiro_last,
#             "Levene (p)": levene_pvalue,
#             "Test Used": test_used,
#             "Test Statistic": test_stat,
#             "p-value": pvalue
#         }

#     # Print the results
#     results_df = pd.DataFrame(results).T
#     print(results_df)
#     return results


# Midstance = {
#     "exoMidstanceMoreAffected": {
#         "First": [37.0, 10.6, 28.3, 28.0, 21.9, 17.9, 3.9],
#         "Last": [29.1, 11.3, 23.1, 22.9, 14.2, 13.2, -3.8]
#     },
#     "exoMidstanceLessAffected": {
#         "First": [31.0, 2.8, 14.8, 24.2, 15.3, -1.4, 2.4],
#         "Last": [33.4, -3.4, 18.5, 19.3, 11.5, 5.1, 0.4]
#     },
#     "baseMidstanceMoreAffected": {
#         "First": [31.4, 17.8, 26.5, 28.3, 42.6, 30.8, 23.8],
#         "Last": [29.7, 19.4, 31.4, 30.7, 25.2, 26.5, 35.8]
#     },
#     "baseMidstanceLessAffected": {
#         "First": [28.6, 6.1, 17.4, 25.2, 29.8, -2.6, 7.2],
#         "Last": [27.2, 1.8, 24.1, 27.4, 20.1, 2.5, 12.5]
#     }
# }

# # Call the function with the data
# results = compare_twodata(Midstance)


# Initial = {
#     "exoInitialMoreAffected": {
#         "First": [39.7, 25.0, 34.1, 33.8, 35.4, 32.9, 33.6],
#         "Last": [31.2, 22.8, 33.5, 30.1, 27.5, 29.1, 41.1]
#     },
#     "exoInitialLessAffected": {
#         "First": [38.7, 8.7, 26.5, 37.6, 33.4, 24.8, 27.7],
#         "Last": [32.1, 11.1, 31.2, 26.2, 30.5, 34.5, 16.4]
#     },
#     "baseInitialMoreAffected": {
#         "First": [30.7, 29.0, 30.5, 29.2, 51.7, 49.7, 48.9],
#         "Last": [31.6, 30.1, 36.8, 33.7, 38.2, 46.1, 50.7]
#     },
#     "baseInitialLessAffected": {
#         "First": [31.0, 21.0, 25.2, 26.7, 44.2, 27.6, 26.8],
#         "Last": [32.5, 21.3, 33.9, 31.1, 28.4, 37.2, 27.8]
#     }
# }

# # Call the function with the data
# results = compare_twodata(Initial)

# stepLength = {
#     "case1": {
#         "First": [0.05, 0.45, 0.18, 0.25, 0.22, 0.41, 0.28],
#         "Last": [0.09, 0.43, 0.35, 0.45, 0.28, 0.49, 0.34]
#     },
#     "case2": {
#         "First": [0.19, 0.39, 0.34, 0.47, 0.25, 0.46, 0.29],
#         "Last": [0.35, 0.43, 0.39, 0.57, 0.29, 0.49, 0.17]
#     },
#     "case3": {
#         "First": [0.30, 0.50, 0.30, 0.40, 0.30, 0.40, 0.40],
#         "Last": [0.30, 0.60, 0.30, 0.50, 0.30, 0.30, 0.30]
#     },
#     "case4": {
#         "First": [0.40, 0.50, 0.40, 0.60, 0.30, 0.40, 0.40],
#         "Last": [0.40, 0.50, 0.40, 0.60, 0.30, 0.40, 0.30]
#     }
# }
# results = compare_twodata(stepLength)


# gaitSpeed = {
#     "case1": {
#         "First": [0.16, 0.66, 0.48, 0.78, 0.42, 0.88, 0.52],
#         "Last": [0.37, 0.75, 0.73, 1.04, 0.57, 1.28, 0.40]
#     },
#     "case2": {
#         "First": [0.60, 1.00, 0.60, 1.00, 0.50, 0.80, 0.70],
#         "Last": [0.60, 0.90, 0.60, 1.10, 0.60, 0.80, 0.50]
#     }
# }
# results = compare_twodata(gaitSpeed)

