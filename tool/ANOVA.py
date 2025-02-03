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
from pingouin import pairwise_gameshowell
from scipy import stats
from scipy.stats import levene, ttest_ind, jarque_bera
from scipy.stats import f_oneway, shapiro, levene, kruskal
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.power import TTestPower
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def one_way_anova(data, group_names=None, alpha=0.05, post_hoc_method='tukey'):
    """
    Perform one-way ANOVA, check requirements, and conduct post-hoc tests if significant.

    Parameters:
        data (list of lists): A list where each inner list represents data for a group.
        group_names (list): Names of the groups (default: None).
        alpha (float): Significance level (default: 0.05).
        post_hoc_method (str): Post-hoc method ('tukey', 'dunn', 'games-howell').

    
    1. Tukey's HSD (Honest Significant Difference)
    - **Usage**: Tukey's HSD is one of the most commonly used post-hoc tests. 
    - **Assumptions**: Assumes equal variances (homoscedasticity) and normally distributed data.
    - **Characteristics**: It is a conservative test, meaning it minimizes the chances of Type I errors (false positives).

    2. **Dunn's Test**
    - **Usage**: Dunn's test is a non-parametric post-hoc test used after a Kruskal-Wallis test (non-parametric equivalent of ANOVA). It compares all pairs of groups without assuming normality or equal variances.
    - **Assumptions**: No assumptions about normality or homogeneity of variance, making it ideal for non-normally distributed data.
    - **Characteristics**: It uses a ranking approach, which makes it less powerful than parametric tests like Tukey, but more robust to violations of assumptions.
    - **Output**: Dunn’s test calculates p-values adjusted for multiple comparisons, typically using the Bonferroni correction or similar adjustments.

    3. **Games-Howell Test**
    - **Usage**: The Games-Howell test is an alternative to Tukey's HSD when the assumption of equal variances is violated. It adjusts for both unequal variances and unequal sample sizes.
    - **Assumptions**: Does not assume homogeneity of variance or normality.
    - **Characteristics**: More robust than Tukey's test when variances are unequal or sample sizes are not the same across groups. It is less conservative than Tukey's HSD but controls for Type I errors.

    4. **Scheffé Test**
    - **Usage**: The Scheffé test is a highly conservative post-hoc test. It is particularly useful when comparing all possible linear combinations of groups (not just pairwise comparisons). It can handle complex contrasts between groups.
    - **Assumptions**: Assumes equal variances and normality.
    - **Characteristics**: Scheffé's test is more conservative than Tukey’s HSD. It is particularly useful when you need to test more complex hypotheses, such as comparing a combination of groups, rather than just pairwise comparisons.
    - **Output**: It provides adjusted p-values and pairwise comparisons, but it is more likely to result in a higher p-value (more conservative).

    5. **Dunnett's Test**
    - **Method**: `pairwise_ttests` with padjust='dunnett'
    - **Usage**: Dunnett's test is used when comparing several experimental groups against a single control group. It is typically used in situations where one group is a control, and you want to compare each experimental group against the control.
    - **Assumptions**: Assumes normality and equal variances (homoscedasticity) in the data.
    - **Characteristics**: Unlike other tests, Dunnett’s test controls for the Type I error rate specifically when comparing multiple treatments to a single control group. It adjusts for multiple comparisons but is less conservative when compared to methods like Scheffé.

    Returns:
        dict: A dictionary with the ANOVA result, requirement checks, and post-hoc results (if applicable).
    """
    # Convert to numpy arrays for easier processing
    group_data = [np.array(group)[~np.isnan(group)] for group in data]
    num_groups = len(group_data)
    group_names = group_names if group_names else [f"Group {i+1}" for i in range(num_groups)]

    # Check requirements
    # 1. Normality (Shapiro-Wilk Test for each group)
    normality_results = {name: shapiro(group)[1] for name, group in zip(group_names, group_data)}
    normality_passed = all(p > alpha for p in normality_results.values())

    # 2. Homogeneity of variances (Levene's Test)
    levene_p = levene(*group_data)[1]
    homogeneity_passed = levene_p > alpha

    # If requirements are not met, provide suggestions
    suggestions = []
    if not normality_passed:
        suggestions.append("Data is not normally distributed. Consider using a Kruskal-Wallis test.")
    if not homogeneity_passed:
        suggestions.append("Data does not meet homogeneity of variance. Consider using Welch's ANOVA.")

    if suggestions:
        return {
            'ANOVA': 'Requirements not met',
            'Normality': normality_results,
            'Homogeneity of variances': levene_p,
            'Suggestion': ' '.join(suggestions)
        }

    # Perform ANOVA
    f_stat, p_value = f_oneway(*group_data)
    anova_result = {'F-statistic': f_stat, 'p-value': p_value}

    # Post-hoc test if ANOVA shows significance
    if p_value < alpha:
        all_values = np.concatenate(group_data)
        all_groups = np.concatenate([[name] * len(group) for name, group in zip(group_names, group_data)])
        df = pd.DataFrame({'value': all_values, 'group': all_groups})

        if post_hoc_method == 'tukey':
            post_hoc = pairwise_tukeyhsd(all_values, all_groups, alpha=alpha)
            post_hoc_result = post_hoc.summary().as_text()

        elif post_hoc_method == 'dunn':
            post_hoc_result = posthoc_dunn(all_values, groups=all_groups, p_adjust='bonferroni').to_string()

        elif post_hoc_method == 'games-howell':
            post_hoc_result = pg.pairwise_gameshowell(dv='value', between='group', data=df).to_string()

        elif post_hoc_method == 'scheffe':
            post_hoc_result = pg.pairwise_ttests(dv='value', between='group', method='scheffe', data=df).to_string()

        elif post_hoc_method == 'dunnett':
            post_hoc_result = pg.pairwise_ttests(dv='value', between='group', padjust='dunnett', data=df).to_string()

        else:
            raise ValueError(f"Unknown post-hoc method: {post_hoc_method}")
    else:
        post_hoc_result = 'No significant differences found'

    return {
        'ANOVA': anova_result,
        'Normality': normality_results,
        'Homogeneity of variances': levene_p,
        'Post-hoc test': post_hoc_result,
    }

def test_one_way_anova():
    import numpy as np

    # Test 1: Balanced groups, normal distribution, equal variances
    print("\nTest 1: Balanced groups, Tukey's HSD")
    data_balanced = [
        [10.1, 9.5, 10.8, 9.9, 10.0],  # Group A
        [12.3, 11.7, 12.0, 11.5, 12.1],  # Group B
        [15.2, 14.8, 15.5, 14.9, 15.1],  # Group C
    ]
    group_names = ['A', 'B', 'C']
    result_balanced = one_way_anova(data_balanced, group_names, post_hoc_method='tukey')
    assert 'ANOVA' in result_balanced, "ANOVA results missing"
    assert 'Post-hoc test' in result_balanced, "Post-hoc results missing"
    print(result_balanced)

    # Test 2: Dunn's Test for nonparametric data
    print("\nTest 2: Nonparametric data, Dunn's Test")
    data_nonparametric = [
        [10, 12, 100],  # Group A (with an outlier)
        [9, 11, 14],    # Group B
        [10, 12, 15],   # Group C
    ]
    result_dunn = one_way_anova(data_nonparametric, group_names, post_hoc_method='dunn')
    assert 'ANOVA' in result_dunn, "ANOVA results missing"
    print(result_dunn)

    # Test 3: Unequal variances
    print("\nTest 3: Unequal variances, Tukey's HSD")
    data_unequal_variances = [
        [10.1, 9.5, 10.8, 9.9, 10.0],      # Group A
        [12.3, 11.7, 12.0, 11.5, 12.1],    # Group B
        [15.2, 50.0, 15.5, 14.9, 15.1],    # Group C (higher variance)
    ]
    result_unequal_var = one_way_anova(data_unequal_variances, group_names, post_hoc_method='tukey')
    assert 'Suggestion' in result_unequal_var, "Suggestion missing for unequal variances"
    print(result_unequal_var)

    # Test 4: Missing values
    print("\nTest 4: Missing values")
    data_missing = [
        [10.1, 9.5, 10.8, np.nan, 10.0],  # Group A
        [12.3, 11.7, 12.0, 11.5, np.nan], # Group B
        [15.2, 14.8, 15.5, 14.9, 15.1],  # Group C
    ]
    result_missing = one_way_anova(data_missing, group_names)
    assert 'ANOVA' in result_missing, "ANOVA results missing"
    print(result_missing)

    # Test 5: Games-Howell Test (Placeholder)
    print("\nTest 5: Games-Howell Test")
    
    result_games_howell = one_way_anova(data_balanced, group_names, post_hoc_method='games-howell')
    assert 'Post-hoc test' in result_games_howell, "Post-hoc results missing for Games-Howell"
    print(result_games_howell)


    # Test 6: Non-normal data
    print("\nTest 6: Non-normal data")
    data_non_normal = [
        [10, 9, 10, 8, 9],  # Group A
        [20, 21, 19, 22, 20],  # Group B
        [30, 40, 35, 50, 45],  # Group C (not normal)
    ]
    result_non_normal = one_way_anova(data_non_normal, group_names, post_hoc_method='dunn')
    assert 'Suggestion' in result_non_normal, "Suggestion missing for non-normal data"
    print(result_non_normal)

    print("\nAll tests completed successfully.")



import numpy as np
import pandas as pd
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from scikit_posthocs import posthoc_dunn
import pingouin as pg


def kruskal_wallis_test(data, group_names=None, alpha=0.05, post_hoc_method='dunn', power_analysis=False):
    """
    Perform Kruskal-Wallis test, optional post-hoc analysis, and optional power analysis.
    non-parametric statistical test used to compare the distributions of more than two independent groups. 
    It is used when the assumptions of one-way ANOVA (such as normality) are not met. 

    Parameters:
        data (list of lists): A list where each inner list represents data for a group.
        group_names (list): Names of the groups (default: None).
        alpha (float): Significance level (default: 0.05).
        post_hoc_method (str): Post-hoc method ('dunn').
        power_analysis (bool): Whether to perform power analysis (default: False).

    Returns:
        dict: A dictionary with the Kruskal-Wallis result, post-hoc results (if applicable), and power analysis results (if applicable).
    """
    # Convert to numpy arrays for easier processing
    group_data = [np.array(group)[~np.isnan(group)] for group in data]
    num_groups = len(group_data)
    group_names = group_names if group_names else [f"Group {i+1}" for i in range(num_groups)]

    # Perform Kruskal-Wallis test
    h_stat, p_value = kruskal(*group_data)
    kw_result = {'H-statistic': h_stat, 'p-value': p_value}

    # Post-hoc test if Kruskal-Wallis shows significance
    if p_value < alpha and post_hoc_method == 'dunn':
        all_values = np.concatenate(group_data)
        all_groups = np.concatenate([[name] * len(group) for name, group in zip(group_names, group_data)])

        # Perform Dunn's test with Bonferroni correction
        post_hoc_result = posthoc_dunn(all_values, groups=all_groups, p_adjust='bonferroni')
    else:
        post_hoc_result = 'No significant differences found or unknown post-hoc method.'

    # Power analysis
    power_result = None
    if power_analysis:
        # Convert data to a DataFrame for Pingouin
        all_values = np.concatenate(group_data)
        all_groups = np.concatenate([[name] * len(group) for name, group in zip(group_names, group_data)])
        df = pd.DataFrame({'value': all_values, 'group': all_groups})

        # Calculate effect size and perform power analysis
        effect_size = pg.compute_effsize(df, dv='value', between='group', eftype='cohen')
        total_n = len(all_values)
        power = pg.power_anova(eta=effect_size, n=total_n, k=num_groups, alpha=alpha)
        power_result = {'Effect size (Cohen f)': effect_size, 'Power': power}

    return {
        'Kruskal-Wallis': kw_result,
        'Post-hoc test': post_hoc_result if isinstance(post_hoc_result, str) else post_hoc_result.to_string(),
        'Power analysis': power_result
    }

# Example usage:
#data = [
#    [5, 7, 8, 6],
#    [8, 9, 7, 10],
#    [2, 4, 3, 5]
#]
#result = kruskal_wallis_test(data, group_names=['A', 'B', 'C'], power_analysis=True)


import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from statsmodels.stats.power import FTestAnovaPower

def welch_anova(data, group_names=None, alpha=0.05, post_hoc_method='games-howell', power_analysis=False):
    """
    Perform Welch ANOVA, check requirements, conduct post-hoc tests if significant, and perform power analysis.
    The Welch ANOVA (also known as Welch's test) is an adaptation of the standard one-way ANOVA that is used 
    when the assumptions of equal variances (homogeneity of variance) across groups are violated.

    Parameters:
        data (list of lists): A list where each inner list represents data for a group.
        group_names (list): Names of the groups (default: None).
        alpha (float): Significance level (default: 0.05).
        post_hoc_method (str): Post-hoc method ('games-howell').
        power_analysis (bool): Whether to perform power analysis (default: False).

    Returns:
        dict: A dictionary with Welch ANOVA results, assumption checks, post-hoc results (if applicable), and power analysis results (if applicable).
    """
    # Convert to numpy arrays for easier processing
    group_data = [np.array(group)[~np.isnan(group)] for group in data]
    group_names = group_names if group_names else [f"Group {i+1}" for i in range(len(group_data))]
    
    # Convert data into a pandas DataFrame for Pingouin
    all_values = np.concatenate(group_data)
    all_groups = np.concatenate([[name] * len(group) for name, group in zip(group_names, group_data)])
    df = pd.DataFrame({'value': all_values, 'group': all_groups})
    
    # Perform Welch ANOVA using Pingouin
    welch_result = pg.welch_anova(dv='value', between='group', data=df)
    
    # Assumption checks
    # Check if data is approximately normally distributed using Shapiro-Wilk test
    normality_results = {name: pg.normality(df[df['group'] == name]['value']).iloc[0, 1] for name in group_names}
    normality_passed = all(p > alpha for p in normality_results.values())

    # Check if variances are unequal using Levene's Test
    levene_p = pg.homoscedasticity(data=df, dv='value', group='group').iloc[0, 1]
    homogeneity_passed = levene_p > alpha

    # Suggestions based on assumptions
    suggestions = []
    if not normality_passed:
        suggestions.append("Data is not normally distributed. Consider using non-parametric tests like Kruskal-Wallis.")
    if homogeneity_passed:
        suggestions.append("Homogeneity of variances assumption is met. Consider using standard one-way ANOVA.")

    # Post-hoc test if Welch ANOVA shows significance
    if welch_result['p-unc'][0] < alpha and post_hoc_method == 'games-howell':
        post_hoc_result = pg.pairwise_gameshowell(dv='value', between='group', data=df).to_string()
    else:
        post_hoc_result = 'No significant differences found or unknown post-hoc method.'

    # Power analysis
    power_result = None
    if power_analysis:
        # Compute effect size (eta-squared)
        ss_between = welch_result['SS_between'][0]
        ss_total = welch_result['SS_total'][0]
        eta_squared = ss_between / ss_total

        # Calculate average group size
        avg_group_size = np.mean([len(group) for group in group_data])

        # Perform power analysis
        analysis = FTestAnovaPower()
        power = analysis.solve_power(effect_size=np.sqrt(eta_squared), nobs=avg_group_size * len(group_data), alpha=alpha, k_groups=len(group_data))
        
        power_result = {
            'Effect size (eta-squared)': eta_squared,
            'Power': power,
            'Note': 'Power analysis assumes approximate robustness for Welch ANOVA.'
        }

    return {
        'Welch ANOVA': welch_result.to_dict('records'),
        'Normality': normality_results,
        'Homogeneity of variances': levene_p,
        'Suggestions': suggestions,
        'Post-hoc test': post_hoc_result,
        'Power analysis': power_result
    }




def one_factor_repeated_anova(*groups):
    """
    Perform a one-factor repeated measures ANOVA with assumption checks.

    The One-Factor Repeated Measures ANOVA is a statistical test used to analyze data 
    when the same subjects are exposed to different conditions or treatments, and 
    the factor being tested is one (hence "one-factor"). 
    It compares the means of these conditions to determine 
    if there are statistically significant differences in a repeated measure scenario 
    (i.e., measurements taken from the same subjects multiple times).

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



import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import FTestPower

def friedman_test(data, subject_ids=None, condition_names=None, alpha=0.05):
    """
    Perform the Friedman test for repeated measures, post-hoc analysis if significant, and calculate power.

    The Friedman Test is a non-parametric statistical test used to detect differences in treatments 
    across multiple test attempts. It is the non-parametric counterpart to the One-Way Repeated Measures ANOVA 
    and is used especially when the assumptions of normality for repeated measures ANOVA are not met.

    Parameters:
        data (list of lists): A list where each inner list represents data for a condition.
        subject_ids (list): List of subject identifiers (optional).
        condition_names (list): Names of the conditions (optional).
        alpha (float): Significance level (default: 0.05).

    Returns:
        dict: Results of the Friedman test, post-hoc analysis (if applicable), and power analysis.
    """
    # Number of conditions
    num_conditions = len(data)

    if condition_names is None:
        condition_names = [f"Condition {i+1}" for i in range(num_conditions)]

    if subject_ids is None:
        subject_ids = [f"Subject {i+1}" for i in range(len(data[0]))]

    # Check input dimensions
    if any(len(group) != len(subject_ids) for group in data):
        raise ValueError("All conditions must have the same number of subjects.")

    # Perform the Friedman test
    stat, p_value = friedmanchisquare(*data)
    friedman_result = {
        'Test Statistic': stat,
        'p-value': p_value
    }

    # Calculate effect size (Kendall's W)
    num_subjects = len(subject_ids)
    sum_ranks = np.sum([np.mean(group) for group in data])
    kendalls_w = stat / (num_subjects * (num_conditions - 1))

    # Power analysis
    power_analysis = FTestPower()
    effect_size = np.sqrt(kendalls_w / (1 - kendalls_w)) if kendalls_w < 1 else 0.0
    power = power_analysis.solve_power(effect_size=effect_size, df_num=num_conditions - 1, df_denom=(num_subjects - 1) * (num_conditions - 1), alpha=alpha)
    power_result = {
        'Effect Size (Kendall\'s W)': kendalls_w,
        'Power': power
    }

    # Post-hoc pairwise comparisons if the test is significant
    if p_value < alpha:
        pairwise_results = []
        for i in range(num_conditions):
            for j in range(i + 1, num_conditions):
                stat, p = wilcoxon(data[i], data[j])
                pairwise_results.append({
                    'Comparison': f"{condition_names[i]} vs {condition_names[j]}",
                    'W-statistic': stat,
                    'p-value': p
                })

        # Adjust p-values for multiple comparisons
        p_values = [res['p-value'] for res in pairwise_results]
        adjusted = multipletests(p_values, alpha=alpha, method='bonferroni')
        for res, adj_p in zip(pairwise_results, adjusted[1]):
            res['Adjusted p-value'] = adj_p

        post_hoc_result = pairwise_results
    else:
        post_hoc_result = "No significant differences found."

    # Compile results
    return {
        'Friedman Test': friedman_result,
        'Post-hoc Analysis': post_hoc_result,
        'Power Analysis': power_result
    }



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



test_one_way_anova()
# ### Defining the "More Affected Midstance" values for each mode (Stance & Swing, Stance, Swing)
# print('## repeated ANOVA assistance mode effect on knee extension angle of More Affected side in Midstance')
# stanceswing = [0.5, 8.0, 8.2, 9.3, 11.0, 19.5, 36.7]    # Stance & Swing
# stance = [-5.3, 1.4, 8.1, 8.9, 6.2, 17.1, 18.4]         # Stance
# swing = [-8.8, 2.5, 5.7, 1.4, -2.5, 5.1, 7.6]           # Swing
# # perform_ANOVA_tests(stanceswing, stance, swing)

# res = one_factor_repeated_anova(stanceswing, stance, swing)
# print(res)

# # # Reduction in Crouch: θ Initial Contact (°) More Affected
# print('## assistance mode effect on knee extension angle of More Affected side in Initial Contact')
# stanceswing = [0.4,7.3,3.3,5.4,10.7,8.5,11.7]   # Stance & Swing
# stance = [-1.9,2.8,1.8,0.9,-0.5,6.0,0.3]        # Stance
# swing = [-12.5,11.8,-2.6,3.9,5.7,15.6,10.6]     # Swing
# #perform_ANOVA_tests(stanceswing, stance, swing)
# res = one_factor_repeated_anova(stanceswing, stance, swing)
# print(res)

# # # Reduction in Crouch: θ Initial Contact (°) less affected
# print('## assistance mode effect on knee extension angle of Less Affected side in Initial Contact')
# stanceswing = [0.5,10.2,2.8,6.7,-2.1,8.8,19.3]   # Stance & Swing
# stance = [-7.3,-4.3,-3.3,1.9,-11.0,1.9,-0.7]        # Stance
# swing = [-6.7,6.1,3.8,10.8,-0.5,5.6,10.9,]     # Swing
# # perform_ANOVA_tests(stanceswing, stance, swing)
# res = one_factor_repeated_anova(stanceswing, stance, swing)
# print(res)


# # # Reduction in Crouch: θ Midstance (°) less affected
# print('## assistance mode effect on knee extension angle of Less Affected side in Midstance')
# stanceswing = [-6.2,5.2,5.6,8.9,8.6,6.9,11.5]   # Stance & Swing
# stance = [-7.6,2.0,6.0,15.0,7.1,5.0,15.3]        # Stance
# swing = [-12.4,5.2,-7.6,3.5,-2.3,-0.4,2.7]     # Swing
# # perform_ANOVA_tests(stanceswing, stance, swing)
# res = one_factor_repeated_anova(stanceswing, stance, swing)
# print(res)




