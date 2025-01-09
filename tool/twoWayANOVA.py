import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import shapiro, levene
from statsmodels.stats.power import FTestAnovaPower


# finish
def two_way_anova_with_posthoc_and_checks(data, dependent_var, factor1, factor2, alpha=0.05):
    """
    Perform Two-Way ANOVA with data requirement checks and post hoc analysis using Pingouin.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the variables.
        dependent_var (str): The name of the dependent variable (continuous).
        factor1 (str): The name of the first categorical factor.
        factor2 (str): The name of the second categorical factor.
        alpha (float): Significance level for the analysis (default 0.05).
        
    Returns:
        dict: A dictionary with the results of checks, suggestions, ANOVA table, post hoc comparisons, and power analysis.
    """
    results = {"checks": {}, "suggestions": [], "anova_table": None, "posthoc": {}, "power_analysis": {}}
    
    # 1. Check group representation
    group_sizes = data.groupby([factor1, factor2]).size()
    if group_sizes.min() == 0:
        results['checks']['group_representation'] = "Fail: Some groups have no data."
        results['suggestions'].append(
            "Consider restructuring the dataset or combining groups to ensure all groups have data."
        )
    else:
        results['checks']['group_representation'] = "Pass: All groups have data."
    
    # 2. Check normality using Shapiro-Wilk test
    normality_pvals = []
    for _, group in data.groupby([factor1, factor2]):
        stat, p = shapiro(group[dependent_var])
        normality_pvals.append(p)
    if all(p > 0.05 for p in normality_pvals):
        results['checks']['normality'] = "Pass: All groups are approximately normal."
    else:
        results['checks']['normality'] = "Fail: At least one group is not normal."
        results['suggestions'].append(
            "Consider using a non-parametric test like the Kruskal-Wallis test or Permutation test."
        )
    
    # 3. Check homogeneity of variances using Levene's test
    groups = [group[dependent_var].values for _, group in data.groupby([factor1, factor2])]
    stat, p = levene(*groups)
    if p > 0.05:
        results['checks']['homogeneity'] = "Pass: Variances are approximately equal."
    else:
        results['checks']['homogeneity'] = "Fail: Variances are not equal."
        results['suggestions'].append(
            "Consider using Welch's ANOVA or a robust ANOVA for unequal variances."
        )
    
    # 4. Perform ANOVA only if all checks pass
    if all("Pass" in check for check in results['checks'].values()):
        
        # Two-way ANOVA
        anova = pg.anova(data=data, dv=dependent_var, between=[factor1, factor2], detailed=True)
        results['anova_table'] = anova

        
        effects = anova[['Source', 'np2']].dropna() # Drop rows with NaN in 'np2'
        for index, row in effects.iterrows():
            print(f"Source: {row['Source']}, Partial Eta-Squared: {row['np2']}")

            # Calculate effect size by eta-squared
            eta_squared = row['np2']  # Partial Eta-Squared
            f_effect_size = np.sqrt(eta_squared / (1 - eta_squared))

            # Power analysis
            total_n = len(data)
            num_groups = len(data.groupby([factor1, factor2]))
            
            power_analysis = FTestAnovaPower()
            achieved_power = power_analysis.solve_power(
                effect_size=f_effect_size,
                nobs=total_n,
                alpha=alpha,
                k_groups=num_groups
            )
            print(achieved_power)
            results["power_analysis"+'_'+row['Source']] = {
                "effect_size (Cohen's f)": f_effect_size,
                "achieved_power": achieved_power,
                "recommended_sample_size (80% power)": power_analysis.solve_power(
                    effect_size=f_effect_size,
                    power=0.8,
                    alpha=alpha,
                    k_groups=num_groups
                )
            }
        
        # Check if any p-value is significant and perform post hoc tests
        if (anova["p-unc"] < 0.05).any():
            for factor in [factor1, factor2]:
                posthoc = pg.pairwise_tukey(data=data, dv=dependent_var, between=factor)
                results["posthoc"][factor] = posthoc
        else:
            results["suggestions"].append(
                "No significant differences found. Post hoc comparisons are not performed."
            )
    else:
        results['anova_table'] = "ANOVA skipped due to failed checks."
    
    return results


# Example dataset
data = pd.DataFrame({
    "Score": [85, 90, 88, 78, 85, 82, 92, 95, 93, 84, 87, 85],
    "Group": ["A", "A", "A", "B", "B", "B", "A", "A", "A", "B", "B", "B"],
    "Time": ["Morning", "Morning", "Morning", "Morning", "Morning", "Morning",
             "Evening", "Evening", "Evening", "Evening", "Evening", "Evening"]
})

# Perform Two-Way ANOVA with checks and post hoc tests
result = two_way_anova_with_posthoc_and_checks(data, dependent_var="Score", factor1="Group", factor2="Time")
print(result)

# Print checks
print("Checks:")
for check, status in result["checks"].items():
    print(f"{check}: {status}")

# Print suggestions
print("\nSuggestions:")
for suggestion in result["suggestions"]:
    print(f"- {suggestion}")

# Print ANOVA Table
if isinstance(result["anova_table"], pd.DataFrame):
    print("\nANOVA Table:")
    print(result["anova_table"])

# Print Post Hoc Comparisons
if result["posthoc"]:
    print("\nPost Hoc Comparisons:")
    for factor, posthoc_df in result["posthoc"].items():
        print(f"\n{factor}:\n{posthoc_df}")