import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, normaltest
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os


confound_vars = ["trial_duration"]  

def cliffs_delta(lst1, lst2):
    """ effect size."""
    from numpy import array, sign
    m, n = len(lst1), len(lst2)
    lst1, lst2 = array(lst1), array(lst2)
    delta = np.mean([np.sign(x - y) for x in lst1 for y in lst2])

def run_statistical_analysis(ms_df, interbrain_df, alpha=0.05, output_path="stats_data.xlsx"):
    # Merge and preprocess
    full_df = pd.merge(ms_df, interbrain_df, on=["dyad", "trial"])
    full_df["is_correct"] = (full_df["result"] == "correct").astype(int)

    # Numeric features only
    numeric_df = full_df.select_dtypes(include=[np.number])
    feature_columns = [col for col in numeric_df.columns if col not in ["dyad", "trial", "is_correct"] + confound_vars]

    results = []

    for feature in feature_columns:
        correct_vals = full_df[full_df["is_correct"] == 1][feature].dropna()
        incorrect_vals = full_df[full_df["is_correct"] == 0][feature].dropna()

        if len(correct_vals) < 3 or len(incorrect_vals) < 3:
            print(f"⚠️ Skipping {feature}: too few datapoints.")
            continue

        # Normality check
        use_parametric = False
        if len(correct_vals) >= 8 and len(incorrect_vals) >= 8:
            _, p_norm_c = normaltest(correct_vals)
            _, p_norm_i = normaltest(incorrect_vals)
            use_parametric = p_norm_c > 0.05 and p_norm_i > 0.05

        if use_parametric:
            stat, p_val = ttest_ind(correct_vals, incorrect_vals, equal_var=False)
            test_used = "t-test"
            pooled_sd = np.sqrt(((np.std(correct_vals)**2 + np.std(incorrect_vals)**2) / 2))
            effect_size = (np.mean(correct_vals) - np.mean(incorrect_vals)) / pooled_sd  # Cohen's d
        else:
            stat, p_val = mannwhitneyu(correct_vals, incorrect_vals, alternative="two-sided")
            test_used = "mannwhitney"
            effect_size = cliffs_delta(correct_vals, incorrect_vals)

        # Mixed effects model with confound control
        formula = f"{feature} ~ is_correct"
        if confound_vars:
            formula += " + " + " + ".join(confound_vars)

        try:
            model = smf.mixedlm(formula, full_df, groups=full_df["dyad"]).fit()
            converged = model.converged
            p_mixed = model.pvalues.get("is_correct", np.nan)
        except Exception as e:
            p_mixed, converged = np.nan, False

        results.append({
            "feature": feature,
            "test": test_used,
            "p_uncorrected": p_val,
            "effect_size": effect_size,
            "mixed_model_p": p_mixed,
            "mixed_model_converged": converged,
            "correct_mean": np.mean(correct_vals),
            "incorrect_mean": np.mean(incorrect_vals),
        })

    # Compile results
    results_df = pd.DataFrame(results)
    reject, p_fdr, _, _ = multipletests(results_df["p_uncorrected"], alpha=alpha, method='fdr_bh')
    results_df["p_fdr"] = p_fdr
    results_df["significant"] = reject
    results_df["interesting_uncorrected"] = results_df["p_uncorrected"] < alpha

    # Export
    results_df.to_excel(output_path, index=False)


    # Console output
    sig = results_df[results_df["significant"]]
    print("\n🔍 SIGNIFICANT FEATURES (FDR < 0.05):\n")
    if sig.empty:
        print("None. But maybe check the uncorrected ones below.")
    else:
        for _, row in sig.iterrows():
            print(f"📌 {row['feature']}: {row['test']} p={row['p_fdr']:.4g} | d={row['effect_size']:.2f} | μ_correct={row['correct_mean']:.2f}, μ_incorrect={row['incorrect_mean']:.2f}")

    maybe = results_df[(~results_df["significant"]) & (results_df["interesting_uncorrected"])]
    if not maybe.empty:
        print("\n⚠️ Interesting (but not FDR-corrected):")
        for _, row in maybe.iterrows():
            print(f"• {row['feature']}: p={row['p_uncorrected']:.4g}, d={row['effect_size']:.2f}")

    return results_df


def plot_significant_features(results_df, full_df, significance_level=0.05):
    """
    Plot features with significant differences between correct and incorrect trials.
    """
    
    sns.set_theme(style="whitegrid")
    palette = {
        "Correct": "#41518f",
        "Incorrect": "#9ba1d9"
    }

    # Filter features based on the mixed model p-value column
    sig_features = results_df[results_df["mixed_model_p"] < significance_level]["feature"].tolist()

    if not sig_features:
        print("⚠️ No significant features found at p <", significance_level)
        return

    # Set a more polished style
    sns.set(style="darkgrid", palette="Set2")

    for feature in sig_features:
        if feature not in full_df.columns:
            print(f"⚠️ Feature '{feature}' not found in data")
            continue

        plt.figure(figsize=(6, 4))
        ax = sns.boxplot(x="result", y=feature, data=full_df, palette=palette, fliersize=0)
        sns.stripplot(x="result", y=feature, data=full_df, color="#41518f", alpha=0.3, jitter=True)

        

        # Get the p-value for the feature
        p_val = results_df.loc[results_df['feature'] == feature, 'mixed_model_p'].values[0]
        
        # Use mixed_model_p in the title and formatting
        ax.set_title(f"{feature.replace('_', ' ').title()} (p = {p_val:.2e})", fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=14)
        
        plt.title(feat.replace("_", " ").capitalize())
        plt.xlabel("Result")
        plt.ylabel(feat.replace("_", " ").capitalize())
        plt.tight_layout()
        plt.show()
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(f"{feature}_comparison.png"), dpi=300)
        
        # Display the plot
        plt.show()
