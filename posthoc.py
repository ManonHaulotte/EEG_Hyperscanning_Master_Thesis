"""
Post-hoc analysis functions 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import chi2_contingency
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import networkx as nx
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel, ttest_ind, chi2_contingency
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def calculate_cohens_d_paired(group1, group2):
    """Calculate Cohen's d for paired samples"""
    diff = np.array(group1) - np.array(group2)
    return np.mean(diff) / np.std(diff, ddof=1)

def calculate_cohens_d_independent(group1, group2):
    """Calculate Cohen's d for independent samples"""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def create_comparison_plot(ax, emp_data, surr_data, emp_color, surr_color, title, p_value, effect_size):
    """Create comparison plot showing ALL data points without stars"""
    from scipy import stats as scipy_stats
    
    # Use ALL data - no sampling
    np.random.seed(42)
    emp_sample = emp_data
    surr_sample = surr_data

    # Add jitter to all points
    emp_x = np.random.normal(1, 0.04, len(emp_sample))
    surr_x = np.random.normal(2, 0.04, len(surr_sample))

    # Adjust alpha based on actual data density for visibility
    alpha_emp = min(0.8, 300/len(emp_sample))
    alpha_surr = min(0.8, 300/len(surr_sample))

    # Plot ALL points
    ax.scatter(emp_x, emp_sample, alpha=alpha_emp, color=emp_color, s=15, edgecolors='none')
    ax.scatter(surr_x, surr_sample, alpha=alpha_surr, color=surr_color, s=15, edgecolors='none')
    
    # Add mean lines
    emp_mean = np.mean(emp_data)
    surr_mean = np.mean(surr_data)
    
    ax.plot([0.8, 1.2], [emp_mean, emp_mean], color=emp_color, linewidth=4, alpha=0.9)
    ax.plot([1.8, 2.2], [surr_mean, surr_mean], color=surr_color, linewidth=4, alpha=0.9)
    
    # Add error bars (SEM)
    emp_sem = scipy_stats.sem(emp_data)
    surr_sem = scipy_stats.sem(surr_data)
    
    ax.errorbar([1], [emp_mean], yerr=[emp_sem], color=emp_color, linewidth=2, 
               capsize=5, capthick=2, alpha=0.9)
    ax.errorbar([2], [surr_mean], yerr=[surr_sem], color=surr_color, linewidth=2, 
               capsize=5, capthick=2, alpha=0.9)
    
    # Formatting
    ax.set_xticks([1, 2])
    ax.set_xlim(0.5, 2.5)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Format p-value properly
    if p_value < 0.001:
        p_text = "p < 0.001"
    elif p_value < 0.01:
        p_text = "p < 0.01"
    elif p_value < 0.05:
        p_text = "p < 0.05"
    else:
        p_text = f"p = {p_value:.3f}"

    # Add stats text with better p-value formatting
    stats_text = f'Effect: {effect_size:.2f}\n{p_text}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           va='top', ha='left', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                   alpha=0.9, edgecolor='gray', linewidth=0.5))
    
    return ax

def create_behavioral_comparison(ax, data1, data2, color1, color2, labels, title):
    """Create strip plot comparison with equal sampling for behavioral data"""
    
    # Calculate statistics FIRST (use full data)
    t_stat, p_val = stats.ttest_ind(data1, data2)
    pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                         (len(data2) - 1) * np.var(data2, ddof=1)) / 
                        (len(data1) + len(data2) - 2))
    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else np.nan
    
    # Show ALL data points
    np.random.seed(42)
    
    # Add jitter to all points
    x1 = np.random.normal(1, 0.04, len(data1))
    x2 = np.random.normal(2, 0.04, len(data2))
    
    # Adjust alpha for visibility with many points
    alpha1 = min(0.8, 300/len(data1))
    alpha2 = min(0.8, 300/len(data2))
    
    # Plot ALL points
    ax.scatter(x1, data1, alpha=alpha1, color=color1, s=15, edgecolors='none')
    ax.scatter(x2, data2, alpha=alpha2, color=color2, s=15, edgecolors='none')
    
    # Add mean lines
    mean1, mean2 = np.mean(data1), np.mean(data2)
    ax.plot([0.8, 1.2], [mean1, mean1], color=color1, linewidth=4, alpha=0.9)
    ax.plot([1.8, 2.2], [mean2, mean2], color=color2, linewidth=4, alpha=0.9)
    
    # Add error bars
    sem1 = stats.sem(data1)
    sem2 = stats.sem(data2)
    ax.errorbar([1], [mean1], yerr=[sem1], color=color1, linewidth=2, 
               capsize=5, capthick=2, alpha=0.9)
    ax.errorbar([2], [mean2], yerr=[sem2], color=color2, linewidth=2, 
               capsize=5, capthick=2, alpha=0.9)
    
    # Formatting
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlim(0.5, 2.5)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Format p-value properly
    if p_val < 0.001:
        p_text = "p < 0.001"
    elif p_val < 0.01:
        p_text = "p < 0.01"
    elif p_val < 0.05:
        p_text = "p < 0.05"
    else:
        p_text = f"p = {p_val:.3f}"
    
    # Add stats text
    stats_text = f'Cohen\'s d: {cohens_d:.2f}\n{p_text}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           va='top', ha='left', fontsize=10,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                   alpha=0.9, edgecolor='gray', linewidth=0.5))
    
    return ax

def analyze_interbrain_vs_surrogates(empirical_df, shuffled_df, markov_df, output_dir):
    """
    Run the surrogate analysis with corrected statistics
    """
    
    print("=== CORRECTED: MATCHED SURROGATE ANALYSIS ===")
    print(f"Empirical trials: {len(empirical_df)}")
    print(f"Shuffled surrogate trials: {len(shuffled_df)}")
    print(f"Markov surrogate trials: {len(markov_df)}")
    
    # Key metrics to analyze
    key_metrics = [
        'inter_shannon_entropy',
        'inter_entropy_rate', 
        'inter_excess_entropy',
        'shared_ratio_time',
        'shared_mean_duration',
        'shared_occurrences_per_sec'
    ]
    
    # Find available metrics
    available_metrics = [m for m in key_metrics if m in empirical_df.columns]
    print(f"Available metrics: {available_metrics}")
    
    if len(available_metrics) == 0:
        print("❌ No suitable metrics found!")
        return None, None, [], []
    
    # Match surrogates to original trials
    results = []
    
    for _, emp_trial in empirical_df.iterrows():
        dyad = emp_trial['dyad']
        trial = emp_trial['trial']
        
        # Find matching surrogate trials for this specific empirical trial
        shuffled_matches = shuffled_df[
            (shuffled_df['dyad'] == dyad) & (shuffled_df['trial'] == trial)
        ]
        markov_matches = markov_df[
            (markov_df['dyad'] == dyad) & (markov_df['trial'] == trial)
        ]
        
        # Skip if we don't have surrogate matches
        if len(shuffled_matches) == 0 and len(markov_matches) == 0:
            continue
        
        trial_result = {
            'dyad': dyad,
            'trial': trial,
            'modality': emp_trial.get('modality', 'unknown'),
            'result': emp_trial.get('result', 'unknown'),
            'trial_duration': emp_trial.get('trial_duration', np.nan)
        }
        
        # For each metric, compare empirical vs surrogate distributions
        for metric in available_metrics:
            emp_value = emp_trial[metric]
            
            # Shuffled comparison
            if len(shuffled_matches) > 0:
                shuffled_values = shuffled_matches[metric].dropna()
                if len(shuffled_values) > 0:
                    p_shuffled = (np.sum(shuffled_values >= emp_value) + 1) / (len(shuffled_values) + 1)
                    shuffled_mean = shuffled_values.mean()
                    shuffled_std = shuffled_values.std()
                else:
                    p_shuffled = np.nan
                    shuffled_mean = np.nan
                    shuffled_std = np.nan
            else:
                p_shuffled = np.nan
                shuffled_mean = np.nan
                shuffled_std = np.nan
            
            # Markov comparison
            if len(markov_matches) > 0:
                markov_values = markov_matches[metric].dropna()
                if len(markov_values) > 0:
                    p_markov = (np.sum(markov_values >= emp_value) + 1) / (len(markov_values) + 1)
                    markov_mean = markov_values.mean()
                    markov_std = markov_values.std()
                else:
                    p_markov = np.nan
                    markov_mean = np.nan
                    markov_std = np.nan
            else:
                p_markov = np.nan
                markov_mean = np.nan
                markov_std = np.nan
            
            # Store trial-level results
            trial_result.update({
                f'{metric}_empirical': emp_value,
                f'{metric}_shuffled_mean': shuffled_mean,
                f'{metric}_shuffled_std': shuffled_std,
                f'{metric}_markov_mean': markov_mean,
                f'{metric}_markov_std': markov_std,
                f'{metric}_p_vs_shuffled': p_shuffled,
                f'{metric}_p_vs_markov': p_markov,
                f'{metric}_n_shuffled': len(shuffled_matches),
                f'{metric}_n_markov': len(markov_matches)
            })
        
        results.append(trial_result)
    
    # Convert to DataFrame
    matched_results_df = pd.DataFrame(results)
    print(f"Successfully matched {len(matched_results_df)} trials")
    
    # Statistical analysis across all trials
    statistical_results = []
    
    for metric in available_metrics:
        print(f"\n--- Analyzing {metric} ---")
        
        # Get empirical vs surrogate mean comparisons
        emp_col = f'{metric}_empirical'
        shuff_col = f'{metric}_shuffled_mean'
        markov_col = f'{metric}_markov_mean'
        
        # Valid data
        valid_data = matched_results_df[[emp_col, shuff_col, markov_col]].dropna()
        
        if len(valid_data) < 5:
            print(f"Insufficient data for {metric}")
            continue
        
        emp_vals = valid_data[emp_col]
        shuff_vals = valid_data[shuff_col] 
        markov_vals = valid_data[markov_col]
        
        print(f"Empirical: {emp_vals.mean():.4f} ± {emp_vals.std():.4f}")
        print(f"Shuffled:  {shuff_vals.mean():.4f} ± {shuff_vals.std():.4f}")
        print(f"Markov:    {markov_vals.mean():.4f} ± {markov_vals.std():.4f}")
        
        # Paired t-tests (since surrogates are matched to empirical trials)
        t_shuff, p_shuff = stats.ttest_rel(emp_vals, shuff_vals)
        t_markov, p_markov = stats.ttest_rel(emp_vals, markov_vals)
        
        # Effect sizes
        d_shuff = calculate_cohens_d_paired(emp_vals, shuff_vals)
        d_markov = calculate_cohens_d_paired(emp_vals, markov_vals)
        
        print(f"Paired t-test vs Shuffled: t={t_shuff:.3f}, p={p_shuff:.4f}, d={d_shuff:.3f} ({interpret_cohens_d(d_shuff)})")
        print(f"Paired t-test vs Markov: t={t_markov:.3f}, p={p_markov:.4f}, d={d_markov:.3f} ({interpret_cohens_d(d_markov)})")
        
        # Mixed-effects model controlling for confounds
        try:
            # Prepare data for mixed model
            model_data = matched_results_df[['dyad', 'trial_duration', 'modality', emp_col, shuff_col, markov_col]].dropna()
            
            # Long format for mixed model
            model_long = []
            for _, row in model_data.iterrows():
                model_long.extend([
                    {'dyad': row['dyad'], 'trial_duration': row['trial_duration'], 
                     'modality': row['modality'], 'value': row[emp_col], 'condition': 'empirical'},
                    {'dyad': row['dyad'], 'trial_duration': row['trial_duration'], 
                     'modality': row['modality'], 'value': row[shuff_col], 'condition': 'shuffled'},
                    {'dyad': row['dyad'], 'trial_duration': row['trial_duration'], 
                     'modality': row['modality'], 'value': row[markov_col], 'condition': 'markov'}
                ])
            
            model_df = pd.DataFrame(model_long)
            
            # Mixed model: Empirical vs Shuffled
            model_df_shuff = model_df[model_df['condition'].isin(['empirical', 'shuffled'])].copy()
            model_df_shuff['empirical_dummy'] = (model_df_shuff['condition'] == 'empirical').astype(int)
            
            formula = f'value ~ empirical_dummy + trial_duration + modality + (1|dyad)'
            mixed_model_shuff = smf.mixedlm(formula, model_df_shuff, groups=model_df_shuff['dyad']).fit()
            mixed_p_shuff = mixed_model_shuff.pvalues['empirical_dummy']
            
            # Mixed model: Empirical vs Markov
            model_df_markov = model_df[model_df['condition'].isin(['empirical', 'markov'])].copy()
            model_df_markov['empirical_dummy'] = (model_df_markov['condition'] == 'empirical').astype(int)
            
            mixed_model_markov = smf.mixedlm(formula, model_df_markov, groups=model_df_markov['dyad']).fit()
            mixed_p_markov = mixed_model_markov.pvalues['empirical_dummy']
            
            print(f"Mixed model vs Shuffled: p={mixed_p_shuff:.4f}")
            print(f"Mixed model vs Markov: p={mixed_p_markov:.4f}")
            
        except Exception as e:
            print(f"Mixed model failed: {e}")
            mixed_p_shuff = np.nan
            mixed_p_markov = np.nan
        
        # Count how many trials show empirical > surrogate
        p_vals_shuff = matched_results_df[f'{metric}_p_vs_shuffled'].dropna()
        p_vals_markov = matched_results_df[f'{metric}_p_vs_markov'].dropna()
        
        n_significant_shuff = np.sum(p_vals_shuff < 0.05) if len(p_vals_shuff) > 0 else 0
        n_significant_markov = np.sum(p_vals_markov < 0.05) if len(p_vals_markov) > 0 else 0
        
        print(f"Trials significantly above shuffled: {n_significant_shuff}/{len(p_vals_shuff)}")
        print(f"Trials significantly above Markov: {n_significant_markov}/{len(p_vals_markov)}")
        
        statistical_results.append({
            'metric': metric,
            'empirical_mean': emp_vals.mean(),
            'shuffled_mean': shuff_vals.mean(),
            'markov_mean': markov_vals.mean(),
            't_vs_shuffled': t_shuff,
            'p_vs_shuffled': p_shuff,
            'd_vs_shuffled': d_shuff,
            't_vs_markov': t_markov,
            'p_vs_markov': p_markov,
            'd_vs_markov': d_markov,
            'mixed_p_vs_shuffled': mixed_p_shuff,
            'mixed_p_vs_markov': mixed_p_markov,
            'n_trials_above_shuffled': n_significant_shuff,
            'n_trials_above_markov': n_significant_markov,
            'total_trials': len(valid_data)
        })
    
    # Apply FDR correction
    stat_df = pd.DataFrame(statistical_results)
    
    if len(stat_df) > 1:
        # FDR correction
        _, p_fdr_shuff, _, _ = multipletests(stat_df['p_vs_shuffled'], method='fdr_bh')
        _, p_fdr_markov, _, _ = multipletests(stat_df['p_vs_markov'], method='fdr_bh')
        
        stat_df['p_fdr_vs_shuffled'] = p_fdr_shuff
        stat_df['p_fdr_vs_markov'] = p_fdr_markov
        stat_df['sig_vs_shuffled'] = p_fdr_shuff < 0.05
        stat_df['sig_vs_markov'] = p_fdr_markov < 0.05
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    matched_results_df.to_csv(output_dir / "matched_surrogate_trial_results.csv", index=False)
    stat_df.to_csv(output_dir / "matched_surrogate_statistics.csv", index=False)
    
    return matched_results_df, stat_df

def analyze_interbrain_vs_surrogates_corrected(empirical_df, shuffled_df, markov_df, output_dir):
    """
    Run the surrogate analysis with corrected statistics and proper figures
    """
    
    # Run the analysis
    result = analyze_interbrain_vs_surrogates(empirical_df, shuffled_df, markov_df, output_dir)
    
    if result[0] is None:
        return None, None, [], []
    
    matched_results_df, stat_df = result
    
    # Get significant metrics
    significant_shuffled = stat_df[stat_df.get('sig_vs_shuffled', False)]['metric'].tolist() if 'sig_vs_shuffled' in stat_df.columns else []
    significant_markov = stat_df[stat_df.get('sig_vs_markov', False)]['metric'].tolist() if 'sig_vs_markov' in stat_df.columns else []
    
    print(f"\n=== CORRECTED FINDINGS ===")
    if len(stat_df) > 0:
        sig_shuff = len(significant_shuffled)
        sig_markov = len(significant_markov)
        
        print(f"Metrics significantly above shuffled: {sig_shuff}/{len(stat_df)}")
        print(f"Metrics significantly above Markov: {sig_markov}/{len(stat_df)}")
        
        for _, row in stat_df.iterrows():
            metric = row['metric']
            print(f"\n{metric}:")
            print(f"  vs Shuffled: p={row.get('p_fdr_vs_shuffled', row['p_vs_shuffled']):.3f}")
            print(f"  vs Markov: p={row.get('p_fdr_vs_markov', row['p_vs_markov']):.3f}")
    
    # Generate clean figures
    print(f"\n📊 Generating clean thesis figures...")
    generate_surrogate_figures(
        matched_results_df, stat_df, output_dir, significant_shuffled, significant_markov
    )
    
    print(f"\n✅ Corrected analysis complete!")
    print(f"Generated figures for {len(significant_shuffled)} vs shuffled, {len(significant_markov)} vs Markov")
    
    return matched_results_df, stat_df, significant_shuffled, significant_markov

def generate_surrogate_figures(matched_results_df, stat_df, output_dir, significant_shuffled, significant_markov):
    """
    
    """


    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    output_dir = Path(output_dir)
    
    empirical_color = "#0173B2"      # Blue
    shuffled_color = "#E09D47"       # Orange
    markov_color = "#CE6A51"         # Red-orange
    
    # Figure 1: Significant vs Shuffled
    if significant_shuffled:
        n_metrics = len(significant_shuffled)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows > 1:
            axes = axes.flatten()
        
        fig.suptitle('Significant Results: Empirical vs Shuffled Surrogates', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        for i, metric in enumerate(significant_shuffled):
            ax = axes[i]
            
            # Get data
            emp_col = f'{metric}_empirical'
            shuff_col = f'{metric}_shuffled_mean'
            
            emp_data = matched_results_df[emp_col].dropna()
            shuff_data = matched_results_df[shuff_col].dropna()
            
            # Get p-value and effect size
            metric_stats = stat_df[stat_df['metric'] == metric].iloc[0]
            p_val = metric_stats.get('p_fdr_vs_shuffled', metric_stats['p_vs_shuffled'])
            effect = metric_stats['d_vs_shuffled']
            
            # Create plot
            clean_title = metric.replace('inter_', '').replace('shared_', '').replace('_', ' ').title()
            ax = create_comparison_plot(ax, emp_data, shuff_data, empirical_color, shuffled_color, 
                                      clean_title, p_val, effect)
            
            ax.set_xticklabels(['Empirical', 'Shuffled'])
        
        # Remove empty subplots
        for j in range(len(significant_shuffled), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(output_dir / "significant_vs_shuffled.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "significant_vs_shuffled.pdf", bbox_inches='tight')
        plt.close()
    
    # Figure 2: Significant vs Markov
    if significant_markov:
        n_metrics = len(significant_markov)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows > 1:
            axes = axes.flatten()
        
        fig.suptitle('Significant Results: Empirical vs Markov Surrogates', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        for i, metric in enumerate(significant_markov):
            ax = axes[i]
            
            # Get data
            emp_col = f'{metric}_empirical'
            markov_col = f'{metric}_markov_mean'
            
            emp_data = matched_results_df[emp_col].dropna()
            markov_data = matched_results_df[markov_col].dropna()
            
            # Get p-value and effect size
            metric_stats = stat_df[stat_df['metric'] == metric].iloc[0]
            p_val = metric_stats.get('p_fdr_vs_markov', metric_stats['p_vs_markov'])
            effect = metric_stats['d_vs_markov']
            
            # Create plot
            clean_title = metric.replace('inter_', '').replace('shared_', '').replace('_', ' ').title()
            ax = create_comparison_plot(ax, emp_data, markov_data, empirical_color, markov_color, 
                                      clean_title, p_val, effect)
            
            ax.set_xticklabels(['Empirical', 'Markov'])
        
        # Remove empty subplots
        for j in range(len(significant_markov), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(output_dir / "significant_vs_markov.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "significant_vs_markov.pdf", bbox_inches='tight')
        plt.close()
    
    print("Files created:")
    if significant_shuffled:
        print("  - significant_vs_shuffled.png/pdf")
    if significant_markov:
        print("  - significant_vs_markov.png/pdf")

def generate_behavioral_figure(parameters_df, output_dir=None):
    """
    """

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    correct_color = "#419a66"        # Green
    incorrect_color = "#aa434f"      # Purple
    verbal_color = '#2980b9'         # Light blue
    gesture_color = '#e67e22'        # Dark orange
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Overall accuracy (with p-value from binomial test)
    total_trials = len(parameters_df)
    correct_trials = len(parameters_df[parameters_df['result'] == 'correct'])
    accuracy = correct_trials / total_trials
    
    bars = axes[0,0].bar(['Correct', 'Incorrect'], 
                        [accuracy, 1-accuracy], 
                        color=[correct_color, incorrect_color], 
                        alpha=0.8, width=0.6,
                        edgecolor='white', linewidth=1.5)
    
    axes[0,0].set_ylabel('Proportion of Trials', fontweight='bold', fontsize=12)
    axes[0,0].set_title('Overall Task Performance', fontweight='bold', fontsize=13, pad=15)
    axes[0,0].set_ylim(0, 1)
    
    # Add percentage labels
    for bar, pct in zip(bars, [accuracy, 1-accuracy]):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                      f'{pct:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 2. Accuracy by modality (bar plot with chi-square test)
    if 'modality' in parameters_df.columns:
        verbal_acc = (parameters_df[parameters_df['modality'] == 'verbal']['result'] == 'correct').mean()
        gesture_acc = (parameters_df[parameters_df['modality'] == 'gesture']['result'] == 'correct').mean()
        
        # Chi-square test for independence
        contingency_table = pd.crosstab(parameters_df['modality'], parameters_df['result'])
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
        
        # Effect size (Cramér's V)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        bars = axes[0,1].bar(['Verbal', 'Gesture'], 
                            [verbal_acc, gesture_acc],
                            color=[verbal_color, gesture_color], 
                            alpha=0.8, width=0.6,
                            edgecolor='white', linewidth=1.5)
        
        axes[0,1].set_ylabel('Accuracy', fontweight='bold', fontsize=12)
        axes[0,1].set_title('Accuracy by Modality', fontweight='bold', fontsize=13, pad=15)
        axes[0,1].set_ylim(0, 1)
        
        # Add percentage labels
        for bar, pct in zip(bars, [verbal_acc, gesture_acc]):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                          f'{pct:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Format p-value 
        if p_val < 0.001:
            p_text = "p < 0.001"
        elif p_val < 0.01:
            p_text = "p < 0.01"
        elif p_val < 0.05:
            p_text = "p < 0.05"
        else:
            p_text = f"p = {p_val:.3f}"
        
        # Add stats text
        stats_text = f'Cramér\'s V: {cramers_v:.2f}\n{p_text}'
        axes[0,1].text(0.02, 0.98, stats_text, transform=axes[0,1].transAxes, 
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    alpha=0.9, edgecolor='gray', linewidth=0.5))
    
    # 3. Duration by outcome
    if 'trial_duration' in parameters_df.columns:
        correct_durations = parameters_df[parameters_df['result'] == 'correct']['trial_duration'].dropna()
        incorrect_durations = parameters_df[parameters_df['result'] == 'incorrect']['trial_duration'].dropna()
        
        create_behavioral_comparison(
            axes[1,0], correct_durations, incorrect_durations,
            correct_color, incorrect_color,
            ['Correct', 'Incorrect'],
            'Duration by Outcome'
        )
        axes[1,0].set_ylabel('Trial Duration (s)', fontweight='bold', fontsize=12)
    
    # 4. Duration by modality
    if 'trial_duration' in parameters_df.columns and 'modality' in parameters_df.columns:
        verbal_durations = parameters_df[parameters_df['modality'] == 'verbal']['trial_duration'].dropna()
        gesture_durations = parameters_df[parameters_df['modality'] == 'gesture']['trial_duration'].dropna()
        
        create_behavioral_comparison(
            axes[1,1], verbal_durations, gesture_durations,
            verbal_color, gesture_color,
            ['Verbal', 'Gesture'],
            'Duration by Modality'
        )
        axes[1,1].set_ylabel('Trial Duration (s)', fontweight='bold', fontsize=12)
    
    plt.tight_layout(pad=2.0)
    
    if output_dir:
        plt.savefig(output_dir / "behavioral_results_clean.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "behavioral_results_clean.pdf", bbox_inches='tight')
        print(f"Clean behavioral figure saved to {output_dir}")
    
    plt.show()

def behavioral_analysis_extended(parameters_df, output_dir=None):
    """
    Extended behavioral analysis including duration by modality and trial counts
    """
    from scipy.stats import ttest_rel
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE BEHAVIORAL ANALYSIS (EXTENDED)")
    print("="*60)
    
    # ===== BASIC DESCRIPTIVES =====
    print("\n1. TRIAL COUNTS AND OVERALL ACCURACY")
    print("-" * 40)
    
    total_trials = len(parameters_df)
    correct_trials = len(parameters_df[parameters_df['result'] == 'correct'])
    overall_accuracy = correct_trials / total_trials
    
    print(f"Total trials: {total_trials}")
    print(f"Correct trials: {correct_trials}")
    print(f"Incorrect trials: {total_trials - correct_trials}")
    print(f"Overall accuracy: {overall_accuracy:.1%}")
    print(f"Total dyads: {parameters_df['dyad'].nunique()}")
    
    # ===== TRIAL COUNTS BY MODALITY =====
    print("\n2. TRIAL COUNTS BY MODALITY")
    print("-" * 40)
    
    if 'modality' in parameters_df.columns:
        modality_counts = parameters_df['modality'].value_counts()
        for modality, count in modality_counts.items():
            accuracy = (parameters_df[parameters_df['modality'] == modality]['result'] == 'correct').mean()
            print(f"{modality.capitalize()} trials: {count} ({accuracy:.1%} accuracy)")

        # Trial counts by modality
        verbal_trials = len(parameters_df[parameters_df['modality'] == 'verbal'])
        gesture_trials = len(parameters_df[parameters_df['modality'] == 'gesture'])
        
        # Accurate trials by modality
        verbal_correct = len(parameters_df[(parameters_df['modality'] == 'verbal') & (parameters_df['result'] == 'correct')])
        gesture_correct = len(parameters_df[(parameters_df['modality'] == 'gesture') & (parameters_df['result'] == 'correct')])
        
        print(f"\nTotal verbal trials: {verbal_trials} ({verbal_correct} correct)")
        print(f"Total gesture trials: {gesture_trials} ({gesture_correct} correct)")
        print(f"Verbal accuracy: {verbal_correct/verbal_trials:.1%}")
        print(f"Gesture accuracy: {gesture_correct/gesture_trials:.1%}")
    
    # ===== PAIRED T-TEST: VERBAL vs GESTURE ACCURACY (WITHIN DYADS) =====
    print("\n3. PAIRED T-TEST: VERBAL vs GESTURE ACCURACY")
    print("-" * 50)
    
    if 'modality' in parameters_df.columns and len(parameters_df['modality'].unique()) >= 2:
        
        # Calculate accuracy per dyad per modality
        dyad_accuracy = parameters_df.groupby(['dyad', 'modality'])['result'].apply(
            lambda x: (x == 'correct').mean()
        ).reset_index()
        dyad_accuracy.columns = ['dyad', 'modality', 'accuracy']
        
        # Pivot to get verbal and gesture accuracy side by side
        dyad_accuracy_wide = dyad_accuracy.pivot(index='dyad', columns='modality', values='accuracy')
        
        # Check which modalities we have
        available_modalities = list(dyad_accuracy_wide.columns)
        print(f"Available modalities: {available_modalities}")
        
        if len(available_modalities) >= 2:
            mod1, mod2 = available_modalities[0], available_modalities[1]
            
            # Only include dyads that have both modalities
            complete_dyads = dyad_accuracy_wide.dropna()
            
            if len(complete_dyads) > 0:
                mod1_scores = complete_dyads[mod1].values
                mod2_scores = complete_dyads[mod2].values
                
                # Paired t-test
                t_stat, p_value = ttest_rel(mod1_scores, mod2_scores)
                
                # Effect size (Cohen's d for paired samples)
                diff_scores = mod1_scores - mod2_scores
                cohens_d = np.mean(diff_scores) / np.std(diff_scores)
                
                # Descriptives
                mod1_mean = np.mean(mod1_scores)
                mod2_mean = np.mean(mod2_scores)
                mod1_std = np.std(mod1_scores)
                mod2_std = np.std(mod2_scores)
                
                print(f"\nDyads with both conditions: {len(complete_dyads)}")
                print(f"{mod1.capitalize()}: M = {mod1_mean:.3f}, SD = {mod1_std:.3f}")
                print(f"{mod2.capitalize()}: M = {mod2_mean:.3f}, SD = {mod2_std:.3f}")
                print(f"\nPaired t-test results:")
                print(f"t({len(complete_dyads)-1}) = {t_stat:.3f}")
                print(f"p = {p_value:.3f}")
                print(f"Cohen's d = {cohens_d:.3f}")
                
                # Interpretation
                if p_value < 0.001:
                    sig_level = "p < 0.001 (***)"
                elif p_value < 0.01:
                    sig_level = "p < 0.01 (**)"
                elif p_value < 0.05:
                    sig_level = "p < 0.05 (*)"
                else:
                    sig_level = "p ≥ 0.05 (ns)"
                
                print(f"Significance: {sig_level}")
    
    # ===== PAIRED T-TEST: DURATION BY MODALITY (WITHIN DYADS) =====
    print("\n4. PAIRED T-TEST: DURATIONS BY MODALITY")
    print("-" * 45)
    
    if 'trial_duration' in parameters_df.columns and 'modality' in parameters_df.columns:
        # Calculate mean duration per dyad per modality
        dyad_duration = parameters_df.groupby(['dyad', 'modality'])['trial_duration'].mean().reset_index()
        dyad_duration_wide = dyad_duration.pivot(index='dyad', columns='modality', values='trial_duration')
        
        if len(dyad_duration_wide.columns) >= 2:
            available_modalities = list(dyad_duration_wide.columns)
            mod1, mod2 = available_modalities[0], available_modalities[1]
            
            # Only include dyads with both modalities
            complete_dyads_dur = dyad_duration_wide.dropna()
            
            if len(complete_dyads_dur) > 0:
                mod1_durations = complete_dyads_dur[mod1].values
                mod2_durations = complete_dyads_dur[mod2].values
                
                # Paired t-test
                t_stat, p_value = ttest_rel(mod1_durations, mod2_durations)
                
                # Effect size
                diff_scores = mod1_durations - mod2_durations
                cohens_d = np.mean(diff_scores) / np.std(diff_scores)
                
                print(f"\nDyads with both conditions: {len(complete_dyads_dur)}")
                print(f"{mod1.capitalize()}: M = {np.mean(mod1_durations):.2f}s, SD = {np.std(mod1_durations):.2f}s")
                print(f"{mod2.capitalize()}: M = {np.mean(mod2_durations):.2f}s, SD = {np.std(mod2_durations):.2f}s")
                print(f"\nPaired t-test results:")
                print(f"t({len(complete_dyads_dur)-1}) = {t_stat:.3f}")
                print(f"p = {p_value:.3f}")
                print(f"Cohen's d = {cohens_d:.3f}")
                
                # Interpretation
                if p_value < 0.001:
                    sig_level = "p < 0.001 (***)"
                elif p_value < 0.01:
                    sig_level = "p < 0.01 (**)"
                elif p_value < 0.05:
                    sig_level = "p < 0.05 (*)"
                else:
                    sig_level = "p ≥ 0.05 (ns)"
                
                print(f"Significance: {sig_level}")
    
    # ===== INDEPENDENT T-TEST: CORRECT vs INCORRECT TRIAL DURATIONS =====
    print("\n5. INDEPENDENT T-TEST: TRIAL DURATIONS BY OUTCOME")
    print("-" * 50)
    
    if 'trial_duration' in parameters_df.columns:
        correct_durations = parameters_df[parameters_df['result'] == 'correct']['trial_duration'].dropna()
        incorrect_durations = parameters_df[parameters_df['result'] == 'incorrect']['trial_duration'].dropna()
        
        if len(correct_durations) > 0 and len(incorrect_durations) > 0:
            # Independent samples t-test
            t_stat, p_value = stats.ttest_ind(correct_durations, incorrect_durations)
            
            # Effect size (Cohen's d for independent samples)
            cohens_d = calculate_cohens_d_independent(correct_durations, incorrect_durations)
            
            # Descriptives
            correct_mean = correct_durations.mean()
            correct_std = correct_durations.std()
            incorrect_mean = incorrect_durations.mean()
            incorrect_std = incorrect_durations.std()
            
            print(f"Correct trials: M = {correct_mean:.2f}s, SD = {correct_std:.2f}s (n = {len(correct_durations)})")
            print(f"Incorrect trials: M = {incorrect_mean:.2f}s, SD = {incorrect_std:.2f}s (n = {len(incorrect_durations)})")
            print(f"\nIndependent t-test results:")
            print(f"t({len(correct_durations) + len(incorrect_durations) - 2}) = {t_stat:.3f}")
            print(f"p = {p_value:.3f}")
            print(f"Cohen's d = {cohens_d:.3f} ({interpret_cohens_d(cohens_d)})")
            
            # Interpretation
            if p_value < 0.001:
                sig_level = "p < 0.001 (***)"
            elif p_value < 0.01:
                sig_level = "p < 0.01 (**)"
            elif p_value < 0.05:
                sig_level = "p < 0.05 (*)"
            else:
                sig_level = "p ≥ 0.05 (ns)"
            
            print(f"Significance: {sig_level}")
    
    # ===== CREATE BEHAVIORAL PLOTS =====
    print("\n6. GENERATING BEHAVIORAL PLOTS")
    print("-" * 35)
    generate_behavioral_figure(parameters_df, output_dir)
    
    print("\n" + "="*60)
    print("BEHAVIORAL ANALYSIS COMPLETE")
    print("="*60)

def analyze_microstate_synchrony_patterns(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Analyze patterns of microstate synchrony across different conditions."""
    
    logger.info("Analyzing microstate synchrony patterns...")
    
    synchrony_features = [col for col in df.columns if any(x in col for x in 
                         ['shared', 'inter_entropy', 'inter_excess'])]
    
    if len(synchrony_features) == 0:
        logger.warning("No synchrony features found")
        return pd.DataFrame()
    
    results = []
    
    for dyad in df['dyad'].unique():
        dyad_data = df[df['dyad'] == dyad]
        
        if len(dyad_data) < 2:
            continue
        
        correct_trials = dyad_data[dyad_data['result'] == 'correct']
        incorrect_trials = dyad_data[dyad_data['result'] == 'incorrect']
        
        for feature in synchrony_features:
            if feature not in dyad_data.columns:
                continue
                
            correct_mean = correct_trials[feature].mean() if len(correct_trials) > 0 else np.nan
            incorrect_mean = incorrect_trials[feature].mean() if len(incorrect_trials) > 0 else np.nan
            
            within_dyad_effect = correct_mean - incorrect_mean
            consistency = 1 / (dyad_data[feature].var() + 1e-6)
            
            results.append({
                'dyad': dyad,
                'feature': feature,
                'correct_mean': correct_mean,
                'incorrect_mean': incorrect_mean,
                'within_dyad_effect': within_dyad_effect,
                'consistency': consistency,
                'n_trials': len(dyad_data)
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        results_df.to_csv(output_dir / "synchrony_patterns_analysis.csv", index=False)
    
    logger.info("Microstate synchrony patterns analysis completed")
    return results_df

def analyze_temporal_dynamics(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Analyze temporal dynamics of microstate sequences."""
    
    logger.info("Analyzing temporal dynamics...")
    
    duration_features = [col for col in df.columns if 'meandurs' in col]
    occurrence_features = [col for col in df.columns if 'occurrences' in col]
    
    if len(duration_features) == 0 or len(occurrence_features) == 0:
        logger.warning("Insufficient temporal features for analysis")
        return pd.DataFrame()
    
    results = []
    
    for feature in duration_features + occurrence_features:
        correct_vals = df[df['result'] == 'correct'][feature].dropna()
        incorrect_vals = df[df['result'] == 'incorrect'][feature].dropna()
        
        if len(correct_vals) < 5 or len(incorrect_vals) < 5:
            continue
        
        stat, p_val = stats.ttest_ind(correct_vals, incorrect_vals)
        
        pooled_std = np.sqrt(((len(correct_vals) - 1) * correct_vals.var() + 
                             (len(incorrect_vals) - 1) * incorrect_vals.var()) / 
                            (len(correct_vals) + len(incorrect_vals) - 2))
        
        effect_size = (correct_vals.mean() - incorrect_vals.mean()) / pooled_std
        
        results.append({
            'feature': feature,
            'feature_type': 'duration' if 'meandurs' in feature else 'occurrence',
            'microstate': feature.split('_')[-1] if '_' in feature else 'unknown',
            'role': 'sender' if 'sender' in feature else 'receiver' if 'receiver' in feature else 'shared',
            'correct_mean': correct_vals.mean(),
            'incorrect_mean': incorrect_vals.mean(),
            'effect_size': effect_size,
            'p_value': p_val,
            'stat': stat
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        _, p_fdr, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_fdr'] = p_fdr
        results_df['significant'] = results_df['p_fdr'] < 0.05
        
        results_df.to_csv(output_dir / "temporal_dynamics.csv", index=False)
    
    logger.info("Temporal dynamics analysis completed")
    return results_df

def analyze_transition_matrices(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Analyze transition matrix patterns between correct and incorrect trials."""
    
    logger.info("Analyzing transition matrix patterns...")
    
    transition_features = [col for col in df.columns if any(x in col for x in 
                          ['interbrain_prob', 'interbrain_A_leads', 'interbrain_B_leads', 
                           'interbrain_mutual_change', 'interbrain_coupling', 'interbrain_transition_entropy'])]
    
    if len(transition_features) == 0:
        logger.warning("No transition matrix features found")
        return pd.DataFrame()
    
    results = []
    
    correct_trials = df[df['result'] == 'correct']
    incorrect_trials = df[df['result'] == 'incorrect']
    
    for feature in transition_features:
        try:
            correct_values = correct_trials[feature].dropna()
            incorrect_values = incorrect_trials[feature].dropna()
            
            if len(correct_values) < 5 or len(incorrect_values) < 5:
                continue
            
            t_stat, p_value = stats.ttest_ind(correct_values, incorrect_values, equal_var=False)
            
            pooled_std = np.sqrt(((len(correct_values) - 1) * correct_values.var() + 
                                (len(incorrect_values) - 1) * incorrect_values.var()) / 
                               (len(correct_values) + len(incorrect_values) - 2))
            
            effect_size = (correct_values.mean() - incorrect_values.mean()) / pooled_std
            
            # Categorize transition features
            if 'prob_stay_sync' in feature:
                category = 'Synchrony Maintenance'
                interpretation = 'Higher = more stable synchrony'
            elif 'prob_become_sync' in feature:
                category = 'Synchrony Formation'
                interpretation = 'Higher = easier to achieve synchrony'
            elif 'A_leads' in feature or 'B_leads' in feature:
                category = 'Leadership Dynamics'
                interpretation = 'Higher = more leadership by this participant'
            elif 'mutual_change' in feature:
                category = 'Mutual Coordination'
                interpretation = 'Higher = more simultaneous changes'
            elif 'coupling' in feature:
                category = 'Microstate Coupling'
                interpretation = 'Higher = stronger coupling for this microstate'
            elif 'transition_entropy' in feature:
                category = 'Transition Predictability'
                interpretation = 'Higher = less predictable transitions'
            else:
                category = 'Other'
                interpretation = 'Unknown'
            
            results.append({
                'feature': feature,
                'category': category,
                'interpretation': interpretation,
                'correct_mean': correct_values.mean(),
                'incorrect_mean': incorrect_values.mean(),
                'correct_std': correct_values.std(),
                'incorrect_std': incorrect_values.std(),
                'effect_size': effect_size,
                'p_value': p_value,
                't_statistic': t_stat,
                'n_correct': len(correct_values),
                'n_incorrect': len(incorrect_values)
            })
            
        except Exception as e:
            logger.warning(f"Error processing {feature}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        _, p_fdr, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_fdr'] = p_fdr
        results_df['significant'] = results_df['p_fdr'] < 0.05
        
        results_df = results_df.sort_values('effect_size', key=abs, ascending=False)
        results_df.to_csv(output_dir / "transition_matrix_analysis.csv", index=False)
    
    logger.info("Transition matrix analysis completed")
    return results_df

def generate_thesis_ready_outputs(df: pd.DataFrame, results_df: pd.DataFrame, 
                                 output_dir: Path) -> None:
    """Generate publication-ready outputs for thesis."""
    
    logger.info("Generating thesis-ready outputs...")
    
    summary_stats = {}
    
    summary_stats['behavioral'] = {
        'accuracy_rate': (df['result'] == 'correct').mean(),
        'accuracy_std': (df['result'] == 'correct').std(),
        'mean_duration': df['trial_duration'].mean(),
        'duration_std': df['trial_duration'].std(),
        'total_trials': len(df),
        'total_dyads': df['dyad'].nunique()
    }
    
    if len(results_df) > 0:
        sig_results = results_df[results_df['significant']]
        summary_stats['microstate_findings'] = {
            'significant_features': len(sig_results),
            'largest_effect': sig_results.iloc[0]['effect_size'] if len(sig_results) > 0 else 0,
            'largest_effect_feature': sig_results.iloc[0]['feature'] if len(sig_results) > 0 else 'None',
            'mean_effect_size': sig_results['effect_size'].mean() if len(sig_results) > 0 else 0
        }
    
    pd.DataFrame([summary_stats]).to_csv(output_dir / "thesis_summary_stats.csv", index=False)
    
    logger.info("Thesis-ready outputs generated")