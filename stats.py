import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Union
import itertools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)
warnings.filterwarnings('ignore', message='.*to numeric.*')
warnings.filterwarnings('ignore', message='.*Unable to parse.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')
pd.options.mode.chained_assignment = None

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def run_statistical_analysis(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Run comprehensive statistical analysis comparing correct vs incorrect trials."""
    
    logger.info("Running main statistical analysis...")
    
    # Create binary correctness flag
    df["is_correct"] = (df["result"] == "correct").astype(int)
    
    correct_count = df[df['is_correct'] == 1].shape[0]
    incorrect_count = df[df['is_correct'] == 0].shape[0]
    logger.info(f"Total trials: {len(df)}")
    logger.info(f"Correct trials: {correct_count}")
    logger.info(f"Incorrect trials: {incorrect_count}")
    
    # Select numerical features
    numerical_df = df.select_dtypes(include=[np.number])
    numerical_features = numerical_df.columns.tolist()
    
    # Variables to ignore
    ignore = ['dyad', 'trial', 'modality', 'result', 'direction', 'is_correct', 
              "inter_p_Markov0", "inter_p_Markov1", "inter_p_Markov2"]
    confound_vars = ["trial_duration"] if "trial_duration" in df.columns else []
    ignore += confound_vars
    
    results = []
    
    for feature in numerical_features:
        if feature in ignore or feature.startswith('p_') or feature.startswith('T_'):
            continue
        
        correct_vals = df[df["is_correct"] == 1][feature].dropna()
        incorrect_vals = df[df["is_correct"] == 0][feature].dropna()
        
        if len(correct_vals) < 2 or len(incorrect_vals) < 2:
            logger.warning(f"Skipping {feature}: too few datapoints")
            continue
        
        # Check for near-zero variance
        if correct_vals.std() < 1e-10 or incorrect_vals.std() < 1e-10:
            logger.warning(f"Near-zero variance detected for {feature}")
            continue
        
        correct_mean = np.mean(correct_vals)
        incorrect_mean = np.mean(incorrect_vals)
        
        # Normality checks
        use_parametric = False
        if len(correct_vals) >= 8 and len(incorrect_vals) >= 8:
            from scipy.stats import normaltest
            _, p_norm_c = normaltest(correct_vals)
            _, p_norm_i = normaltest(incorrect_vals)
            use_parametric = p_norm_c > 0.05 and p_norm_i > 0.05
        
        # Statistical test
        if use_parametric:
            from scipy.stats import ttest_ind
            stat, p_val = ttest_ind(correct_vals, incorrect_vals, equal_var=False)
            test_used = "t-test"
            pooled_sd = np.sqrt(((np.std(correct_vals, ddof=1)**2 + np.std(incorrect_vals, ddof=1)**2) / 2))
            effect_size = (correct_mean - incorrect_mean) / pooled_sd
        else:
            from scipy.stats import mannwhitneyu
            stat, p_val = mannwhitneyu(correct_vals, incorrect_vals, alternative="two-sided")
            test_used = "Mann-Whitney U"
            effect_size = _cliffs_delta(correct_vals, incorrect_vals)
        
        # Mixed-effects model (your original approach)
        try:
            import statsmodels.formula.api as smf
            
            formula = f"{feature} ~ is_correct"
            if confound_vars:
                formula += " + " + " + ".join(confound_vars)
            
            model_df = df[["dyad", "is_correct", feature] + confound_vars].dropna()
            
            if model_df[feature].std() < 1e-6:
                logger.warning(f"Skipping {feature}: nearly zero variance")
                continue
            
            if model_df["is_correct"].nunique() < 2:
                logger.warning(f"Skipping {feature}: is_correct has only one class")
                continue
            
            model = smf.mixedlm(formula, model_df, groups=model_df["dyad"])
            result = model.fit(reml=False, method='cg')
            converged = result.converged if hasattr(result, 'converged') else True
            p_mixed = result.pvalues.get("is_correct", np.nan)
            
        except Exception as e:
            logger.warning(f"Mixed model error for {feature}: {e}")
            p_mixed, converged = np.nan, False
        
        results.append({
            "feature": feature,
            "test": test_used,
            "p_uncorrected": p_val,
            "statistic": stat,
            "effect_size": effect_size,
            "mixed_model_p": p_mixed,
            "mixed_model_converged": converged,
            "correct_mean": correct_mean,
            "incorrect_mean": incorrect_mean,
            "correct_std": np.std(correct_vals, ddof=1),
            "incorrect_std": np.std(incorrect_vals, ddof=1),
            "n_correct": len(correct_vals),
            "n_incorrect": len(incorrect_vals),
        })
    
    # Multiple comparisons correction
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        reject, p_fdr, _, _ = multipletests(results_df["mixed_model_p"], alpha=0.05, method='fdr_bh')
        results_df["p_fdr"] = p_fdr
        results_df["significant"] = reject
        
        # Sort by effect size
        results_df = results_df.sort_values('effect_size', key=abs, ascending=False)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "statistical_results.csv", index=False)
    
    # Summary
    n_sig = len(results_df[results_df["significant"]]) if len(results_df) > 0 else 0
    logger.info(f"Found {n_sig} significant features after FDR correction")
    
    return results_df

def _cliffs_delta(lst1, lst2):
    """Effect size using Cliff's Delta."""
    lst1, lst2 = np.array(lst1), np.array(lst2)
    delta = np.mean([np.sign(x - y) for x in lst1 for y in lst2])
    return delta

def surrogate_significance(empirical_df: pd.DataFrame, surrogate_df: pd.DataFrame, 
                         surrogate_type: str, output_dir: Path, 
                         features_of_interest: List[str]) -> pd.DataFrame:
    """
    Test significance against surrogate data.
    
    Args:
        empirical_df: Original data
        surrogate_df: Surrogate data
        surrogate_type: Type of surrogate ('shuffled' or 'markov')
        output_dir: Output directory
        features_of_interest: Features to test
        
    Returns:
        DataFrame with surrogate test results
    """
    
    logger.info(f"Running {surrogate_type} surrogate significance testing...")
    
    results = []
    
    for feature in features_of_interest:
        if feature not in empirical_df.columns or feature not in surrogate_df.columns:
            logger.warning(f"Feature {feature} not found in data")
            continue
            
        try:
            # Get empirical values
            empirical_correct = empirical_df[empirical_df['result'] == 'correct'][feature].dropna()
            empirical_incorrect = empirical_df[empirical_df['result'] == 'incorrect'][feature].dropna()
            
            if len(empirical_correct) < 5 or len(empirical_incorrect) < 5:
                continue
            
            # Calculate empirical effect size
            empirical_effect = calculate_effect_size(empirical_correct, empirical_incorrect)
            
            # Get surrogate distribution
            surrogate_effects = []
            
            # Group surrogates by trial
            for (dyad, trial), group in surrogate_df.groupby(['dyad', 'trial']):
                if len(group) < 2:  # Need at least 2 surrogates per trial
                    continue
                    
                # Get corresponding empirical trial info
                empirical_trial = empirical_df[
                    (empirical_df['dyad'] == dyad) & 
                    (empirical_df['trial'] == trial)
                ]
                
                if len(empirical_trial) == 0:
                    continue
                    
                result = empirical_trial['result'].iloc[0]
                
                # Calculate surrogate effects for this trial
                surrogate_values = group[feature].dropna()
                
                if len(surrogate_values) > 0:
                    # Simple approach: use surrogate distribution directly
                    surrogate_effects.extend(surrogate_values.tolist())
            
            if len(surrogate_effects) == 0:
                continue
            
            # Calculate percentile-based p-value
            empirical_p = calculate_surrogate_p_value(empirical_effect, surrogate_effects)
            
            # Store results
            results.append({
                'feature': feature,
                'empirical_effect': empirical_effect,
                'surrogate_mean': np.mean(surrogate_effects),
                'surrogate_std': np.std(surrogate_effects),
                'empirical_p': empirical_p,
                'surrogate_type': surrogate_type,
                'n_surrogates': len(surrogate_effects)
            })
            
        except Exception as e:
            logger.warning(f"Error in surrogate testing for {feature}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Multiple testing correction
    if len(results_df) > 0:
        _, p_fdr, _, _ = multipletests(results_df['empirical_p'], method='fdr_bh')
        results_df['p_fdr'] = p_fdr
        results_df['significant_surrogate'] = results_df['p_fdr'] < 0.05
    
    # Save results
    results_df.to_csv(output_dir / f"{surrogate_type}_surrogate_results.csv", index=False)
    
    logger.info(f"Completed {surrogate_type} surrogate testing")
    
    return results_df

def calculate_effect_size(group1: pd.Series, group2: pd.Series) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * group1.std()**2 + (n2 - 1) * group2.std()**2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std

def calculate_surrogate_p_value(empirical_value: float, surrogate_distribution: List[float]) -> float:
    """Calculate p-value based on surrogate distribution."""
    surrogate_array = np.array(surrogate_distribution)
    
    # Two-tailed test
    if empirical_value >= 0:
        p_value = 2 * (np.sum(surrogate_array >= empirical_value) / len(surrogate_array))
    else:
        p_value = 2 * (np.sum(surrogate_array <= empirical_value) / len(surrogate_array))
    
    return min(p_value, 1.0)  # Ensure p-value doesn't exceed 1

def plot_significant_features(df: pd.DataFrame, results_df: pd.DataFrame, 
                            output_dir: Path, max_features: int = 12) -> None:
    """Plot significant features as violin plots."""
    
    logger.info("Creating significant features plots...")
    
    # Get significant features
    sig_features = results_df[results_df['significant']].head(max_features)
    
    if len(sig_features) == 0:
        logger.warning("No significant features to plot")
        return
    
    # Create subplot grid
    n_features = len(sig_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(sig_features.iterrows()):
        ax = axes[i]
        feature = row['feature']
        
        # Create violin plot
        data_to_plot = [
            df[df['result'] == 'correct'][feature].dropna(),
            df[df['result'] == 'incorrect'][feature].dropna()
        ]
        
        parts = ax.violinplot(data_to_plot, positions=[1, 2], widths=0.6)
        
        # Color the violins
        colors = ['lightgreen', 'lightcoral']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Add statistics
        ax.text(0.02, 0.98, f"d = {row['effect_size']:.3f}\np = {row['p_fdr']:.3f}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Correct', 'Incorrect'])
        ax.set_title(feature.replace('_', ' ').title(), fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "significant_features.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Significant features plot saved")

def plot_transition_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot transition matrix analysis."""
    
    logger.info("Creating transition analysis plots...")
    
    # This is a placeholder - would need transition matrices from microstate analysis
    # For now, create a summary plot of microstate coverage
    
    microstate_cols = [col for col in df.columns if 'timecov' in col]
    
    if len(microstate_cols) == 0:
        logger.warning("No microstate coverage features found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sender microstates
    sender_cols = [col for col in microstate_cols if 'sender' in col]
    if sender_cols:
        sender_data = df[sender_cols].mean()
        
        axes[0].bar(range(len(sender_data)), sender_data.values)
        axes[0].set_xticks(range(len(sender_data)))
        axes[0].set_xticklabels([col.split('_')[-1] for col in sender_data.index])
        axes[0].set_title('Sender Microstate Coverage')
        axes[0].set_ylabel('Time Coverage')
    
    # Receiver microstates
    receiver_cols = [col for col in microstate_cols if 'receiver' in col]
    if receiver_cols:
        receiver_data = df[receiver_cols].mean()
        
        axes[1].bar(range(len(receiver_data)), receiver_data.values)
        axes[1].set_xticks(range(len(receiver_data)))
        axes[1].set_xticklabels([col.split('_')[-1] for col in receiver_data.index])
        axes[1].set_title('Receiver Microstate Coverage')
        axes[1].set_ylabel('Time Coverage')
    
    plt.tight_layout()
    plt.savefig(output_dir / "transition_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Transition analysis plot saved")

def plot_surrogate_results(df: pd.DataFrame, shuffled_results: pd.DataFrame, 
                          markov_results: pd.DataFrame, output_dir: Path) -> None:
    """Plot surrogate testing results."""
    
    logger.info("Creating surrogate results plots...")
    
    # Combine surrogate results
    shuffled_results['surrogate_type'] = 'shuffled'
    markov_results['surrogate_type'] = 'markov'
    combined_surrogates = pd.concat([shuffled_results, markov_results], ignore_index=True)
    
    if len(combined_surrogates) == 0:
        logger.warning("No surrogate results to plot")
        return
    
    # Get features present in both surrogate types
    common_features = set(shuffled_results['feature']) & set(markov_results['feature'])
    
    if len(common_features) == 0:
        logger.warning("No common features between surrogate types")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Effect size comparison
    for i, feature in enumerate(list(common_features)[:8]):  # Limit to 8 features
        shuffled_row = shuffled_results[shuffled_results['feature'] == feature].iloc[0]
        markov_row = markov_results[markov_results['feature'] == feature].iloc[0]
        
        axes[0].scatter(shuffled_row['empirical_effect'], markov_row['empirical_effect'], 
                       s=60, alpha=0.7, label=feature if i < 5 else "")
    
    axes[0].set_xlabel('Shuffled Surrogate Effect')
    axes[0].set_ylabel('Markov Surrogate Effect')
    axes[0].set_title('Empirical Effects: Shuffled vs Markov')
    axes[0].plot([-2, 2], [-2, 2], 'k--', alpha=0.5)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # P-value comparison
    shuffled_p = shuffled_results['empirical_p'].values
    markov_p = markov_results['empirical_p'].values
    
    axes[1].scatter(shuffled_p, markov_p, s=60, alpha=0.7)
    axes[1].set_xlabel('Shuffled Surrogate p-value')
    axes[1].set_ylabel('Markov Surrogate p-value')
    axes[1].set_title('P-values: Shuffled vs Markov')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='p=0.05')
    axes[1].axvline(x=0.05, color='r', linestyle='--', alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "surrogate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Surrogate comparison plot saved")

# Traditional transition probability analysis functions
def run_traditional_tp_analysis(parameters_df, output_dir):
    """
    Run traditional transition probability analysis for both intra-brain and inter-brain
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting traditional transition probability analysis...")
    
    # Run inter-brain analysis (this is what we have data for)
    inter_results = analyze_interbrain_tp_traditional(parameters_df, output_dir)
    
    # Create summary report
    create_tp_summary_report({'inter_brain': inter_results}, output_dir)
    
    logger.info("Traditional TP analysis completed!")
    return {'inter_brain': inter_results}

def analyze_interbrain_tp_traditional(parameters_df, output_dir):
    """
    Traditional inter-brain TP analysis (16x16 matrices)
    """
    
    logger.info("Analyzing inter-brain transition probabilities...")
    
    # Find transition matrix columns
    transition_cols = [col for col in parameters_df.columns if col.startswith('T_') and '_to_' in col]
    
    if len(transition_cols) == 0:
        logger.error("No transition columns found!")
        return None
    
    # Separate by condition
    correct_trials = parameters_df[parameters_df['result'] == 'correct']
    incorrect_trials = parameters_df[parameters_df['result'] == 'incorrect']
    
    # Calculate average TP matrices
    states = ['A_A', 'A_B', 'A_C', 'A_D', 'B_A', 'B_B', 'B_C', 'B_D', 
              'C_A', 'C_B', 'C_C', 'C_D', 'D_A', 'D_B', 'D_C', 'D_D']
    
    correct_matrix = build_tp_matrix_from_df(correct_trials, transition_cols, states)
    incorrect_matrix = build_tp_matrix_from_df(incorrect_trials, transition_cols, states)
    
    # Expected matrix (uniform random)
    expected_matrix = np.full((16, 16), 1/16)
    
    # Create visualizations
    create_interbrain_tp_plots(correct_matrix, incorrect_matrix, expected_matrix, states, output_dir)
    
    # Statistical analysis
    stats_results = run_interbrain_tp_stats(correct_trials, incorrect_trials, transition_cols, output_dir)
    
    return {
        'correct_matrix': correct_matrix,
        'incorrect_matrix': incorrect_matrix,
        'expected_matrix': expected_matrix,
        'statistics': stats_results
    }

def build_tp_matrix_from_df(df, transition_cols, states):
    """
    Build 16x16 transition probability matrix from dataframe
    """
    
    matrix = np.zeros((16, 16))
    
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            col_name = f'T_{from_state}_to_{to_state}'
            if col_name in transition_cols:
                values = df[col_name].dropna()
                if len(values) > 0:
                    matrix[i, j] = values.mean()
    
    return matrix

def create_interbrain_tp_plots(correct_matrix, incorrect_matrix, expected_matrix, states, output_dir):
    """
    Create inter-brain TP plots (16x16 matrices)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Observed matrices
    sns.heatmap(correct_matrix, annot=False, fmt='.2f',
               xticklabels=states, yticklabels=states,
               ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title('Correct Trials')
    axes[0, 0].set_xlabel('To State')
    axes[0, 0].set_ylabel('From State')
    
    sns.heatmap(incorrect_matrix, annot=False, fmt='.2f',
               xticklabels=states, yticklabels=states,
               ax=axes[0, 1], cmap='Reds')
    axes[0, 1].set_title('Incorrect Trials')
    axes[0, 1].set_xlabel('To State')
    axes[0, 1].set_ylabel('From State')
    
    # Expected matrix
    sns.heatmap(expected_matrix, annot=False, fmt='.2f',
               xticklabels=states, yticklabels=states,
               ax=axes[1, 0], cmap='Greys')
    axes[1, 0].set_title('Expected (Random)')
    axes[1, 0].set_xlabel('To State')
    axes[1, 0].set_ylabel('From State')
    
    # Difference matrix
    diff_matrix = correct_matrix - incorrect_matrix
    sns.heatmap(diff_matrix, annot=False, fmt='.3f',
               xticklabels=states, yticklabels=states,
               ax=axes[1, 1], cmap='RdBu_r', center=0)
    axes[1, 1].set_title('Difference (Correct - Incorrect)')
    axes[1, 1].set_xlabel('To State')
    axes[1, 1].set_ylabel('From State')
    
    plt.suptitle('Inter-brain Transition Probabilities (16x16)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "interbrain_tp_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

def run_interbrain_tp_stats(correct_trials, incorrect_trials, transition_cols, output_dir):
    """
    Run statistical tests for inter-brain transitions (with proper mixed-effects)
    """
    
    logger.info("Running inter-brain TP statistics...")
    
    # Import for mixed-effects
    try:
        from statsmodels.formula.api import mixedlm
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        logger.error("statsmodels not available for mixed-effects analysis")
        return {}
    
    results = []
    
    # Test each transition probability
    for col in transition_cols:
        try:
            # Combine data
            combined_data = pd.concat([
                correct_trials[['dyad', 'trial_duration', col, 'result']],
                incorrect_trials[['dyad', 'trial_duration', col, 'result']]
            ]).dropna()
            
            if len(combined_data) < 20:  # Minimum sample size
                continue
            
            # Create numeric result variable
            combined_data['result_numeric'] = (combined_data['result'] == 'correct').astype(int)
            
            # Mixed-effects model
            formula = f'{col} ~ result_numeric + trial_duration'
            model = mixedlm(formula, combined_data, groups=combined_data['dyad'])
            fitted_model = model.fit(method='lbfgs')
            
            # Extract results
            result_coef = fitted_model.params.get('result_numeric', np.nan)
            result_pval = fitted_model.pvalues.get('result_numeric', np.nan)
            
            # Effect size
            residual_std = np.sqrt(fitted_model.scale)
            effect_size = result_coef / residual_std if residual_std > 0 else np.nan
            
            results.append({
                'transition': col,
                'coefficient': result_coef,
                'p_value': result_pval,
                'effect_size': effect_size,
                'n_trials': len(combined_data),
                'converged': fitted_model.converged
            })
            
        except Exception as e:
            logger.warning(f"Failed to analyze {col}: {e}")
            continue
    
    # Convert to DataFrame and apply corrections
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        
        # Multiple testing correction
        valid_p = ~results_df['p_value'].isna()
        if valid_p.sum() > 0:
            _, p_fdr, _, _ = multipletests(results_df.loc[valid_p, 'p_value'], method='fdr_bh')
            results_df.loc[valid_p, 'p_fdr'] = p_fdr
            results_df.loc[valid_p, 'significant'] = p_fdr < 0.05
        
        # Sort by effect size
        results_df = results_df.sort_values('effect_size', key=abs, ascending=False)
        
        # Save results
        results_df.to_csv(output_dir / "interbrain_tp_statistics.csv", index=False)
        
        n_sig = results_df['significant'].sum() if 'significant' in results_df.columns else 0
        logger.info(f"Inter-brain TP analysis: {n_sig} significant transitions after FDR correction")
        
        return results_df
    
    return pd.DataFrame()

def create_tp_summary_report(all_results, output_dir):
    """
    Create summary report of TP analysis
    """
    
    logger.info("Creating TP summary report...")
    
    summary = {
        'inter_brain_transitions_tested': len(all_results['inter_brain']['statistics']) if 'statistics' in all_results['inter_brain'] else 0,
        'inter_brain_significant': 0,
    }
    
    # Count significant results
    if isinstance(all_results['inter_brain']['statistics'], pd.DataFrame):
        if 'significant' in all_results['inter_brain']['statistics'].columns:
            summary['inter_brain_significant'] = all_results['inter_brain']['statistics']['significant'].sum()
    
    # Save summary
    pd.DataFrame([summary]).to_csv(output_dir / "tp_analysis_summary.csv", index=False)
    
    print(f"TP Analysis Summary:")
    print(f"  Inter-brain transitions tested: {summary['inter_brain_transitions_tested']}")
    print(f"  Inter-brain significant: {summary['inter_brain_significant']}")

# Figure generation functions
def generate_thesis_figures(parameters_df, output_dir):
    """
    Generate publication-ready figures for thesis posthoc results
    Based on your actual data structure with already computed surrogate statistics
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Key metrics based on your reported results - check what's actually in your data
    potential_metrics = [
        'inter_shannon_entropy',
        'inter_entropy_rate', 
        'inter_excess_entropy',
        'shared_ratio_time',
        'shared_mean_duration',
        'shared_occurrences_per_sec'
    ]
    
    # Find columns that match your metric patterns
    all_columns = parameters_df.columns.tolist()
    print(f"All columns in parameters_df: {all_columns[:20]}...")  # Show first 20
    
    # Look for empirical metric columns
    empirical_metrics = []
    surrogate_p_columns = []
    
    for metric in potential_metrics:
        # Look for the base metric
        if metric in all_columns:
            empirical_metrics.append(metric)
        # Look for p-value columns
        p_col_shuffled = f"{metric}_p_vs_shuffled"
        p_col_markov = f"{metric}_p_vs_markov"
        if p_col_shuffled in all_columns:
            surrogate_p_columns.append((metric, p_col_shuffled, p_col_markov))
    
    print(f"Found empirical metrics: {empirical_metrics}")
    print(f"Found surrogate p-value columns: {surrogate_p_columns}")
    
    # If no direct metrics found, let's look for any inter/shared columns
    if len(empirical_metrics) == 0:
        empirical_metrics = [col for col in all_columns if any(pattern in col for pattern in 
                           ['inter_', 'shared_']) and not any(suffix in col for suffix in 
                           ['_p_vs_', '_mean', '_std', '_n_'])]
        print(f"Found pattern-based metrics: {empirical_metrics}")
    
    available_metrics = empirical_metrics[:6]  # Limit to 6 for plotting
    print(f"Using metrics for plots: {available_metrics}")
    
    if len(available_metrics) == 0:
        print("❌ No suitable metrics found for plotting!")
        return pd.DataFrame()
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    results_summary = []
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        
        # Check if we have empirical data for this metric
        if metric not in parameters_df.columns:
            ax.text(0.5, 0.5, f'No data for\n{metric}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            continue
        
        # Get empirical data
        empirical_data = parameters_df[metric].dropna()
        
        # Try to find corresponding surrogate data columns
        shuffled_col = f"{metric}_shuffled_mean"
        markov_col = f"{metric}_markov_mean"
        p_shuffled_col = f"{metric}_p_vs_shuffled"
        p_markov_col = f"{metric}_p_vs_markov"
        
        # Handle different possible column naming patterns
        possible_shuffled = [shuffled_col, f"{metric}_shuff_mean", f"shuffled_{metric}"]
        possible_markov = [markov_col, f"{metric}_markov_mean", f"markov_{metric}"]
        
        shuffled_data = None
        markov_data = None
        p_shuffled = None
        p_markov = None
        
        # Find shuffled data
        for col in possible_shuffled:
            if col in parameters_df.columns:
                shuffled_data = parameters_df[col].dropna()
                break
        
        # Find markov data  
        for col in possible_markov:
            if col in parameters_df.columns:
                markov_data = parameters_df[col].dropna()
                break
        
        # Find p-values
        if p_shuffled_col in parameters_df.columns:
            p_shuffled = parameters_df[p_shuffled_col].dropna()
        if p_markov_col in parameters_df.columns:
            p_markov = parameters_df[p_markov_col].dropna()
        
        # If no surrogate data found, create a simple histogram
        if shuffled_data is None and markov_data is None:
            ax.hist(empirical_data, bins=20, alpha=0.7, color='#2E8B57', label='Empirical')
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency')
            ax.legend()
            
            # Add basic statistics
            mean_val = empirical_data.mean()
            std_val = empirical_data.std()
            ax.text(0.02, 0.98, f'M = {mean_val:.3f}\nSD = {std_val:.3f}', 
                    transform=ax.transAxes, va='top', ha='left', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            results_summary.append({
                'metric': metric,
                'empirical_mean': mean_val,
                'empirical_std': std_val,
                'n_trials': len(empirical_data),
                'shuffled_mean': np.nan,
                'markov_mean': np.nan,
                'p_vs_shuffled': np.nan,
                'p_vs_markov': np.nan
            })
            continue
        
        # Create comparison plot if surrogate data exists
        data_dict = {'Empirical': empirical_data}
        colors = ['#2E8B57']
        
        if shuffled_data is not None:
            data_dict['Shuffled'] = shuffled_data
            colors.append('#CD5C5C')
            
        if markov_data is not None:
            data_dict['Markov'] = markov_data
            colors.append('#4682B4')
        
        # Create violin plot
        positions = list(range(1, len(data_dict) + 1))
        violin_data = list(data_dict.values())
        
        violin_parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
        
        # Color the violins
        for patch, color in zip(violin_parts['bodies'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Calculate statistics
        emp_mean = empirical_data.mean()
        shuff_mean = shuffled_data.mean() if shuffled_data is not None else np.nan
        markov_mean = markov_data.mean() if markov_data is not None else np.nan
        
        # Statistical significance (use existing p-values if available)
        if p_shuffled is not None and len(p_shuffled) > 0:
            p_shuff_val = p_shuffled.median()  # Use median p-value across trials
        else:
            p_shuff_val = np.nan
            
        if p_markov is not None and len(p_markov) > 0:
            p_markov_val = p_markov.median()
        else:
            p_markov_val = np.nan
        
        # Add significance annotations
        y_max = max([data.max() for data in violin_data])
        y_range = y_max - min([data.min() for data in violin_data])
        
        # Add significance stars
        if not np.isnan(p_shuff_val):
            sig_shuff = '***' if p_shuff_val < 0.001 else '**' if p_shuff_val < 0.01 else '*' if p_shuff_val < 0.05 else 'ns'
            if len(positions) >= 2:
                ax.text(1.5, y_max + 0.1*y_range, sig_shuff, ha='center', fontsize=12, fontweight='bold')
        
        if not np.isnan(p_markov_val):
            sig_markov = '***' if p_markov_val < 0.001 else '**' if p_markov_val < 0.01 else '*' if p_markov_val < 0.05 else 'ns'
            if len(positions) >= 3:
                ax.text(2, y_max + 0.05*y_range, sig_markov, ha='center', fontsize=12, fontweight='bold')
        
        # Formatting
        ax.set_xticks(positions)
        ax.set_xticklabels(list(data_dict.keys()))
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Emp: {emp_mean:.3f}'
        if not np.isnan(shuff_mean):
            stats_text += f'\nShuff: {shuff_mean:.3f}'
        if not np.isnan(markov_mean):
            stats_text += f'\nMarkov: {markov_mean:.3f}'
            
        ax.text(0.02, 0.98, stats_text, 
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        results_summary.append({
            'metric': metric,
            'empirical_mean': emp_mean,
            'shuffled_mean': shuff_mean,
            'markov_mean': markov_mean,
            'p_vs_shuffled': p_shuff_val,
            'p_vs_markov': p_markov_val,
            'n_trials': len(empirical_data)
        })
    
    # Remove empty subplots
    for j in range(len(available_metrics), 6):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(output_dir / "posthoc_surrogate_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "posthoc_surrogate_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results_summary)
    
    if len(results_df) > 0:
        # Save results
        results_df.to_csv(output_dir / "posthoc_statistical_summary.csv", index=False)
        
        # Print summary for thesis
        print("\n" + "="*60)
        print("THESIS POSTHOC RESULTS SUMMARY")
        print("="*60)
        
        # Count significant results
        sig_shuffled = 0
        sig_markov = 0
        
        for _, row in results_df.iterrows():
            if not pd.isna(row['p_vs_shuffled']) and row['p_vs_shuffled'] < 0.05:
                sig_shuffled += 1
            if not pd.isna(row['p_vs_markov']) and row['p_vs_markov'] < 0.05:
                sig_markov += 1
        
        print(f"Metrics analyzed: {len(results_df)}")
        print(f"Significant vs Shuffled surrogates: {sig_shuffled}/{len(results_df)} metrics")
        print(f"Significant vs Markov surrogates: {sig_markov}/{len(results_df)} metrics")
        
        print("\nResults summary:")
        for _, row in results_df.iterrows():
            metric = row['metric'].replace('_', ' ').title()
            print(f"\n{metric}:")
            print(f"  Empirical mean: {row['empirical_mean']:.4f}")
            if not pd.isna(row['shuffled_mean']):
                print(f"  Shuffled mean: {row['shuffled_mean']:.4f}")
            if not pd.isna(row['markov_mean']):
                print(f"  Markov mean: {row['markov_mean']:.4f}")
            if not pd.isna(row['p_vs_shuffled']):
                sig_status = "significant" if row['p_vs_shuffled'] < 0.05 else "not significant"
                print(f"  vs Shuffled: p = {row['p_vs_shuffled']:.4f} ({sig_status})")
            if not pd.isna(row['p_vs_markov']):
                sig_status = "significant" if row['p_vs_markov'] < 0.05 else "not significant"
                print(f"  vs Markov: p = {row['p_vs_markov']:.4f} ({sig_status})")
    
    return results_df