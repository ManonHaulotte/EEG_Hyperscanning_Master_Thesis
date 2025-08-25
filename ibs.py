import hypyp
from hypyp import prep, analyses, stats, viz
import mne
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Union
import re
from itertools import groupby
import itertools
import json
from joblib import Parallel, delayed
import logging
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
logger = logging.getLogger(__name__)
mne.set_log_level('ERROR')



def _skip_trial(trials, dyad, trial_num, reason):
    msg = f"Skipping dyad {dyad} trial {trial_num} — reason: {reason}"
    logger.warning(msg)
    trials.append({"dyad": dyad, "trial": trial_num, "reason": reason})
def _group_trials(input_dir):
    file_groups = {}

    for file in os.listdir(input_dir):
        if not file.endswith(".fif"):
            continue
        
        parts = file.replace('.fif', '').split("_")
        if len(parts) < 5:
            print(f"Skipping file with unexpected format: {file}")
            continue
        
        dyad_participant, trial, modality, result, direction = parts
        
        match = re.match(r"^(\d{2})([AB])$", dyad_participant)
        if not match:
            print(f"Skipping file with invalid dyad/participant: {file}")
            continue
        
        dyad, participant = match.groups()
        trial_num = trial.replace('trial', '')

        try:
            trial_num = int(trial_num)
        except ValueError:
            print(f"Skipping file with invalid trial number: {file}")
            continue

        file_groups.setdefault((dyad, trial_num), []).append({
            "filename": file,
            "participant": participant,
            "direction": direction,
            "modality": modality,
            "result": result
        })
    
    return file_groups
def run_ibs_analysis(input_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Simple PLV analysis focused on task sensitivity, following Chan et al. 2022 approach.
    Tests whether PLV shows same correct vs incorrect pattern as microstate measures.
    """
    
    logger.info("Running task-sensitive PLV analysis...")
    
    # Basic frequency bands
    freq_bands = {
        'alpha': (8, 12),
        'beta': (13, 30), 
        'gamma': (30, 45)
    }
    
    results = []
    file_groups = _group_trials(input_dir)  # Use same grouping as microstate
    
    processed = 0
    failed = 0
    
    for (dyad, trial_num), files in file_groups.items():
        print(f"Processing PLV for dyad {dyad}, trial {trial_num}")
        
        if len(files) != 2:
            print(f"  SKIP: Wrong number of files ({len(files)})")
            failed += 1
            continue
            
        try:
            # Use SAME logic as microstate extraction
            sender = next((f for f in files if (f["participant"] == "A" and f["direction"] == "AtoB") or 
                                             (f["participant"] == "B" and f["direction"] == "BtoA")), None)
            receiver = next((f for f in files if f != sender), None)
            
            if not sender or not receiver:
                print(f"  SKIP: Could not identify sender/receiver")
                print(f"    Files: {[(f['participant'], f['direction']) for f in files]}")
                failed += 1
                continue
            
            print(f"  Sender: {sender['filename']}")
            print(f"  Receiver: {receiver['filename']}")
            
            # Load files
            sender_raw = mne.io.read_raw_fif(
                os.path.join(input_dir, sender['filename']), preload=True, verbose=False
            )
            receiver_raw = mne.io.read_raw_fif(
                os.path.join(input_dir, receiver['filename']), preload=True, verbose=False
            )
            
            # Basic info
            sfreq = sender_raw.info['sfreq']
            trial_duration = (sender_raw.times[-1] - sender_raw.times[0])
            
            print(f"  Duration: {trial_duration:.2f}s, Fs: {sfreq}Hz")
            
            # For each frequency band, compute simple PLV
            band_results = {
                'dyad': dyad,
                'trial': trial_num,
                'modality': sender['modality'],
                'result': sender['result'],
                'trial_duration': trial_duration,
                'n_channels': sender_raw.info['nchan']
            }
            
            for band_name, (fmin, fmax) in freq_bands.items():
                try:
                    # Simple bandpass filtering
                    from scipy.signal import butter, filtfilt
                    nyq = sfreq / 2
                    low = fmin / nyq
                    high = fmax / nyq
                    
                    if high >= 1.0:  # Adjust if too high
                        high = 0.95
                    
                    b, a = butter(4, [low, high], btype='band')
                    
                    # Get data and filter
                    sender_data = sender_raw.get_data()
                    receiver_data = receiver_raw.get_data()
                    
                    filtered_sender = filtfilt(b, a, sender_data, axis=1)
                    filtered_receiver = filtfilt(b, a, receiver_data, axis=1)
                    
                    # Compute PLV across electrode pairs (simplified)
                    from scipy.signal import hilbert
                    
                    plv_values = []
                    n_channels = min(8, sender_data.shape[0])  # Use first 8 channels
                    
                    for ch in range(n_channels):
                        # Hilbert transform
                        sender_analytic = hilbert(filtered_sender[ch, :])
                        receiver_analytic = hilbert(filtered_receiver[ch, :])
                        
                        # Phase difference and PLV
                        sender_phase = np.angle(sender_analytic)
                        receiver_phase = np.angle(receiver_analytic)
                        phase_diff = sender_phase - receiver_phase
                        plv_ch = np.abs(np.mean(np.exp(1j * phase_diff)))
                        
                        plv_values.append(plv_ch)
                    
                    # Store band results
                    band_results[f'plv_mean_{band_name}'] = np.mean(plv_values)
                    band_results[f'plv_std_{band_name}'] = np.std(plv_values)
                    
                    print(f"    {band_name}: PLV = {np.mean(plv_values):.3f}")
                    
                except Exception as e:
                    print(f"    Error in {band_name} band: {e}")
                    band_results[f'plv_mean_{band_name}'] = np.nan
                    band_results[f'plv_std_{band_name}'] = np.nan
            
            results.append(band_results)
            processed += 1
            print(f"  SUCCESS")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            continue
    
    print(f"\nPLV Analysis Summary:")
    print(f"  Processed: {processed}")
    print(f"  Failed: {failed}")
    
    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_excel(os.path.join(output_dir, "task_sensitive_plv_results.xlsx"), index=False)
    
    logger.info(f"PLV analysis completed: {len(results_df)} trials processed")
    return results_df

def compare_microstate_vs_plv_task_sensitivity(microstate_df: pd.DataFrame, 
                                             plv_df: pd.DataFrame, 
                                             output_dir: str) -> pd.DataFrame:
    """
    Compare whether microstate and PLV measures show similar task sensitivity.
    Focus: Do both methods detect correct vs incorrect trial differences?
    """
    
    logger.info("Comparing microstate vs PLV task sensitivity...")
    
    # Merge datasets
    merged_df = pd.merge(
        microstate_df, plv_df, 
        on=['dyad', 'trial', 'modality', 'result'], 
        how='inner'
    )
    
    logger.info(f"Merged dataset: {len(merged_df)} trials")


    # Key measures to compare
    microstate_measures = [
        'entropy_rate_sender', 'entropy_rate_receiver', 
        'inter_entropy_rate', 'shared_ratio_time'
    ]
    
    plv_measures = [
        'plv_mean_alpha', 'plv_mean_beta', 'plv_mean_gamma'
    ]
    
    comparison_results = []
    
    for measure in microstate_measures + plv_measures:
        if measure not in merged_df.columns:
            logger.warning(f"Measure {measure} not found in data")
            continue
            
        # Create model data
        model_data = merged_df[[measure, 'result', 'trial_duration_x', 'dyad', 'modality']].dropna()
        
        if len(model_data) < 20:  # Minimum sample size
            logger.warning(f"Insufficient data for {measure}")
            continue
            
        # Convert result to numeric (correct = 1, incorrect = 0)
        model_data['result_numeric'] = (model_data['result'] == 'correct').astype(int)
    
        
        try:
            # CORRECTED: Mixed-effects model with proper controls
            formula = f'{measure} ~ result_numeric + trial_duration_x'
            model = smf.mixedlm(formula, model_data, groups=model_data['dyad'])
            fitted_model = model.fit(method='lbfgs')
            
            # Extract results
            result_coef = fitted_model.params['result_numeric']
            result_pval = fitted_model.pvalues['result_numeric']
            result_ci = fitted_model.conf_int().loc['result_numeric']
            
            # Effect size calculation (standardized coefficient)
            outcome_std = model_data[measure].std()
            effect_size = result_coef / outcome_std
            
            comparison_results.append({
                'measure': measure,
                'method': 'microstate' if measure in microstate_measures else 'plv',
                'coefficient': result_coef,
                'p_value': result_pval,
                'ci_lower': result_ci[0],
                'ci_upper': result_ci[1],
                'effect_size': effect_size,
                'n_trials': len(model_data),
                'n_dyads': model_data['dyad'].nunique()
            })
            
        except Exception as e:
            logger.error(f"Model fitting failed for {measure}: {e}")
            continue
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    if len(comparison_df) > 0:
        # CORRECTED: Apply multiple comparison correction
        p_values = comparison_df['p_value'].values
        rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        
        comparison_df['p_fdr'] = p_corrected
        comparison_df['significant'] = rejected
        
        # Summary analysis
        microstate_results = comparison_df[comparison_df['method'] == 'microstate']
        plv_results = comparison_df[comparison_df['method'] == 'plv']
        
        logger.info(f"CORRECTED RESULTS:")
        logger.info(f"Microstate significant: {microstate_results['significant'].sum()}")
        logger.info(f"PLV significant: {plv_results['significant'].sum()}")
        logger.info(f"Total significant: {comparison_df['significant'].sum()}")
        
        # Save results
        comparison_df.to_csv(os.path.join(output_dir, "corrected_method_comparison.csv"), index=False)
        
        # Generate summary
        summary = {
            'microstate_significant': int(microstate_results['significant'].sum()),
            'plv_significant': int(plv_results['significant'].sum()),
            'total_significant': int(comparison_df['significant'].sum()),
            'microstate_features_tested': len(microstate_results),
            'plv_features_tested': len(plv_results)
        }
        
        pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "task_sensitivity_summary.csv"), index=False)
        
    
    return comparison_df

def analyze_left_temporoparietal_gamma(input_dir):
    """
    Test gamma-band PLV in left temporoparietal regions for successful alignment
    """
    
    # Define left temporoparietal electrodes (standard 10-20 system)
    left_temporoparietal = [
        'T7',     # Left temporal 
        'TP9',    # Left temporoparietal (if available)
        'P7',     # Left parietal
        'P3',     # Left central parietal
        'CP5',    # Left centroparietal
        'CP1',    # Left centroparietal
    ]
    
    # Check which electrodes you actually have
    # Let's first see your electrode names from a sample file
    sample_files = [f for f in os.listdir(input_dir) if f.endswith('.fif')][:1]
    if sample_files:
        import mne
        raw = mne.io.read_raw_fif(os.path.join(input_dir, sample_files[0]), verbose=False)
        available_electrodes = raw.ch_names
        print(f"Available electrodes: {available_electrodes}")
        
        # Find left temporoparietal electrodes in your data
        left_tp_available = [ch for ch in left_temporoparietal if ch in available_electrodes]
        print(f"Left temporoparietal electrodes available: {left_tp_available}")
        
        if not left_tp_available:
            # Alternative: use closest available electrodes
            # Common alternatives in 32-channel systems:
            alternatives = ['T7', 'T8', 'P7', 'P8', 'P3', 'P4', 'CP1', 'CP2', 'CP5', 'CP6']
            left_tp_available = [ch for ch in alternatives if ch in available_electrodes and 
                               (ch.endswith('7') or ch.endswith('3') or ch.endswith('1') or ch.endswith('5'))]
            print(f"Using alternative left-side electrodes: {left_tp_available}")
    
    return left_tp_available
s

def compare_ibs_with_microstates(microstate_df: pd.DataFrame, ibs_df: pd.DataFrame, output_dir: str):





    
    """
    Compare IBS results with microstate results
    Both DataFrames should have same structure: dyad, trial, modality, result
    """
    
    # Merge on the same keys you use for microstate analysis
    merged_df = pd.merge(
        microstate_df, 
        ibs_df, 
        on=['dyad', 'trial', 'modality', 'result'], 
        how='inner'
    )
    
    logger.info(f"Merged {len(merged_df)} trials for comparison")
    
    # Compare correct vs incorrect for both approaches
    microstate_features = [
        'entropy_rate_sender', 'excess_entropy_sender', 
        'entropy_rate_receiver', 'excess_entropy_receiver',
        'inter_entropy_rate', 'inter_excess_entropy'
    ]
    
    ibs_features = [
        'plv_mean_alpha', 'ccorr_mean_alpha', 'coh_mean_alpha',
        'plv_mean_broad', 'ccorr_mean_broad', 'coh_mean_broad'
    ]
    
    # Run statistical tests on both types of measures
    comparison_results = {}
    
    for feature in microstate_features + ibs_features:
        if feature in merged_df.columns:
            correct_vals = merged_df[merged_df['result'] == 'correct'][feature].dropna()
            incorrect_vals = merged_df[merged_df['result'] == 'incorrect'][feature].dropna()
            
            if len(correct_vals) > 5 and len(incorrect_vals) > 5:
                stat, p_val = mannwhitneyu(correct_vals, incorrect_vals, alternative='two-sided')
                effect_size = (correct_vals.mean() - incorrect_vals.mean()) / np.sqrt((correct_vals.var() + incorrect_vals.var()) / 2)
                
                comparison_results[feature] = {
                    'method': 'microstate' if feature in microstate_features else 'ibs',
                    'p_value': p_val,
                    'effect_size': effect_size,
                    'correct_mean': correct_vals.mean(),
                    'incorrect_mean': incorrect_vals.mean()
                }
    
    # Save comparison
    comparison_df = pd.DataFrame(comparison_results).T
    comparison_df.to_excel(os.path.join(output_dir, "microstate_vs_ibs_comparison.xlsx"), index=True)
    
    return merged_df, comparison_results

def test_gamma_alignment_hypothesis(results_df):
    """
    Test: successful conceptual alignment → increased gamma PLV in left temporoparietal
    """
    
    print("=" * 60)
    print("TESTING: Gamma PLV in Left Temporoparietal → Successful Alignment")
    print("=" * 60)
    
    # Basic statistics
    correct_trials = results_df[results_df['result'] == 'correct']
    incorrect_trials = results_df[results_df['result'] == 'incorrect']
    
    print(f"Correct trials: {len(correct_trials)}")
    print(f"Incorrect trials: {len(incorrect_trials)}")
    
    # Compare gamma PLV between correct and incorrect trials
    correct_plv = correct_trials['left_tp_gamma_plv_mean']
    incorrect_plv = incorrect_trials['left_tp_gamma_plv_mean']
    
    print(f"\nLeft Temporoparietal Gamma PLV (28-40 Hz):")
    print(f"  Correct trials: {correct_plv.mean():.4f} ± {correct_plv.std():.4f}")
    print(f"  Incorrect trials: {incorrect_plv.mean():.4f} ± {incorrect_plv.std():.4f}")
    print(f"  Difference: {correct_plv.mean() - incorrect_plv.mean():.4f}")
    
    # Statistical tests
    from scipy import stats
    
    # Basic t-test
    tstat, pval_basic = stats.ttest_ind(correct_plv, incorrect_plv)
    print(f"\nBasic t-test:")
    print(f"  t-statistic: {tstat:.3f}")
    print(f"  p-value: {pval_basic:.4f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((correct_plv.var() + incorrect_plv.var()) / 2)
    cohens_d = (correct_plv.mean() - incorrect_plv.mean()) / pooled_std
    print(f"  Cohen's d: {cohens_d:.3f}")
    
    # Control for duration (major confound from your earlier analysis)
    correct_dur = correct_trials['trial_duration']
    incorrect_dur = incorrect_trials['trial_duration']
    
    print(f"\nDuration check:")
    print(f"  Correct duration: {correct_dur.mean():.2f}±{correct_dur.std():.2f}s")
    print(f"  Incorrect duration: {incorrect_dur.mean():.2f}±{incorrect_dur.std():.2f}s")
    
    dur_ttest = stats.ttest_ind(correct_dur, incorrect_dur)
    print(f"  Duration difference p-value: {dur_ttest.pvalue:.4f}")
    
    # Mixed-effects model controlling for duration and dyad
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import mixedlm
        
        model_df = results_df.copy()
        model_df['result_numeric'] = (model_df['result'] == 'correct').astype(int)
        
        formula = 'left_tp_gamma_plv_mean ~ result_numeric + trial_duration'
        model = mixedlm(formula, model_df, groups=model_df['dyad'])
        fitted_model = model.fit()
        
        result_coef = fitted_model.params['result_numeric']
        result_pval = fitted_model.pvalues['result_numeric']
        
        print(f"\nMixed-effects model (controlling for duration and dyad):")
        print(f"  Correct > Incorrect effect: {result_coef:.4f}")
        print(f"  p-value: {result_pval:.4f}")
        print(f"  Significant: {'YES' if result_pval < 0.05 else 'NO'}")
        
    except Exception as e:
        print(f"Mixed-effects analysis failed: {e}")
    
    # Test by modality (verbal vs gesture)
    print(f"\nBy communication modality:")
    for modality in results_df['modality'].unique():
        mod_data = results_df[results_df['modality'] == modality]
        mod_correct = mod_data[mod_data['result'] == 'correct']['left_tp_gamma_plv_mean']
        mod_incorrect = mod_data[mod_data['result'] == 'incorrect']['left_tp_gamma_plv_mean']
        
        if len(mod_correct) > 5 and len(mod_incorrect) > 5:
            mod_tstat, mod_pval = stats.ttest_ind(mod_correct, mod_incorrect)
            print(f"  {modality.capitalize()}:")
            print(f"    Correct: {mod_correct.mean():.4f}, Incorrect: {mod_incorrect.mean():.4f}")
            print(f"    p-value: {mod_pval:.4f}")
    
    return {
        'basic_p_value': pval_basic,
        'cohens_d': cohens_d,
        'correct_mean': correct_plv.mean(),
        'incorrect_mean': incorrect_plv.mean()
    }
def compute_left_tp_gamma_plv(input_dir, output_dir, left_tp_electrodes):
    """
    Recompute gamma PLV specifically for left temporoparietal electrode pairs
    """
    
    file_groups = _group_trials(input_dir)  # Your existing function
    results = []
    
    # Focus only on gamma band
    gamma_band = [28, 40]  # Matching the study you referenced
    
    for (dyad, trial_num), files in file_groups.items():
        
        if len(files) != 2:
            continue
            
        # Load files
        try:
            sender_raw = mne.io.read_raw_fif(os.path.join(input_dir, files[0]['filename']), 
                                           preload=True, verbose=False)
            receiver_raw = mne.io.read_raw_fif(os.path.join(input_dir, files[1]['filename']), 
                                             preload=True, verbose=False)
            
            # Get trial info
            trial_duration = min(sender_raw.times[-1], receiver_raw.times[-1]) - 0.1
            if trial_duration < 2.0:
                continue
                
            # Crop to safe duration
            sender_cropped = sender_raw.copy().crop(tmin=0, tmax=trial_duration)
            receiver_cropped = receiver_raw.copy().crop(tmin=0, tmax=trial_duration)
            
            # Filter to gamma band (28-40 Hz)
            sender_gamma = sender_cropped.copy().filter(
                l_freq=gamma_band[0], h_freq=gamma_band[1],
                l_trans_bandwidth=3.0, h_trans_bandwidth=3.0,
                verbose=False
            )
            receiver_gamma = receiver_cropped.copy().filter(
                l_freq=gamma_band[0], h_freq=gamma_band[1],
                l_trans_bandwidth=3.0, h_trans_bandwidth=3.0,
                verbose=False
            )
            
            # Get data
            sender_data = sender_gamma.get_data()
            receiver_data = receiver_gamma.get_data()
            
            # Find indices of left temporoparietal electrodes
            left_tp_indices = [i for i, ch in enumerate(sender_gamma.ch_names) 
                              if ch in left_tp_electrodes]
            
            if len(left_tp_indices) < 2:
                continue
                
            # Compute PLV specifically for left temporoparietal electrode pairs
            from scipy.signal import hilbert
            plv_values = []
            
            for idx_s in left_tp_indices:
                for idx_r in left_tp_indices:
                    
                    # Apply Hilbert transform
                    sender_analytic = hilbert(sender_data[idx_s, :])
                    receiver_analytic = hilbert(receiver_data[idx_r, :])
                    
                    # Extract phases
                    sender_phase = np.angle(sender_analytic)
                    receiver_phase = np.angle(receiver_analytic)
                    
                    # Phase difference and PLV
                    phase_diff = sender_phase - receiver_phase
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    plv_values.append(plv)
            
            # Store results
            result = {
                'dyad': dyad,
                'trial': trial_num,
                'result': files[0]['result'],  # correct/incorrect
                'modality': files[0]['modality'],  # verbal/gesture
                'trial_duration': trial_duration,
                'left_tp_gamma_plv_mean': np.mean(plv_values),
                'left_tp_gamma_plv_std': np.std(plv_values),
                'left_tp_gamma_plv_max': np.max(plv_values),
                'n_electrode_pairs': len(plv_values),
                'electrodes_used': left_tp_electrodes
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing dyad {dyad} trial {trial_num}: {e}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(output_dir, "left_tp_gamma_plv_results.xlsx"), index=False)
    
    return results_df

