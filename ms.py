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
from scipy.stats import chi2
from eeg_microstates3 import *
import mne
import pandas as pd
from pandas.api.types import CategoricalDtype
import pycrostates
from pycrostates.preprocessing import extract_gfp_peaks, apply_spatial_filter
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.segmentation import *
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from scipy.stats import entropy
import warnings

import eeg_microstates3
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
import time
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',  
    handlers=[
        logging.FileHandler('ms_extraction.log'),
        logging.StreamHandler()
        ]
)
logger = logging.getLogger(__name__)
mne.set_log_level('ERROR')



def ms_load_and_filter(file_path: str,  band: str = "broad") -> mne.io.Raw:
    """Load preprocessed EEG data and apply microstate-specific filtering.
    
    Args:
        file_path: Path to the FIF file containing preprocessed EEG data
        
    Returns:
        mne.io.Raw: Filtered raw data with spatial filter applied
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the loaded data has unexpected shape
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading {file_path}...")
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    raw.resample(128, verbose='error')
    if band == "alpha":
        raw.filter(8, 13, method='iir', iir_params=dict(order=8, ftype="butter"))
    elif band == "gamma":
        raw.filter(28, 40, method='iir', iir_params=dict(order=8, ftype="butter"))
    else:  # "broad"
        raw.filter(1, 40, method='iir', iir_params=dict(order=8, ftype="butter"))
    apply_spatial_filter(raw, n_jobs=-1)
        
    return raw


def run_microstate_analysis(input_dir: str,
                            output_dir: str,
                            cluster_range: List[int] = [4, 5, 6]) -> None:
    """Individual then group clustering with GEV to evaluate best fit."""
    
    participant_files = defaultdict(list)
    
    all_files = os.listdir(input_dir)
    logger.info(f"Found {len(all_files)} files in {input_dir}")
    
    for f in all_files:
          if not f.endswith('.fif'):
              continue
          match = re.search(r'([0-9]{2}[AB])', f)
          if match:
              subject_id = match.group(1)
              participant_files[subject_id].append(os.path.join(input_dir, f))
          else:
              logger.warning(f"Filename skipped (no subject match): {f}")
    
    logger.info(f"Added files to processing queue for {len(participant_files)} subjects")

    # === INDIVIDUAL LEVEL CLUSTERING ===

    individual_maps = {k: [] for k in cluster_range}
    gev_scores = []

    for subject_id, files in participant_files.items():
        logger.info(f"Processing subject {subject_id} with {len(files)} trials...")
        try:
            raws = [ms_load_and_filter(f) for f in files]
            raw = mne.concatenate_raws(raws)
            gfp_peaks = extract_gfp_peaks(raw)

            for k in cluster_range:
                logger.info(f"  Clustering K={k}")
                modk = ModKMeans(n_clusters=k, random_state=42)
                modk.fit(gfp_peaks)
                
                score = modk.GEV_
                gev_scores.append([subject_id, k, score])

                # Save individual maps
                out_path = os.path.join(output_dir, f"maps_subj_{subject_id}_k_{k}.npy")
                np.save(out_path, modk.cluster_centers_)
                
                # Add to individual maps list
                if len(modk.cluster_centers_) > 0:
                    individual_maps[k].append(modk.cluster_centers_)
                else:
                    logger.warning(f"No maps found for subject {subject_id} with K={k}")
                    individual_maps[k].append(np.zeros((k, gfp_peaks.shape[1])))  # Add empty array if needed
                    
        except Exception as e:
            logger.error(f"Failed subject {subject_id}: {str(e)}")
            continue

    for k in cluster_range:
        if len(individual_maps[k]) == 0:
            logger.warning(f"No valid individual maps for K={k}. Skipping group clustering.")
            continue


    gev_df = pd.DataFrame(gev_scores, columns=["subject", "k", "gev"])
    gev_df.to_csv(os.path.join(output_dir, "gev_scores.csv"), index=False)

    # === GROUP LEVEL CLUSTERING ===

    gev_scores_group = []

    for k in cluster_range:
        if len(individual_maps[k]) == 0:
            continue 
        
        logger.info(f"\nGroup-level clustering for K={k}")
        group_data = np.vstack(individual_maps[k])
        group_data = ChData(group_data.T, gfp_peaks.info)

        modk = ModKMeans(n_clusters=k, random_state=42)
        modk.fit(group_data)

        score = modk.GEV_
        gev_scores_group.append([k, score])

        np.save(os.path.join(output_dir, f"maps_group_k_{k}.npy"), modk.cluster_centers_)
        modk.save(os.path.join(output_dir, f"clustering_group_k_{k}.fif"))

        fig = modk.plot()
        fig.savefig(os.path.join(output_dir, f"group_k{k}_templates.png"))
        plt.close(fig)

    gevgroupe_df = pd.DataFrame(gev_scores_group, columns=["k", "gev"])
    gevgroupe_df.to_csv(os.path.join(output_dir, "gev_scores_group.csv"), index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(gevgroupe_df["k"], gevgroupe_df["gev"], marker='o', linestyle='-')
    plt.title("Group-level GEV vs Number of Microstate Clusters")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Global Explained Variance (GEV)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_gev_plot.png"))
    plt.show()




def extract_parameters(
    input_dir: str,
    output_dir: str,
    ModK: ModKMeans,
) -> pd.DataFrame:
    """Extract intrabrain and interbrain parameters.
    
    Args:
        input_dir: segmented EEG files
        output_dir: save results
        ModK: pre-fitted ModKMeans model
        
    Returns:

            parameters_df: DataFrame with all parameters merged together
            
    """
    # Check ModK 
    if not hasattr(ModK, 'cluster_centers_'):
        raise ValueError("ModK model must be fitted before use")
    
    # Initialize variables
    labels = ModK.cluster_names
    skipped = []
    parameters = []
    sender_states = ['A', 'B', 'C', 'D']
    receiver_states = ['A', 'B', 'C', 'D']
    combined_states = [f"{s}_{r}" for s, r in itertools.product(sender_states, receiver_states)]
    combined_state_to_int = {state: idx for idx, state in enumerate(combined_states)}
    n_maps = len(combined_state_to_int)
    print(n_maps)
    mapping_path = os.path.join(output_dir, "combined_state_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(combined_state_to_int, f, indent=4)


    # Import files, grouped per dyad and trials 
    file_groups = _group_trials(input_dir)
    for (dyad, trial_num), files in file_groups.items():
        
        if len(files) != 2:
            _skip_trial(skipped, dyad, trial_num, "not exactly 2 files")
            continue
        start = time.time()
        # Identify roles
        sender = next((f for f in files if (f["participant"] == "A" and f["direction"] == "AtoB") or 
                                         (f["participant"] == "B" and f["direction"] == "BtoA")), None)
        receiver = next((f for f in files if f != sender), None)
        if not sender or not receiver:
            _skip_trial(skipped, dyad, trial_num, "missing sender or receiver")
            continue

        # Load file with ms preprocessing (and ensure same duration)
        sender_file = ms_load_and_filter(os.path.join(input_dir, sender['filename']))
        receiver_file = ms_load_and_filter(os.path.join(input_dir, receiver['filename']))
        # print(sender_file.times[-1],receiver_file.times[-1])
        # print(sender_file._data.shape,receiver_file._data.shape)
        # print(sender_file.info['sfreq'], receiver_file.info['sfreq'])
        if not (np.isclose(sender_file.times[0], receiver_file.times[0]) and
        np.isclose(sender_file.times[-1], receiver_file.times[-1])):
            _skip_trial(skipped, dyad, trial_num, "mismatched durations")
            continue

       #  Segmentation
        seg_sender =  ModK.predict(sender_file, factor=10, half_window_size=1, min_segment_length=1, reject_edges=False)
        seg_receiver = ModK.predict(receiver_file, factor=10, half_window_size=1, min_segment_length=1, reject_edges=False)
        #  timestamps based on factor/sfreq 
        num_segments_s = len(seg_sender.labels)
        num_segments_r = len(seg_receiver.labels)
        sample_duration = 10 / 128  # seconds per segment
        time_sender = np.arange(num_segments_s) * sample_duration
        time_receiver = np.arange(num_segments_r) * sample_duration
        
        if not np.array_equal(time_sender, time_receiver):
            _skip_trial(skipped, dyad, trial_num, "segmented sequences not aligned")
            continue
        
        # Extract intrabrain (sender and receiver) parameters
        trial_duration = sender_file.times[-1] - sender_file.times[0]
        intra_row, p_hat_s, T_hat_s, p_hat_r, T_hat_r = compute_intrabrain_metrics(dyad, trial_num, seg_sender, seg_receiver, trial_duration, sender, labels, n_maps)

        # Create sender+receiver combined sequence (format: labelsender_labelreceiver, e.g.: A_B)
        combined_sequence = [] 
        combined_labels = [f"{labels[s]}_{labels[r]}" for s, r in zip(seg_sender.labels, seg_receiver.labels)]
        int_sequence = [combined_state_to_int[label] for label in combined_labels]
        for i, (t, s_label, r_label, comb_label) in enumerate(zip(time_sender, seg_sender.labels, seg_receiver.labels, combined_labels)):
            combined_sequence.append({
                "dyad": dyad,
                "trial": trial_num,
                "timestamp": t,
                "sender_label": labels[s_label],
                "receiver_label": labels[r_label],
                "combined_label": comb_label,
                "combined_int": combined_state_to_int[comb_label] 
            })

        # Calculate interbrain parameters
        inter_row, p_hat, T_hat = compute_interbrain_metrics(dyad, trial_num, combined_sequence, int_sequence, n_maps, trial_duration)
        merged_row = {**intra_row, **inter_row}
        parameters.append(merged_row)

        # Generate surrogate
        n_surrogates = 20
        results = Parallel(n_jobs=-1)(
            delayed(compute_surrogates)(
                seg_sender, seg_receiver, 
                int_sequence, time_sender,
                dyad, trial_num, trial_duration, sender, labels, n_maps,
                p_hat_s, T_hat_s, p_hat_r, T_hat_r, p_hat, T_hat,
                combined_sequence
            ) for _ in range(n_surrogates)  
)

        shuffled_surrogates, markov_surrogates = zip(*results)
        shuffled_surrogates = list(shuffled_surrogates)
        markov_surrogates = list(markov_surrogates)


        logger.info(f"Extracted parameters and surrogates for dyad {dyad}, trial {trial_num}")
        logger.info("Extraction done in %.2f seconds", time.time() - start)



    skipped_df = pd.DataFrame(skipped, columns=["dyad", "trial", "reason"])
    skipped_df.to_csv(os.path.join(output_dir, "skipped_trials.csv"), index=False)
    sequence_df = pd.DataFrame(combined_sequence)
    sequence_df.to_csv(os.path.join(output_dir, "combined_sequence.csv"), index=False)
    parameters_df = pd.DataFrame(parameters)
    parameters_df.to_excel(os.path.join(output_dir, "all_parameters.xlsx"), index=False)
    shuffled_surrogates_df = pd.DataFrame(shuffled_surrogates)
    markov_surrogates_df = pd.DataFrame(markov_surrogates)
    shuffled_surrogates_df.to_excel(os.path.join(output_dir, "shuffled_surrogates.xlsx"), index=False)
    markov_surrogates_df.to_excel(os.path.join(output_dir, "markov_surrogates.xlsx"), index=False)

    
    return parameters_df, shuffled_surrogates_df, markov_surrogates_df



def compute_interbrain_metrics(dyad, trial_num, combined_sequence, int_sequence, n_maps, trial_duration):
    
    timestamps = [row['timestamp'] for row in combined_sequence]
    sender_states = [row['sender_label'] for row in combined_sequence]
    receiver_states = [row['receiver_label'] for row in combined_sequence]
    combined_states = [row['combined_label'] for row in combined_sequence]
    unique_combined_states = sorted(list(set(combined_states)))  # ['A_A', 'A_B', 'A_C', ...]
    
    # double checking aligned lenghts. 
    min_len = min(len(sender_states), len(receiver_states), len(timestamps))
    sender_states = sender_states[:min_len]
    receiver_states = receiver_states[:min_len]
    combined_states = combined_states[:min_len]
    timestamps = timestamps[:min_len]


    timestamps = np.array(timestamps)
    if not np.all(np.diff(timestamps) >= 0):
        timestamps = np.sort(timestamps)

    row = {'dyad': dyad, 'trial': trial_num}


    #  === SHARED STATE METRICS ===

    # identify same states
    same_state = np.array(sender_states) == np.array(receiver_states)
    
    durations = np.diff(timestamps)
    if len(durations) != len(sender_states) - 1:
        raise ValueError("Mismatch between durations and state lengths")

    total_time = trial_duration
    
    # Time coverage (ratio of time they shared a state)
    shared_time = np.sum(durations[same_state[:-1]]) 
    shared_ratio_time = shared_time / total_time if total_time > 0 else 0
    
    # Shared occurences, count how often they start synchronizing (0â†’1 transitions)(normalised)
    shared_occurrences = np.sum(np.diff(same_state.astype(int)) == 1) 
    
    # Mean duration of shared states (average length of shared episodes)
    shared_durations = []
    for k, g in groupby(zip(same_state[:-1], durations), key=lambda x: x[0]):
        if k:  # only shared states
            group = list(g)
            shared_durations.append(np.sum([d for _, d in group]))
    mean_shared_duration = np.mean(shared_durations) if shared_durations else 0

   
    row.update({
        "shared_ratio_time": shared_ratio_time, 
        "shared_mean_duration": mean_shared_duration,
        "shared_occurrences_per_sec": shared_occurrences / total_time if total_time > 0 else 0
    })

     # === INFORMATION METRICS  ===

    # Distribution and Transition matrix
    p_hat = p_empirical(int_sequence, n_maps)
    for i in range(n_maps):
        row[f"p_{i}"] = p_hat[i]

    T_hat = T_empirical(int_sequence, n_maps)
    for i, from_state in enumerate(unique_combined_states):
        for j, to_state in enumerate(unique_combined_states):
            if i < T_hat.shape[0] and j < T_hat.shape[1]:  # Safety check
                row[f"T_{from_state}_to_{to_state}"] = T_hat[i,j]



    if len(int_sequence) > 50:

        # Shannon entropy
        h = H_1(int_sequence, n_maps)
        h_max = max_entropy(n_maps)
        row['inter_shannon_entropy'] = h
        row['inter_shannon_entropy_max'] = h_max

        # Excess entropy 
        kmax = 6
        h_rate, excess_entropy = eeg_microstates3.excess_entropy_rate(int_sequence, n_maps, kmax)
        h_mc = mc_entropy_rate(p_hat, T_hat)
        row['inter_entropy_rate'] = h_rate          # slope a
        row['inter_excess_entropy'] = excess_entropy  # intercept b
        row['inter_mc_entropy_rate'] = h_mc

        row.update({
            # Relative entropy measures
            "inter_relative_entropy": compute_relative_entropy(int_sequence, n_maps),
            "inter_adjusted_excess_entropy": compute_length_adjusted_excess_entropy(int_sequence, n_maps),
            "inter_entropy_rate_per_sec": compute_entropy_rate_per_second(int_sequence, n_maps, trial_duration),
            
            # Transition-based normalized measures
            "inter_transitions_per_sec": len(int_sequence) / trial_duration if trial_duration > 0 else np.nan,
            "inter_unique_states_ratio": len(np.unique(int_sequence)) / n_maps,  # How many of 16 possible states used
            
            # Temporal efficiency measures
            "inter_entropy_efficiency": h / h_max if h_max > 0 else 0,  # How close to max entropy
            "inter_temporal_efficiency": excess_entropy / trial_duration if trial_duration > 0 else np.nan,
        })

        # Markov tests
        alpha = 0.01
        p0 = testMarkov0(int_sequence, n_maps, alpha)
        p1 = testMarkov1(int_sequence, n_maps, alpha=0.01, min_len=50, verbose=False)
        p2 = testMarkov2(int_sequence, n_maps, alpha=0.01, min_len=50, verbose=False)
        row['inter_p_Markov0'] = p0
        row['inter_p_Markov1'] = p1
        row['inter_p_Markov2'] = p2

        # AIF
        l_max = 100  
        aif_empirical = mutinf(int_sequence, n_maps, l_max)

        row['aif_max'] = np.max(aif_empirical)
        row['aif_mean'] = np.mean(aif_empirical)
        row['aif_peak_lag'] = np.argmax(aif_empirical)
        row['aif_auc'] = np.trapz(aif_empirical)
        row['aif_full'] = json.dumps(aif_empirical.tolist())

        row.update({
            "aif_max_per_sec": np.max(aif_empirical) / trial_duration if trial_duration > 0 else np.nan,
            "aif_mean_per_sec": np.mean(aif_empirical) / trial_duration if trial_duration > 0 else np.nan,
            "aif_auc_per_sec": np.trapz(aif_empirical) / trial_duration if trial_duration > 0 else np.nan,
        })

    return row, p_hat, T_hat

def compute_intrabrain_metrics(dyad, trial_num, seg_s, seg_r, duration, sender, labels, n_maps):
    row = {
        "dyad": dyad,
        "trial": trial_num,
        "modality": sender['modality'],
        "result": sender['result'],
        "trial_duration": duration,
    }

    if hasattr(seg_s, "entropy"):
        es, xs, *_ = pycrostates.segmentation.excess_entropy_rate(seg_s, history_length=6)
        er, xr, *_ = pycrostates.segmentation.excess_entropy_rate(seg_r, history_length=6)
        row.update({
            "entropy_sender": seg_s.entropy(ignore_repetitions=False),
            "entropy_receiver": seg_r.entropy(ignore_repetitions=False),
            "entropy_rate_sender": es / duration,
            "excess_entropy_sender": xs / duration,
            "entropy_rate_receiver": er / duration,
            "excess_entropy_receiver": xr / duration,
            "transitions_sender_per_sec": len(seg_s.labels) / duration if duration > 0 else np.nan,
            "transitions_receiver_per_sec": len(seg_r.labels) / duration if duration > 0 else np.nan,
            # Relative entropy (0-1 scale)
            "relative_entropy_sender": compute_relative_entropy(seg_s.labels, len(labels)),
            "relative_entropy_receiver": compute_relative_entropy(seg_r.labels, len(labels))
        })

        seg_s_params = seg_s.compute_parameters()
        seg_r_params = seg_r.compute_parameters()
    

        for label in labels:
            row.update({
                f"timecov_sender_{label}": seg_s_params.get(f"{label}_timecov", np.nan),
                f"meandurs_sender_{label}": seg_s_params.get(f"{label}_meandurs", np.nan),
                f"occurrences_sender_{label}": seg_s_params.get(f"{label}_occurrences", np.nan), 

                f"timecov_receiver_{label}": seg_r_params.get(f"{label}_timecov", np.nan),
                f"meandurs_receiver_{label}": seg_r_params.get(f"{label}_meandurs", np.nan),
                f"occurrences_receiver_{label}": seg_r_params.get(f"{label}_occurrences", np.nan), 

            })

        p_hat_s = p_empirical(seg_s.labels, len(labels))
        T_hat_s = T_empirical(seg_s.labels, len(labels))
        p_hat_r = p_empirical(seg_r.labels, len(labels))
        T_hat_r = T_empirical(seg_r.labels, len(labels))

    else:  # Surrogate sequence (numpy array)
        h_s = H_1(seg_s, n_maps)
        h_r = H_1(seg_r, n_maps)
        kmax = 6
        h_rate_s, excess_entropy_s = eeg_microstates3.excess_entropy_rate(seg_s, n_maps, kmax)
        h_rate_r, excess_entropy_r = eeg_microstates3.excess_entropy_rate(seg_r, n_maps, kmax)

        # MATCH THE NORMALIZATION FROM THE REAL DATA BRANCH
        row.update({
            'entropy_sender': h_s,  # Shannon entropy (same name as real data)
            'entropy_receiver': h_r,
            'entropy_rate_sender': h_rate_s / duration,  # NORMALIZE like real data
            'excess_entropy_sender': excess_entropy_s / duration,  # NORMALIZE like real data
            'entropy_rate_receiver': h_rate_r / duration,  # NORMALIZE like real data
            'excess_entropy_receiver': excess_entropy_r / duration,  # NORMALIZE like real data
            
            # Add the other normalized measures to match
            'transitions_sender_per_sec': len(seg_s) / duration if duration > 0 else np.nan,
            'transitions_receiver_per_sec': len(seg_r) / duration if duration > 0 else np.nan,
            'relative_entropy_sender': compute_relative_entropy(seg_s, len(labels)),
            'relative_entropy_receiver': compute_relative_entropy(seg_r, len(labels)),
        })

        p_hat_s = p_empirical(seg_s, len(labels))
        T_hat_s = T_empirical(seg_s, len(labels))
        p_hat_r = p_empirical(seg_r, len(labels))
        T_hat_r = T_empirical(seg_r, len(labels))

    

    return row, p_hat_s, T_hat_s, p_hat_r, T_hat_r

def compute_relative_entropy(sequence, n_states=4):
    """Compute entropy relative to maximum possible for this sequence length"""
    
    if len(sequence) == 0:
        return np.nan
    
    # Actual Shannon entropy
    actual_entropy = H_1(sequence, n_states)
    
    # Maximum possible entropy for this sequence length
    # Can't have more entropy than log(unique_states) or log(sequence_length)
    max_entropy = np.log(min(n_states, len(np.unique(sequence))))
    
    # Relative entropy (0 = minimum randomness, 1 = maximum randomness)
    if max_entropy > 0:
        relative_entropy = actual_entropy / max_entropy
    else:
        relative_entropy = 0
    
    return relative_entropy

def compute_length_adjusted_excess_entropy(sequence, n_states=4, baseline_length=100):
    """Compute excess entropy adjusted for sequence length"""
    
    if len(sequence) < 10:  # Too short for reliable excess entropy
        return np.nan
    
    try:
        # Get raw excess entropy
        _, raw_excess = eeg_microstates3.excess_entropy_rate(sequence, n_states, kmax=6)
        
        # Adjust for sequence length (longer sequences can have higher excess entropy)
        length_factor = np.log(len(sequence)) / np.log(baseline_length)
        
        if length_factor > 0:
            adjusted_excess = raw_excess / length_factor
        else:
            adjusted_excess = raw_excess
            
        return adjusted_excess
        
    except Exception as e:
        print(f"Error computing adjusted excess entropy: {e}")
        return np.nan

def compute_entropy_rate_per_second(sequence, n_states=4, duration_seconds=1.0):
    """Compute entropy rate normalized by duration"""
    
    if len(sequence) < 10 or duration_seconds <= 0:
        return np.nan
    
    try:
        # Get raw entropy rate
        entropy_rate, _ = eeg_microstates3.excess_entropy_rate(sequence, n_states, kmax=6)
        
        # Normalize by duration
        entropy_rate_per_sec = entropy_rate / duration_seconds
        
        return entropy_rate_per_sec
        
    except Exception as e:
        print(f"Error computing entropy rate per second: {e}")
        return np.nan
    
    
def compute_surrogates(seg_sender, seg_receiver, int_sequence, time_sender, 
                      dyad, trial_num, trial_duration, sender, labels, n_maps, 
                      p_hat_s, T_hat_s, p_hat_r, T_hat_r, p_hat, T_hat, combined_sequence):
    """Compute shuffled and Markov surrogates for a trial."""
    
    # ===== SHUFFLED SURROGATES (0-order) =====
    
    # Shuffle sender and receiver sequences independently
    shuffled_sender = np.random.permutation(seg_sender.labels)
    shuffled_receiver = np.random.permutation(seg_receiver.labels)

    # Create shuffled combined sequence
    combined_labels_shuffled = [f"{labels[s]}_{labels[r]}" for s, r in zip(shuffled_sender, shuffled_receiver)]
    combined_sequence_shuffled = []
    for i, (t, s_label, r_label, comb_label) in enumerate(zip(time_sender, shuffled_sender, shuffled_receiver, combined_labels_shuffled)):
        combined_sequence_shuffled.append({
            "dyad": dyad,
            "trial": trial_num,
            "timestamp": t,
            "sender_label": labels[s_label],
            "receiver_label": labels[r_label],
            "combined_label": comb_label
        })
    
    # Shuffle the inter-brain sequence
    shuffled_int_sequence = int_sequence.copy()
    np.random.shuffle(shuffled_int_sequence)

    # Compute shuffled intrabrain metrics
    shuffle_intrarow, *_ = compute_intrabrain_metrics(
        dyad, trial_num, shuffled_sender, shuffled_receiver, trial_duration, sender, labels, n_maps
    )
    
    # Compute shuffled interbrain metrics
    shuffle_interrow, *_ = compute_interbrain_metrics(
        dyad, trial_num, combined_sequence_shuffled, shuffled_int_sequence, n_maps, trial_duration
    )
    
    # Combine shuffled results
    shuffle_row = {**shuffle_intrarow, **shuffle_interrow}

    # ===== MARKOV SURROGATES (1st-order) =====
    
    # Generate Markov surrogates preserving transition structure, individual and interbrain
    surrogate_s = surrogate_mc(p_hat_s, T_hat_s, n_maps, len(seg_sender.labels))
    surrogate_r = surrogate_mc(p_hat_r, T_hat_r, n_maps, len(seg_receiver.labels))
    
    surrogate_combined_sequence = []
    surrogate_combined_labels = [f"{labels[s]}_{labels[r]}" for s, r in zip(surrogate_s, surrogate_r)]
    combined_state_to_int = {f"{s}_{r}": i for i, (s, r) in enumerate(itertools.product(labels, labels))}
    surrogate_int_sequence = [combined_state_to_int[label] for label in surrogate_combined_labels]

    for i, (t, s_label, r_label, comb_label) in enumerate(zip(time_sender, surrogate_s, surrogate_r, surrogate_combined_labels)):
        surrogate_combined_sequence.append({
            "dyad": dyad,
            "trial": trial_num,
            "timestamp": t,
            "sender_label": labels[s_label],
            "receiver_label": labels[r_label],
            "combined_label": comb_label
        })

    # Compute Markov intrabrain metrics
    surrogate_intrarow, *_ = compute_intrabrain_metrics(
        dyad, trial_num, surrogate_s, surrogate_r, trial_duration, sender, labels, len(labels)
    )

    # Compute Markov interbrain metrics 
    surrogate_interrow, *_ = compute_interbrain_metrics(
        dyad, trial_num, surrogate_combined_sequence, surrogate_int_sequence, n_maps, trial_duration  
    )
   
    surrogate_row = {**surrogate_intrarow, **surrogate_interrow}


    return shuffle_row, surrogate_row

    # Compute Markov intrabrain metrics
    surrogate_intrarow, *_ = compute_intrabrain_metrics(
        dyad, trial_num, surrogate_s, surrogate_r, trial_duration, sender, labels, n_maps
    )

    # Compute Markov interbrain metrics  
    surrogate_interrow, *_ = compute_interbrain_metrics(
        dyad, trial_num, combined_sequence, surrogate_seq, n_maps, trial_duration
    )

    # Combine Markov results
    surrogate_row = {**surrogate_intrarow, **surrogate_interrow}

    # Return both surrogate types
    return shuffle_row, surrogate_row


# Also make sure you have the surrogate_mc function defined. If it's missing, add this:

# Replace the surrogate_mc function with this corrected version:

def surrogate_mc(p_hat, T_hat, n_maps, sequence_length):
    """Generate Markov surrogate sequence preserving transition structure."""
    
    if sequence_length <= 0:
        return np.array([])
    
    surrogate = np.zeros(sequence_length, dtype=int)
    
    # Determine actual number of states from probability vector
    actual_n_states = len(p_hat)
    
    # Sample initial state from stationary distribution
    if len(p_hat) > 0 and np.sum(p_hat) > 0:
        p_normalized = p_hat / np.sum(p_hat)
        # Use actual_n_states instead of n_maps for the choice
        surrogate[0] = np.random.choice(actual_n_states, p=p_normalized)
    else:
        surrogate[0] = np.random.choice(actual_n_states)
    
    # Generate remaining states using transition matrix
    for i in range(1, sequence_length):
        prev_state = surrogate[i-1]
        
        if prev_state < T_hat.shape[0]:
            transition_probs = T_hat[prev_state, :]
            
            # Handle zero probability transitions
            if np.sum(transition_probs) == 0:
                surrogate[i] = np.random.choice(actual_n_states)
            else:
                # Normalize probabilities
                transition_probs = transition_probs / np.sum(transition_probs)
                surrogate[i] = np.random.choice(actual_n_states, p=transition_probs)
        else:
            # Fallback for invalid state
            surrogate[i] = np.random.choice(actual_n_states)
    
    return surrogate




# ----- helper functions ----- 

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


def _skip_trial(trials, dyad, trial_num, reason):
    msg = f"Skipping dyad {dyad} trial {trial_num} â€” reason: {reason}"
    logger.warning(msg)
    trials.append({"dyad": dyad, "trial": trial_num, "reason": reason})

def testMarkov1(X, ns, alpha=0.01, min_len=50, verbose=False):
    """
    Test for first-order Markovianity:
    H0: p(X[t+1] | X[t]) = p(X[t+1] | X[t], X[t-1])
    adapted from von Wegner et al., 2018
    """
    if len(X) < min_len:
        if verbose: print("Sequence too short for Markov1 test.")
        return 1.0

    f_ijk = np.ones((ns, ns, ns)) * 1e-6
    f_ij = np.ones((ns, ns)) * 1e-6
    f_jk = np.ones((ns, ns)) * 1e-6
    f_j = np.ones(ns) * 1e-6

    for t in range(len(X) - 2):
        i, j, k = X[t], X[t+1], X[t+2]
        f_ijk[i, j, k] += 1
        f_ij[i, j] += 1
        f_jk[j, k] += 1
        f_j[j] += 1

    T = 0.0
    for i in range(ns):
        for j in range(ns):
            for k in range(ns):
                num = f_ijk[i, j, k] * f_j[j]
                den = f_ij[i, j] * f_jk[j, k]
                T += f_ijk[i, j, k] * np.log(num / den)

    T *= 2.0
    df = ns * (ns - 1) * (ns - 1)
    p = chi2.sf(T, df)

    if verbose:
        print(f"Markov1: T = {T:.3f}, df = {df}, p = {p:.4f}")

    return p

def testMarkov2(X, ns, alpha=0.01, min_len=50, verbose=False):
    """
    Test for second-order Markovianity:
    H0: p(X[t+1] | X[t], X[t-1]) = p(X[t+1] | X[t], X[t-1], X[t-2])
    """
    if len(X) < min_len:
        if verbose: print("Sequence too short for Markov2 test.")
        return 1.0

    f_ijkl = np.ones((ns, ns, ns, ns)) * 1e-6
    f_ijk = np.ones((ns, ns, ns)) * 1e-6
    f_jkl = np.ones((ns, ns, ns)) * 1e-6
    f_jk = np.ones((ns, ns)) * 1e-6

    for t in range(len(X) - 3):
        i, j, k, l = X[t], X[t+1], X[t+2], X[t+3]
        f_ijkl[i, j, k, l] += 1
        f_ijk[i, j, k] += 1
        f_jkl[j, k, l] += 1
        f_jk[j, k] += 1

    T = 0.0
    for i in range(ns):
        for j in range(ns):
            for k in range(ns):
                for l in range(ns):
                    num = f_ijkl[i, j, k, l] * f_jk[j, k]
                    den = f_ijk[i, j, k] * f_jkl[j, k, l]
                    T += f_ijkl[i, j, k, l] * np.log(num / den)

    T *= 2.0
    df = ns * ns * (ns - 1) * (ns - 1)
    p = chi2.sf(T, df)

    if verbose:
        print(f"Markov2: T = {T:.3f}, df = {df}, p = {p:.4f}")

    return p