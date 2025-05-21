import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
from typing import List, Dict

import mne
import pandas as pd

from pycrostates.preprocessing import extract_gfp_peaks, apply_spatial_filter
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.segmentation import excess_entropy_rate
from scipy.stats import entropy
from scipy.signal import correlate
import nolds 
from pycrostates.io import ChData


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
mne.set_log_level('ERROR')

def ms_load_and_filter(file_path):
    """Load preprocessed EEG data and apply ms-specific filtering."""
    print(f"Loading {file_path}...")
    raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    raw.resample(128, verbose='error')
    raw.filter(1, 40, method='iir', iir_params=dict(order=8, ftype="butter"))
    apply_spatial_filter(raw, n_jobs=-1)
    return raw


def run_microstate_analysis(input_dir: str,
                            output_dir: str,
                            cluster_range: List[int] = [3, 4, 5, 6, 7],
                            subjects: List[str] = None) -> None:
    """Individual then group clustering with GEV to evaluate best fit."""
    
    participant_files = defaultdict(list)
    
    for f in os.listdir(input_dir):
            if f.endswith('.fif'):
                subject_id = f.split('_')[0]
                if subjects and subject_id not in subjects:
                    continue
                participant_files[subject_id].append(os.path.join(input_dir, f))


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

                out_path = os.path.join(output_dir, f"maps_subj_{subject_id}_k_{k}.npy")
                np.save(out_path, modk.cluster_centers_)
                individual_maps[k].append(modk.cluster_centers_)

        except Exception as e:
            logger.error(f"Failed subject {subject_id}: {str(e)}")
            continue

    gev_df = pd.DataFrame(gev_scores, columns=["subject", "k", "gev"])
    gev_df.to_csv(os.path.join(output_dir, "gev_scores.csv"), index=False)

    # === GROUP LEVEL CLUSTERING ===

    gev_scores_group = []

    for k in cluster_range:
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

def group_trials(input_dir):
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

        file_groups.setdefault((dyad, trial_num), []).append({
            "filename": file,
            "participant": participant,
            "direction": direction,
            "modality": modality,
            "result": result
        })
    
    return file_groups

def extract_sequences_and_intrabrain_parameters(input_dir, output_dir, ModK):
    ms_data = []
    sequence_rows = []
    file_groups = group_trials(input_dir)

    for (dyad, trial_num), files in file_groups.items():
        if len(files) != 2:
            print(f"⚠️ Skipping trial {trial_num} for dyad {dyad}: not exactly 2 files.")
            continue

        # Identify sender and receiver
        sender_info = next((f for f in files if (f["participant"] == "A" and f["direction"] == "AtoB") or 
                                                     (f["participant"] == "B" and f["direction"] == "BtoA")), None)
        receiver_info = next((f for f in files if f != sender_info), None)

        if not sender_info or not receiver_info:
            print(f"⚠️ Skipping trial {trial_num} for dyad {dyad}: missing sender/receiver.")
            continue

        # Load data
        file_sender_path = os.path.join(input_dir, sender_info['filename'])
        file_receiver_path = os.path.join(input_dir, receiver_info['filename'])
        
        raw_sender = ms_load_and_filter(file_sender_path)
        raw_receiver = ms_load_and_filter(file_receiver_path)

        # Check durations
        if not np.isclose(raw_sender.times[-1], raw_receiver.times[-1], atol=1e-3):
            print(f"⚠️ Skipping trial {trial_num} for dyad {dyad}: mismatched durations.")
            continue

        # Segment
        seg_sender = ModK.predict(raw_sender, factor=10, half_window_size=1, min_segment_length=1, reject_edges=True)
        seg_receiver = ModK.predict(raw_receiver, factor=10, half_window_size=1, min_segment_length=1, reject_edges=True)

        # Extract sequences
        cluster_names = {i: name for i, name in enumerate(seg_sender.cluster_names)}
        times_sender = np.arange(len(seg_sender.labels)) / raw_sender.info['sfreq']
        times_receiver = np.arange(len(seg_receiver.labels)) / raw_receiver.info['sfreq']

        sequence_sender = [(t, cluster_names[lab]) for t, lab in zip(times_sender, seg_sender.labels) if lab != -1]
        sequence_receiver = [(t, cluster_names[lab]) for t, lab in zip(times_receiver, seg_receiver.labels) if lab != -1]

        for t, label in sequence_sender:
            sequence_rows.append({
                "dyad": dyad, "trial": trial_num, "result": sender_info['result'], 
                "modality": sender_info['modality'], "timestamp": t, 
                "sender_label": label, "receiver_label": np.nan
            })
        
        for t, label in sequence_receiver:
            sequence_rows.append({
                "dyad": dyad, "trial": trial_num, "result": sender_info['result'], 
                "modality": sender_info['modality'], "timestamp": t, 
                "sender_label": np.nan, "receiver_label": label
            })

        # Microstate Parameters
        ms_params_sender = seg_sender.compute_parameters()
        ms_params_receiver = seg_receiver.compute_parameters()
        

        entropy_sender = seg_sender.entropy()
        entropy_receiver = seg_receiver.entropy()

        entropy_rate_sender, excess_entropy_sender, *_ = excess_entropy_rate(seg_sender, history_length=9)
        entropy_rate_receiver, excess_entropy_receiver, *_ = excess_entropy_rate(seg_receiver, history_length=9)

        row = {
            "dyad": dyad,
            "trial": trial_num,
            "trial_duration": raw_sender.times[-1],
            "modality": sender_info['modality'],
            "result": sender_info['result'],
            "entropy_sender": entropy_sender,
            "entropy_rate_sender": entropy_rate_sender,
            "excess_entropy_sender": excess_entropy_sender,
            "entropy_receiver": entropy_receiver,
            "entropy_rate_receiver": entropy_rate_receiver,
            "excess_entropy_receiver": excess_entropy_receiver,
        }

        # Loop through microstates
        microstate_labels = ['A', 'B', 'C', 'D', 'F']


        for label in microstate_labels:
            row.update({
                f"timecov_sender_{label}": ms_params_sender.get(f"{label}_timecov", np.nan),
                f"meandurs_sender_{label}": ms_params_sender.get(f"{label}_meandurs", np.nan),
                f"occurrences_sender_{label}": ms_params_sender.get(f"{label}_occurrences", np.nan),
                
                f"timecov_receiver_{label}": ms_params_receiver.get(f"{label}_timecov", np.nan),
                f"meandurs_receiver_{label}": ms_params_receiver.get(f"{label}_meandurs", np.nan),
                f"occurrences_receiver_{label}": ms_params_receiver.get(f"{label}_occurrences", np.nan),
            })
        ms_data.append(row)
        print(f"✅ Processed Dyad {dyad} Trial {trial_num}")

    # Save results
    ms_df = pd.DataFrame(ms_data)
    sequence_df = pd.DataFrame(sequence_rows)

    os.makedirs(output_dir, exist_ok=True)
    ms_df.to_excel(os.path.join(output_dir, "microstate_parameters.xlsx"), index=False)
    sequence_df.to_csv(os.path.join(output_dir, "sequence_transitions.csv"), index=False)

    return ms_df, sequence_df



def compute_interbrain_features(sequence_df, timestep=0.4):
    
    """
    Interbrain features:
 """
    interbrain_data = []

    grouped = sequence_df.groupby(["dyad", "trial"])

    for (dyad, trial), group in grouped:
        group = group.sort_values("timestamp")

        # Build aligned sender-receiver state vectors
        sender_states = group['sender_label'].dropna().values
        receiver_states = group['receiver_label'].dropna().values
        timestamps = group['timestamp'].values

        min_len = min(len(sender_states), len(receiver_states))
        if min_len < 2:
            continue  # not enough data to compute anything

        sender_states = sender_states[:min_len]
        receiver_states = receiver_states[:min_len]
        timestamps = timestamps[:min_len]

        same_state = sender_states == receiver_states

        total_duration = timestamps[-1] - timestamps[0]
        if total_duration == 0:
            print(f"Skipping dyad {dyad}, trial {trial} due to zero duration.")
            continue
        
        
        # --- SHARED TIME & RATIO TOTAL ---
        shared_time = np.sum(same_state) * timestep
        shared_ratio = shared_time / total_duration
        
        # --- SHARED OCCURRENCES (transitions into shared state) ---
        transitions = np.diff(np.where(np.concatenate(([same_state[0]], same_state[:-1] != same_state[1:], [True])))[0])
        shared_occurrences = np.sum(transitions[::2]) if len(transitions) > 0 else 0

        # --- PER-MAP METRICS ---
        per_map_duration = {}
        per_map_occurrence = {}
        unique_maps = np.unique(np.concatenate((sender_states, receiver_states)))

        for map_name in unique_maps:
            per_map = (sender_states == map_name) & (receiver_states == map_name)
            per_map_time = np.sum(per_map) * timestep
            per_map_duration[map_name] = per_map_time / total_duration
            trans = np.diff(np.where(np.concatenate(([per_map[0]], per_map[:-1] != per_map[1:], [True])))[0])
            per_map_occurrence[map_name] = np.sum(trans[::2]) if len(trans) > 0 else 0

        # --- JOINT ENTROPY ---
        joint_states = [f"{s}_{r}" for s, r in zip(sender_states, receiver_states)]
        state_counts = pd.Series(joint_states).value_counts(normalize=True)
        joint_entropy = entropy(state_counts, base=2)

        # --- DFA --- # !!! feedback:need to convert into random walk first
        dfa_val = nolds.dfa(same_state.astype(float))

        # --- TIME-LAGGED SYNCHRONY ---
        sync_series = same_state.astype(float)
        lag_corr = correlate(sync_series, sync_series, mode='full')
        lags = np.arange(-len(sync_series)+1, len(sync_series))
        max_lag_idx = np.argmax(lag_corr)
        lag_samples = lags[max_lag_idx]
        lag_ms = lag_samples * timestep

        row = {
            "dyad": dyad,
            "trial": trial,
            "total_duration": total_duration,
            "total_shared_time": shared_time,
            "shared_ratio": shared_ratio,
            "shared_occurrences": shared_occurrences,
            "joint_entropy": joint_entropy,
            "dfa_shared_states": dfa_val,
            "time_lagged_sync": np.max(lag_corr),
            "lag_ms": lag_ms,
        }

        # Per-map features (coverage, occurrence, absolute time)
        microstate_labels = ['A', 'B', 'C', 'D', 'F']
        for m in microstate_labels:
            row[f"coverage_{m}"] = per_map_duration.get(m, 0)
            row[f"occurrence_{m}"] = per_map_occurrence.get(m, 0)
            row[f"time_{m}"] = per_map_duration.get(m, 0) * total_duration

        interbrain_data.append(row)

    interbrain_df = pd.DataFrame(interbrain_data)
    interbrain_df.to_excel("interbrain_data.xlsx", index=False)
    
    return interbrain_df
