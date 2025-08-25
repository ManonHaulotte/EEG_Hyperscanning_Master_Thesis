# -*- coding: utf-8 -*-
"""
Segmentation Pipeline

Key Functions:
    - load_preprocessed_data
    - segment_and_save
    - extract_baseline_periods
"""

import os
import logging
from typing import Optional, Tuple, Dict
import pandas as pd
import mne
from mne import Annotations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eeg_segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
mne.set_log_level('ERROR')

# Constants
MODALITY_MAP = {
    'gesture': ('start_gesture', 'end_gesture'),
    'verbal': ('start_verbal', 'end_verbal')
}


def load_preprocessed_data(input_dir: str, dyad: str, participant: str) -> Optional[mne.io.Raw]:
    """Load preprocessed EEG data."""
    try:
        logger.info(f"Loading preprocessed data for {dyad}{participant}...")
        file_path = os.path.join(input_dir, f"{dyad}{participant}_preprocessed.fif")

        if not os.path.exists(file_path):
            logger.error(f"File {file_path} not found!")
            return None

        raw = mne.io.read_raw_fif(file_path, preload=True)
        logger.info(f"Loaded {raw}")
        logger.debug(f"Annotations: {raw.annotations}")
        return raw
    except Exception as e:
        logger.exception(f"Error loading data for {dyad}{participant}: {e}")
        return None


def extract_baseline_periods(raw: mne.io.Raw, dyad: str, participant: str, output_dir: str) -> None:
    """Extract and save baseline rest and gaze periods."""
    logger.info(f"Extracting baseline periods for {dyad}{participant}...")

    # Skip baseline rest for dyad 2
    if dyad == '02':
        baseline_types = ["baseline_gaze"]
    else:
        baseline_types = ["baseline_rest", "baseline_gaze"]

    onsets = raw.annotations.onset
    descriptions = raw.annotations.description

    for btype in baseline_types:
        start_key = f"start_{btype}"
        stop_key = f"stop_{btype}"

        try:
            start_indices = [i for i, desc in enumerate(descriptions) if desc == start_key]
            stop_indices = [i for i, desc in enumerate(descriptions) if desc == stop_key]

            if len(start_indices) != len(stop_indices):
                logger.warning(f"Mismatched number of {start_key} and {stop_key} for {dyad}{participant}. Skipping.")
                continue

            for idx, (start_idx, stop_idx) in enumerate(zip(start_indices, stop_indices)):
                start_time = onsets[start_idx] - raw.annotations.onset[0]
                stop_time = onsets[stop_idx] - raw.annotations.onset[0]
                duration = stop_time - start_time

                if duration <= 0:
                    logger.warning(f"Invalid {btype} duration #{idx+1} for {dyad}{participant}. Skipping.")
                    continue

                try:
                    segment = raw.copy().crop(tmin=start_time, tmax=stop_time)
                    output_file = os.path.join(output_dir, f"{dyad}{participant}_{btype}_{idx+1}.fif")
                    segment.save(output_file, overwrite=True)
                    logger.info(f"Saved {btype} baseline #{idx+1} to {output_file}")
                except Exception as e:
                    logger.exception(f"Error saving {btype} segment #{idx+1}: {e}")

                
        except StopIteration:
            logger.warning(f"{start_key} or {stop_key} not found. Skipping {btype} for {dyad}{participant}.")
            continue
        except Exception as e:
            logger.exception(f"Error finding baseline markers for {btype}: {e}")
            continue


def segment_and_save(input_dir: str, output_dir: str, dyad: str, participant: str) -> None:
    """Segment trials from raw EEG and save them."""
    try:
        metadata_path = 'metadata.csv'
        if not os.path.exists(metadata_path):
            logger.error("metadata.csv not found!")
            return

        metadata_all = pd.read_csv(metadata_path)
        metadata_all['dyad'] = metadata_all['dyad'].astype(str).str.zfill(2)
        dyad_metadata = metadata_all[metadata_all['dyad'] == dyad]
        if dyad_metadata.empty:
            logger.warning(f"No metadata found for dyad {dyad}")
            return
        logger.info(f"Found metadata for dyad {dyad}. Processing...")

        raw = load_preprocessed_data(input_dir, dyad, participant)
        if raw is None:
            logger.warning(f"Skipping {dyad}{participant} due to missing data.")
            return

        # Extract baselines
        extract_baseline_periods(raw, dyad, participant, output_dir)

        sfreq = raw.info["sfreq"]

        events, event_id = mne.events_from_annotations(raw)
        event_id["response/correct"] = event_id.pop("correct_trial")
        event_id["response/incorrect"] = event_id.pop("incorrect_trial")

        # Extract metadata
        metadata_tmin, metadata_tmax = -3, 100
        row_events = ["start_gesture", "start_verbal"]
        keep_first = "response"

        metadata, events, event_id = mne.epochs.make_metadata(
            events=events,
            event_id=event_id,
            tmin=metadata_tmin,
            tmax=metadata_tmax,
            sfreq=sfreq,
            row_events=row_events,
            keep_first=keep_first,
        )

        # Clean and align metadata
        drop_cols = ['response', 'response/incorrect', 'response/correct',
                     'start_baseline_gaze', 'start_baseline_rest',
                     'stop_baseline_gaze', 'stop_baseline_rest',
                     'word_shown_gesture', 'word_shown_verbal']
        metadata = metadata.drop(columns=drop_cols, errors='ignore')
        metadata = metadata.rename(columns={"first_response": "result"})
        metadata = metadata.fillna(0).reset_index(drop=True)

        event_sample = events[:, 0][metadata.index]
        event_onsets = event_sample / sfreq

        for col in ["start_gesture", "end_gesture", "start_verbal", "end_verbal"]:
            if col in metadata.columns:
                metadata[col] += event_onsets

        # Segment and save trials
        for i, row in metadata.iterrows():
            try:
                modality = "gesture" if "gesture" in row["event_name"] else "verbal"
                start_key, stop_key = MODALITY_MAP[modality]
                start_abs = row[start_key]
                stop_abs = row[stop_key]


                label = "correct" if row["result"] == "correct" else "incorrect"
                tag = f"{modality}_{label}"

                trial_metadata = dyad_metadata[dyad_metadata['n_trial'] == i + 1]
                if trial_metadata.empty or trial_metadata['guess'].values[0] == 'pass':
                    logger.info(f"Skipping trial {i + 1} for {dyad}{participant} (pass or missing).")
                    continue

                emitter = trial_metadata['emitter'].values[0]
                direction = 'AtoB' if emitter == 'A' else 'BtoA' if emitter == 'B' else ''
                tag = f"{modality}_{label}_{direction}"

                start_time = start_abs - raw.annotations.onset[0]
                stop_time = stop_abs - raw.annotations.onset[0]

                logger.info(f"Trial {i + 1}: {tag} | Start: {start_time:.2f}s Stop: {stop_time:.2f}s")

                segment = raw.copy().crop(tmin=start_time, tmax=stop_time)
                output_file = os.path.join(output_dir, f"{dyad}{participant}_trial{i + 1}_{tag}.fif")
                segment.save(output_file, overwrite=True)
                logger.info(f"Saved trial {i + 1} to {output_file}")
            except Exception as e:
                logger.exception(f"Error processing trial {i + 1} for {dyad}{participant}: {e}")

        plot_sample_segments(output_dir, dyad, participant)
        
    except Exception as e:
        logger.exception(f"Failed to complete segmentation for {dyad}{participant}: {e}")


def plot_sample_segments(output_dir: str, dyad: str, participant: str):
    """
    Plots one of each segment type (baseline_rest, baseline_gaze, gesture, verbal)
    if available.
    """
    logger.info(f"Sampling segments to plot for {dyad}{participant}...")

    all_files = os.listdir(output_dir)
    prefix = f"{dyad}{participant}_"

    def try_plot(name_contains: str):
        try:
            fname = next(f for f in all_files if f.startswith(prefix) and name_contains in f)
            raw = mne.io.read_raw_fif(os.path.join(output_dir, fname), preload=True)
            logger.info(f"Plotting: {fname}")
            raw.plot(title=f"{fname}")
        except StopIteration:
            logger.warning(f"No file found for '{name_contains}' in {dyad}{participant}")
        except Exception as e:
            logger.warning(f"Failed to plot segment for '{name_contains}': {e}")

    try_plot("baseline_rest_1")
    try_plot("baseline_gaze_1")
    try_plot("baseline_rest_2")
    try_plot("baseline_gaze_2")
    try_plot("gesture")
    try_plot("verbal")
