# -*- coding: utf-8 -*-
"""
EEG Data Loading and Preprocessing Pipeline

This module provides functions for loading, preprocessing, and saving EEG data from BDF files.
The pipeline includes resampling, channel splitting, montage setup, event detection,
filtering, bad channel detection/interpolation, ICA artifact removal, and data saving.

Key Functions:
    - load_and_resample: Loads and resamples raw EEG data
    - split_channels: Splits data into two participants
    - rename_channels_set_montage: Standardizes channel names and sets montage
    - detect_and_annotate_events: Extracts and annotates events from status channel
    - crop: Crops data to relevant time period
    - filter_and_reference: Applies notch/bandpass filters and sets reference
    - detect_bad_channels: Identifies and interpolates bad channels
    - apply_ICA: Runs ICA for artifact removal with probability scoring and manual selection
    - save_preprocessed: Saves processed data to disk
    - preprocess_pipeline: Runs the complete preprocessing workflow
"""

import os
import logging
from typing import Tuple, Dict, List, Optional
from multiprocessing import Pool
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import pycrostates as py
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eeg_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
mne.set_log_level('ERROR')

# Constants
STANDARD_CHANNEL_NAMES = [
    'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 
    'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 
    'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'
]

EVENT_DICT = {
    10: 'word_shown_verbal', 20: 'word_shown_gesture',
    11: 'start_verbal', 12: 'end_verbal',
    23: 'start_gesture', 24: 'end_gesture',
    98: 'correct_trial', 99: 'incorrect_trial',
    101: 'start_baseline_rest', 102: 'stop_baseline_rest',
    201: 'start_baseline_gaze', 202: 'stop_baseline_gaze'
}



def load_and_resample(file_path: str, import_resample: float) -> mne.io.Raw:
    """Load and resample EEG data, bdf format"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        selected_channels = [f'1-A{i}' for i in range(1, 33)] + [f'2-A{i}' for i in range(1, 33)] + ['Status']
        raw = mne.io.read_raw_bdf(file_path, preload=True)
        raw.pick_channels(selected_channels)
        raw.resample(import_resample, verbose='error')
        logger.info(f"Successfully loaded and resampled data from {file_path}. /n Info: {raw.info}")
        return raw
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

def split_channels(raw: mne.io.Raw) -> Tuple[mne.io.Raw, mne.io.Raw]:
    """Split data into two participants and remove status channel."""
    try:
        status_channel = raw.copy().pick_channels(['Status'])
        raw.drop_channels(['Status'])

        rawA = raw.copy().pick_channels(raw.ch_names[:32])
        rawB = raw.copy().pick_channels(raw.ch_names[32:])
        
        # Process both participants in parallel
        with Pool(processes=2) as pool:
            results = pool.starmap(
                _process_participant,
                [(rawA, status_channel, 'A'), (rawB, status_channel, 'B')]
            )
        
        return results[0][0], results[1][0]
    except Exception as e:
        logger.error(f"Error splitting channels: {str(e)}")
        raise

def _process_participant(raw: mne.io.Raw, status_channel: mne.io.Raw, pid: str) -> Tuple[mne.io.Raw, str]:
    """Helper function for parallel processing of participants."""
    raw.add_channels([status_channel.copy()], force_update_info=True)
    raw.set_channel_types({'Status': 'stim'})
    return raw, pid

def rename_channels_set_montage(rawA: mne.io.Raw, rawB: mne.io.Raw, 
                              select_montage: str) -> Tuple[mne.io.Raw, mne.io.Raw]:
    """Standardize channel names and montage with parallel processing."""
    try:
        mapping_A = {f'1-A{i}': name for i, name in enumerate(STANDARD_CHANNEL_NAMES, 1)}
        mapping_B = {f'2-A{i}': name for i, name in enumerate(STANDARD_CHANNEL_NAMES, 1)}
        
        # Process in parallel
        with Pool(processes=2) as pool:
            results = pool.starmap(
                _rename_and_montage,
                [(rawA, mapping_A, select_montage), (rawB, mapping_B, select_montage)]
            )
        
        return results[0], results[1]
    except Exception as e:
        logger.error(f"Error in channel renaming/montage: {str(e)}")
        raise

def _rename_and_montage(raw: mne.io.Raw, mapping: dict, montage_name: str) -> mne.io.Raw:
    """Helper function for parallel channel processing."""
    raw.rename_channels(mapping)
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing='ignore')
    return raw

def detect_and_annotate_events(rawA: mne.io.Raw, rawB: mne.io.Raw, 
                             dyad: str) -> Tuple[mne.io.Raw, mne.io.Raw, dict, 
                                               mne.Annotations, mne.Annotations]:
    """Detect and annotate events."""
    try:
        if rawA.info['sfreq'] != rawB.info['sfreq']:
            raise ValueError(f"Sampling frequency mismatch between participants")
        
        # Process both participants in parallel
        with Pool(processes=2) as pool:
            results = pool.starmap(
                _process_events,
                [(rawA, 'A', dyad), (rawB, 'B', dyad)]
            )
        
        rawA, annotationsA = results[0]
        rawB, annotationsB = results[1]
        
        logger.info(f"Completed event detection for dyad {dyad}")
        return rawA, rawB, EVENT_DICT, annotationsA, annotationsB
    except Exception as e:
        logger.error(f"Error in event detection: {str(e)}")
        raise

def _process_events(raw: mne.io.Raw, pid: str, dyad: str) -> Tuple[mne.io.Raw, mne.Annotations]:
    """Helper function for parallel event processing."""
    annotations = mne.annotations_from_events(mne.find_events(raw, stim_channel='Status'), event_desc=EVENT_DICT, sfreq=raw.info['sfreq'])
    raw.set_annotations(annotations)
    raw.drop_channels(['Status'])
    logger.info(f"Events {dyad}{pid}: {annotations}")
    return raw, annotations

def crop(rawA: mne.io.Raw, rawB: mne.io.Raw, annotationsA: mne.Annotations, dyad: str) -> Tuple[mne.io.Raw, mne.io.Raw]:
    """Crop EEG data between start of baseline rest and end of baseline gaze."""
    
    if dyad == "02":
        stop_time = annotationsA.onset[np.where(annotationsA.description == 'stop_baseline_gaze')[0][-1]]
        rawA = rawA.copy().crop(tmin=0, tmax=stop_time)
        rawB = rawB.copy().crop(tmin=0, tmax=stop_time)
        return rawA, rawB  # recording issues w dyad 2, no start restline

    else:
        start_time = annotationsA.onset[np.where(annotationsA.description == 'start_baseline_rest')[0][0]]
        stop_time = annotationsA.onset[np.where(annotationsA.description == 'stop_baseline_gaze')[0][-1]]
    
    
        print(f"\n ⏳ Cropping dyad {dyad} between {start_time:.2f}s and {stop_time:.2f}s\n")
    
        rawA = rawA.copy().crop(tmin=start_time, tmax=stop_time)
        rawB = rawB.copy().crop(tmin=start_time, tmax=stop_time)
    
        return rawA, rawB
    
    

def filter_and_reference(raw: mne.io.Raw, notch_freq: float, 
                        low_freq_preprocess: float, high_freq_preprocess: float,
                        reference: str, filter_preprocess: str) -> mne.io.Raw:
    """Apply filtering and set average reference"""
    try:
        raw.notch_filter(notch_freq)
        raw.filter(low_freq_preprocess, high_freq_preprocess, method=filter_preprocess)
        raw.set_eeg_reference(reference, projection=True)
        raw.plot(title="After filtering", block=True)
        logger.info("Completed filtering and referencing")
        return raw
    except Exception as e:
        logger.error(f"Error in filtering: {str(e)}")
        raise

def detect_bad_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """Detect bad channels"""
    try:
        bad_channels = raw.info['bads']
        logger.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
        if bad_channels:
            raw.interpolate_bads(reset_bads=False)
            raw.plot(block=True, title="After interpolation")
        return raw
    except Exception as e:
        logger.error(f"Error in bad channel detection: {str(e)}")
        raise

def apply_ICA(raw: mne.io.Raw, identifier: str, ica_dir: str) -> Tuple[mne.io.Raw, ICA]:
    """ICA with artifact detection and interactive selection."""
    try:
        logger.info(f"Starting ICA for {identifier}")
        
        # Fit ICA
        ica = ICA(
            n_components=15,
            max_iter="auto",
            method="infomax",
            random_state=97, 
            fit_params=dict(extended=True),
        )
        
        ica.fit(raw)
        fig1 = ica.plot_components(inst=raw, title=f'ICA Components - {identifier}', show=False)
        fig1.savefig(os.path.join(ica_dir, f"{identifier}_ica_components.png"))
        plt.close(fig1)  
        

        artifact_scores = label_components(raw, ica, method="iclabel")
        labels = artifact_scores["labels"]
        
        for idx, label in enumerate(labels):
            if label != "brain":
                figs = ica.plot_properties(raw, picks=idx, show=False)
                for i, fig in enumerate(figs):
                    fig.savefig(os.path.join(ica_dir, f"{identifier}_ica_component_{idx}_{label}_{i}.png"))
                    plt.close(fig)  
        
        ica.plot_sources(raw, title=f'ICA Sources - {identifier}')
        logger.info(f"Artifact correlation scores for {identifier}: {artifact_scores}")
        
        # Manual component selection
        print("\n" + "="*50)
        print(f"ICA vomponents selection for {identifier}")
        print("="*50)
        bad_components = input(
            f"\nEnter components to reject  (comma-separated): "
        )
        bad_components = [int(x.strip()) for x in bad_components.split(",") if x.strip().isdigit()]
        
        # Apply ICA correction
        ica.exclude = bad_components
        raw_before = raw.copy()
        raw = ica.apply(raw)
    
        
        # Generate comparison plot
        ica.plot_overlay(raw, exclude=bad_components, title=f"Artifact Removal - {identifier}")
        raw_before.plot(title=f"Before ICA - {identifier}")
        raw.plot(title=f"After ICA - {identifier}", block=True)

        # Save results
        ica_path = os.path.join(ica_dir, f"{identifier}_ica.fif")
        ica.save(ica_path, overwrite=True)        
        with open(os.path.join(ica_dir, f"{identifier}_excluded_components.txt"), "w") as f:
            f.write(",".join(map(str, bad_components)))
        
        logger.info(f"Completed ICA for {identifier}, excluded components: {bad_components}")
        return raw, ica
        
    except Exception as e:
        logger.error(f"Error in ICA processing for {identifier}: {str(e)}")
        raise


def save_preprocessed(rawA: mne.io.Raw, rawB: mne.io.Raw, event_dict: Dict,
                     annotationsA: mne.Annotations, annotationsB: mne.Annotations,
                     output_dir: str, dyad: str) -> None:
    """Save data with improved error handling."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save in parallel
        with Pool(processes=2) as pool:
            pool.starmap(
                _save_participant_data,
                [
                    (rawA, annotationsA, output_dir, f"{dyad}A"),
                    (rawB, annotationsB, output_dir, f"{dyad}B")
                ]
            )
        
        logger.info(f"Successfully saved preprocessed data for dyad {dyad}")
    except Exception as e:
        logger.error(f"Error saving data for dyad {dyad}: {str(e)}")
        raise

def _save_participant_data(raw: mne.io.Raw, annotations: mne.Annotations,
                          output_dir: str, identifier: str) -> None:
    """Helper function for parallel saving."""
    raw.save(os.path.join(output_dir, f"{identifier}_preprocessed.fif"), overwrite=True)
    annotations.save(os.path.join(output_dir, f"{identifier}_annotations.csv"), overwrite=True)

def preprocess_pipeline(input_dir: str, ica_dir: str, output_dir: str, dyad: str, params: Dict,
                       require_cropping: bool = True) -> None:
    """Enhanced pipeline with all improvements."""
    try:
        # Validate inputs
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")
        if not isinstance(dyad, str):
            raise ValueError("Dyad must be a string")
        
        required_params = ["import_resample", "select_montage", "notch_freq"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        
        logger.info(f"Starting preprocessing for dyad {dyad}")
        
        # 1. Load and resample
        raw = load_and_resample(os.path.join(input_dir, f"{dyad}.bdf"), params["import_resample"])
        
        # 2. Split channels
        rawA, rawB = split_channels(raw)
        
        # 3. Standardize channel names and montage
        rawA, rawB = rename_channels_set_montage(rawA, rawB, params["select_montage"])
        
        # 4. Event detection and annotation
        rawA, rawB, event_dict, annotationsA, annotationsB = detect_and_annotate_events(rawA, rawB, dyad)
        
        # 5. cropping
        rawA, rawB = crop(rawA, rawB, annotationsA, dyad)  # Crop to baseline time
        
        # 6. Filtering and referencing 
        rawA = filter_and_reference(rawA, params["notch_freq"], params["low_freq_preprocess"], params["high_freq_preprocess"], params["reference"], params["filter_preprocess"])
        rawB = filter_and_reference(rawB, params["notch_freq"], params["low_freq_preprocess"], params["high_freq_preprocess"], params["reference"], params["filter_preprocess"])
        
        rawA = detect_bad_channels(rawA)
        rawB = detect_bad_channels(rawB)
        
        # 7. ICA 
        rawA, icaA = apply_ICA(rawA, f"{dyad}A", ica_dir)
        rawB, icaB = apply_ICA(rawB, f"{dyad}B", ica_dir)
        
        # 8. Save results
        save_preprocessed(rawA, rawB, event_dict, annotationsA, annotationsB, output_dir, dyad)
        
        logger.info(f"Successfully completed preprocessing for dyad {dyad}")
        
    except Exception as e:
        logger.error(f"Pipeline failed for dyad {dyad}: {str(e)}")
        raise

