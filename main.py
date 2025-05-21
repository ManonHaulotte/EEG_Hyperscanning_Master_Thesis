# -*- coding: utf-8 -*-
"""
Updated May 15 2025
@author: Manon Haulotte (manon.haulotte@unifr.ch)

This code is 

Data were aquired

input: 

"""
import prep, segm, ms, stats, rnn
from pathlib import Path
import joblib
import pandas as pd

# ==== Select which steps to run ====
run_preprocessing = False
run_segmentation = False
run_microstates_clustering = False
run_microstates_sequencing = False
run_stats_analysis = False
run_rnn_hyperparamters = False
run_rnn = False

# Dyads to analyse: 
dyads  = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# Parameters: 
preprocess_params = {
    "select_montage": "biosemi32",
    "import_resample": 512,
    "notch_freq": 50,
    "reference": 'average',
    "low_freq_preprocess": 1,
    "high_freq_preprocess": 70,
    "filter_preprocess": 'fir'
}

ModK_path = 'modk_all_5_clusters.pkl'
meta_df = pd.read_csv("metadata.csv")

# Directories:
dirs = {
    "raw": Path("0_data_raw"),
    "preprocessed": Path("1_data_preprocessed"),
    "ica": Path("1_data_preprocessed/ica"),
    "segmented": Path("2_data_segmented"),
    "ms_clustering": Path("3_microstates/clustering"),
    "ms_fitting": Path("3_microstates/fitting"),
    "ms_features": Path("3_microstates/features"),
    "analyses": Path("4_analyses"),
    "figures": Path("5_figures"),
}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)


# ==========================================================
# ----------------------  Pipelines -----------------------
# ==========================================================

for dyad in dyads:
    if run_preprocessing:
        prep.preprocess_pipeline(input_dir=dirs["raw"], 
                                 ica_dir=dirs["ica"],
                                 output_dir=dirs["preprocessed"], 
                                 dyad=dyad, 
                                 params=preprocess_params)
        
        
    if run_segmentation:
        for participant in ['A', 'B']:
            segm.segment_and_save(input_dir=dirs["preprocessed"], 
                                  output_dir=dirs["segmented"], 
                                  dyad=dyad,
                                  participant=participant)
    
            
    if run_microstates_clustering:
        ms.run_microstate_analysis(input_dir: str,
                            output_dir: str,
                            cluster_range: List[int] = [3, 4, 5, 6, 7],
                            subjects: List[str] = None)
        

    if run_microstates_sequencing:
        ModK = joblib.load(ModK_path)
        ModK.plot()
        ModK.reorder_clusters(order=[4, 3, 2, 1, 0])
        ModK.invert_polarity([True, True, True, False, True])
        ModK.rename_clusters(new_names=['A', 'B', 'C', 'D', 'F'])
        ModK.plot()
        print(ModK.GEV_)

        ms_df, sequence_df = ms.extract_sequences_and_intrabrain_parameters(input_dir=dirs["segmented"],
                                                                            output_dir=dirs["ms_fitting"],
                                                                            ModK=ModK)
        interbrain_df = ms.compute_interbrain_features(sequence_df)

    if run_stats_analysis:
        results_df = stats.run_statistical_analysis(ms_df, interbrain_df)
        full_df = pd.merge(ms_df, interbrain_df, on=["dyad", "trial"])
        stats.plot_significant_features(results_df, full_df)

        # pseudo_interbrain_df = ms.compute_pseudo_pairs(sequence_df)


    if run_rnn_hyperparamters:
        merged_df = ms.merge_microstate_sequences(sequence_df, step_ms=40, sfreq=128)
        merged_df = merged_df.merge(
            meta_df[['dyad', 'n_trial', 'result']], 
            left_on=['dyad', 'trial'], 
            right_on=['dyad', 'n_trial'], 
            how='left'
        )
        merged_df = merged_df.drop(columns=['n_trial'])
        rnn.hyperparameter_grid_search(merged_df, device='cpu')

    if run_rnn:
        rnn.run(merged_df, device='cpu')



