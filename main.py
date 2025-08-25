# -*- coding: utf-8 -*-
"""
Main analysis pipeline
Updated August 2025
@author: Manon Haulotte (manon.haulotte@unifr.ch)

"""
# Core imports
import prep, segm, ms, stats, ibs, posthoc
from pathlib import Path
from pycrostates.io import read_cluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne
import warnings
import logging


# Configure logging and warnings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('thesis_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
np.set_printoptions(threshold=50, edgeitems=3)

mne.set_log_level('ERROR')
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)
warnings.filterwarnings('ignore', message='.*to numeric.*')
pd.options.mode.chained_assignment = None


# ==== PIPELINE CONTROL ====
STEPS = {
    'preprocessing': False,
    'segmentation': False, 
    'microstate_clustering': False,
    'feature_extraction': False,
    'statistical_analysis': False,
    'comparative_analysis': False,  
    'visualization': False,
    'posthoc': True,
    'output': True,
}


# ==== CONFIGURATION ====

dyads = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

preprocess_params = {
    "select_montage": "biosemi32",
    "import_resample": 512,
    "notch_freq": 50,
    "reference": 'average',
    "low_freq_preprocess": 1,
    "high_freq_preprocess": 70,
    "filter_preprocess": 'fir'
}

ModK_path = '3_microstates/clustering/clustering_group_k_4.fif'

ModK = read_cluster(ModK_path)
ModK.reorder_clusters(order=[1, 2, 0, 3])
ModK.invert_polarity([True, False, False, False])
ModK.rename_clusters(new_names=['A', 'B', 'C', 'D'])


dirs = {
    "raw": Path("0_data_raw"),
    "preprocessed": Path("1_data_preprocessed"),
    "ica": Path("1_data_preprocessed/ica"),
    "segmented": Path("2_data_segmented"),
    "ms_clustering": Path("3_microstates/clustering/"),
    "ms_features": Path("3_microstates/features/"),
    "ibs_results": Path("7_ibs"),
    "analyses": Path("4_analyses"),
    "figures": Path("5_figures"),
    "posthoc": Path("6_posthoc"),
    "thesis_outputs": Path("8_thesis_outputs") 
}

# Create directories
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)


# ==== MAIN PIPELINE ====
def main():
    """Main pipeline execution"""
    
    logger.info("=== STARTING THESIS PIPELINE ===")
    
    # 1. PREPROCESSING
    if STEPS['preprocessing']:
        logger.info("Step 1: Preprocessing")
        for dyad in dyads:
            prep.preprocess_pipeline(
                input_dir=dirs["raw"],
                ica_dir=dirs["ica"],
                output_dir=dirs["preprocessed"],
                dyad=dyad,
                params=preprocess_params
            )
    
    # 2. SEGMENTATION
    if STEPS['segmentation']:
        logger.info("Step 2: Segmentation")
        for dyad in dyads:
            for participant in ['A', 'B']:
                segm.segment_and_save(
                    input_dir=dirs["preprocessed"],
                    output_dir=dirs["segmented"],
                    dyad=dyad,
                    participant=participant
                )
    
    # 3. MICROSTATE CLUSTERING
    if STEPS['microstate_clustering']:
        logger.info("Step 3: Microstate Clustering")
        ms.run_microstate_analysis(
            input_dir=dirs["segmented"],
            output_dir=dirs["ms_clustering"],
            cluster_range=[3, 4, 5, 6, 7]
        )
    
    # 4. FEATURE EXTRACTION
    if STEPS['feature_extraction']:
        logger.info("Step 4: Feature Extraction")
        
        # Load clustering model
        ModK = read_cluster(ModK_path)
        ModK.reorder_clusters(order=[1, 2, 0, 3])
        ModK.invert_polarity([True, False, False, False])
        ModK.rename_clusters(new_names=['A', 'B', 'C', 'D'])
        
        # Extract microstate parameters
        parameters_df, shuffled_df, markov_df = ms.extract_parameters(
            input_dir=dirs["segmented"],
            output_dir=dirs["ms_features"],
            ModK=ModK
        )
        
        # Run IBS analysis
        ibs_df = ibs.run_ibs_analysis(
            input_dir=dirs["segmented"],
            output_dir=dirs["ibs_results"]
        )
    
   # 5. STATISTICAL ANALYSIS
    if STEPS['statistical_analysis']:
        logger.info("Step 5: Statistical Analysis")
        
        # Load data
        parameters_df = pd.read_excel(dirs["ms_features"] / "all_parameters.xlsx")
        shuffled_df = pd.read_excel(dirs["ms_features"] / "shuffled_surrogates.xlsx")
        markov_df = pd.read_excel(dirs["ms_features"] / "markov_surrogates.xlsx")
        
        # Main statistical analysis
        logger.info("Running main statistical analysis...")
        original_results = stats.run_statistical_analysis(parameters_df, dirs["analyses"])
        
        # Surrogate testing
        logger.info("Running surrogate testing...")
        features_of_interest = [
            'entropy_rate_sender', 'excess_entropy_sender',
            'entropy_rate_receiver', 'excess_entropy_receiver',
            'inter_entropy_rate', 'inter_excess_entropy',
            'shared_ratio_time', 'shared_mean_duration'
        ]
        
        shuffled_results = stats.surrogate_significance(
            parameters_df, shuffled_df, "shuffled", 
            dirs["analyses"], features_of_interest
        )
        
        markov_results = stats.surrogate_significance(
            parameters_df, markov_df, "markov",
            dirs["analyses"], features_of_interest
        )

        tp_results = stats.run_traditional_tp_analysis(
            parameters_df, 
            dirs["thesis_outputs"] / "transition_analysis"
        )
    
    # 6. COMPARATIVE ANALYSIS (New)
    if STEPS['comparative_analysis']:
        logger.info("Step 6: Comparative Analysis")
        
        # Load IBS results
        ibs_df = pd.read_excel(dirs["ibs_results"] / "task_sensitive_plv_results.xlsx")
        
        # Compare microstate vs IBS approaches
        comparison_results = ibs.compare_microstate_vs_plv_task_sensitivity(
                parameters_df, ibs_df, dirs["analyses"]
            )
    
    # 7. VISUALIZATION
    if STEPS['visualization']:
        logger.info("Step 7: Visualization")
        
        # Core statistical plots
        stats.plot_significant_features(
            parameters_df, original_results, dirs["figures"]
        )
        
        # Transition matrix analysis
        stats.plot_transition_analysis(parameters_df, dirs["figures"])
        
        # Surrogate comparison plots
        stats.plot_surrogate_results(
            parameters_df, shuffled_results, markov_results, dirs["figures"]
        )
        
    
    # 8. POSTHOC 
    if STEPS['posthoc']:
        logger.info("Step 8: PostHoc Analysis")
        
        # Load data
        parameters_df = pd.read_excel(dirs["ms_features"] / "all_parameters.xlsx")
        shuffled_df = pd.read_excel(dirs["ms_features"] / "shuffled_surrogates.xlsx")
        markov_df = pd.read_excel(dirs["ms_features"] / "markov_surrogates.xlsx")

        # Behavioral analysis
        logger.info("Running behavioral analysis...")
        posthoc.behavioral_analysis_extended(parameters_df, dirs["posthoc"])
        logger.info("Generating behavioral figures...")
        posthoc.generate_behavioral_figure(parameters_df, dirs["posthoc"])
        
        # Surrogate analysis 
        logger.info("Running surrogate analysis...")
        matched_results_df, stat_df, significant_shuffled, significant_markov = posthoc.analyze_interbrain_vs_surrogates(
            parameters_df, shuffled_df, markov_df, dirs["posthoc"]
        )
        
        # Transition matrix analysis
        logger.info("Running transition matrix analysis...")
        transition_results = posthoc.analyze_transition_matrices(
            parameters_df, dirs["posthoc"]
        )
        
        # Thesis figure analysis 
        logger.info("Generating thesis figures...")
        figure_results = stats.generate_thesis_figures(parameters_df, dirs["posthoc"])
        

    # 9. OUTPUT
    if STEPS['output']:
        logger.info("Step 9: Generating Final Results")
        
        # Generate thesis-ready outputs
        posthoc.generate_thesis_ready_outputs(
            parameters_df, original_results, dirs["thesis_outputs"]
        )


if __name__ == "__main__":
    main()