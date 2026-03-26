"""
Parallelized version of variability_analysis.py
Processes multiple targets concurrently using multiprocessing
"""
import numpy as np
import os
import sys
import pandas as pd
import gc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from view_and_clean import offset_corrector
from metrics import (mean_med_flux, largest_amplitude, chi, percentile_amplitude, 
                     compute_lomb_scargle, amplitude_per_period, offset_warning)


def load_shared_data():
    """Load all shared data that will be used across workers"""
    print("Loading shared data files...")
    coords = pd.read_csv('merged_smc_lmc_coords_all.csv', comment='#', sep="\\s+", names=['ra', 'dec'])
    temperature_data = pd.read_csv('synth_phot_temp_estimation/ysg_temp_fitting_summary_03152026.csv')
    df_lmc = pd.read_csv('ysg_candidates/final_lmc_ysgcands_allphot_simbad.csv', comment='#')
    df_smc = pd.read_csv('ysg_candidates/final_smc_ysgcands_allphot_simbad.csv', comment='#')
    df_lmc_prefinal = pd.read_csv('ysg_candidates/prefinal_lmc_ysgcands_allphot_simbad.csv', comment='#')
    df_smc_prefinal = pd.read_csv('ysg_candidates/prefinal_smc_ysgcands_allphot_simbad.csv', comment='#')
    smc_vis_binaries = pd.read_csv('ysg_candidates/original_files_from_anna/smc_vis_bin.csv', comment='#')
    smc_opt_binaries = pd.read_csv('ysg_candidates/original_files_from_anna/smc_opt_bin.csv', comment='#') 
    lmc_vis_binaries = pd.read_csv('ysg_candidates/original_files_from_anna/lmc_vis_bin.csv', comment='#')
    lmc_opt_binaries = pd.read_csv('ysg_candidates/original_files_from_anna/lmc_opt_bin.csv', comment='#')
    
    # Validate alignment
    print(f"Loaded {len(coords)} coordinates from file")
    print(f"Loaded {len(temperature_data)} temperature records from file")
    
    if len(coords) != len(temperature_data):
        raise ValueError(f"File mismatch: {len(coords)} coords but {len(temperature_data)} temperature entries")
    
    # Verify coordinate alignment for first few rows
    for i in range(min(5, len(coords))):
        coord_ra, coord_dec = coords.iloc[i]['ra'], coords.iloc[i]['dec']
        temp_ra, temp_dec = temperature_data.iloc[i]['RA'], temperature_data.iloc[i]['DEC']
        if not (abs(coord_ra - temp_ra) < 1e-5 and abs(coord_dec - temp_dec) < 1e-5):
            raise ValueError(f"Coordinate mismatch at row {i}: coords ({coord_ra}, {coord_dec}) vs temp ({temp_ra}, {temp_dec})")
    
    print("✓ File alignment validated: coordinates match between files")
    
    return {
        'coords': coords,
        'temperature_data': temperature_data,
        'df_lmc': df_lmc,
        'df_smc': df_smc,
        'df_lmc_prefinal': df_lmc_prefinal,
        'df_smc_prefinal': df_smc_prefinal,
        'smc_vis_binaries': smc_vis_binaries,
        'smc_opt_binaries': smc_opt_binaries,
        'lmc_vis_binaries': lmc_vis_binaries,
        'lmc_opt_binaries': lmc_opt_binaries
    }


def process_single_target(target, shared_data):
    """
    Process a single target and return results as a dictionary.
    This function will be called in parallel by multiple workers.
    
    Parameters:
        target: int, the target index
        shared_data: dict containing all the shared dataframes
    
    Returns:
        dict: Dictionary containing results for this target, or None if failed
    """

    # Extract shared data (before try block so always available)
    coords = shared_data['coords']
    temperature_data = shared_data['temperature_data']
    df_lmc = shared_data['df_lmc']
    df_smc = shared_data['df_smc']
    df_lmc_prefinal = shared_data['df_lmc_prefinal']
    df_smc_prefinal = shared_data['df_smc_prefinal']
    smc_vis_binaries = shared_data['smc_vis_binaries']
    smc_opt_binaries = shared_data['smc_opt_binaries']
    lmc_vis_binaries = shared_data['lmc_vis_binaries']
    lmc_opt_binaries = shared_data['lmc_opt_binaries']
    
    # Get coordinates (before try block)
    ra = coords.iloc[target]['ra']
    dec = coords.iloc[target]['dec']
    
    # Temperature and luminosity data (before try block so always included)
    temp_row = temperature_data.iloc[target]
    teff = temp_row['final_teff_mean']
    teff_std = temp_row['final_teff_std']
    logT = temp_row['final_logT_mean']
    logT_std = temp_row['final_logT_std']
    logL = temp_row['final_logL_mean']
    logL_std = temp_row['final_logL_std']
    
    try:
        # Load and process light curve data
        df, telescopes = offset_corrector(target, additive=False, show=False)
        
        # Run all analyses
        overall_mean, means, overall_median, medians, overall_mags_mean, overall_mags_median, mags_means, mags_medians = mean_med_flux(target, df=df, telescopes=telescopes)
        chi_squared_95, chi2_threshold_95, dof_95, chi_flag_95 = chi(target, df=df, telescopes=telescopes, confidence=0.95)
        chi_squared_997, chi2_threshold_997, dof_997, chi_flag_997 = chi(target, df=df, telescopes=telescopes, confidence=0.997)
        chi_squared_68, chi2_threshold_68, dof_68, chi_flag_68 = chi(target, df=df, telescopes=telescopes, confidence=0.68)
        offset_flag = offset_warning(target)
        amplitude_5, lower_percentile_5, upper_percentile_5, mag_amplitude_5 = percentile_amplitude(target, df=df, telescopes=telescopes, tails=5)
        amplitude_1, lower_percentile_1, upper_percentile_1, mag_amplitude_1 = percentile_amplitude(target, df=df, telescopes=telescopes, tails=1)
        largest_amp, largest_amp_mag = largest_amplitude(target, df=df, telescopes=telescopes)
        lombs = compute_lomb_scargle(target, df=df, telescopes=telescopes, auto=False, median=False, subtract_median=False, samples_per_peak=10, report=False)
        best_period = lombs['best_period']
        alarm_level_flag = lombs['alarm_level_flag']
        result = amplitude_per_period(target, best_period, df=df, telescopes=telescopes, report=False)
        std_amp = result['std_amplitude']
        
        # Check if in binary catalogs
        in_smc_vis_binary = ((smc_vis_binaries['ra'] - ra).abs() < 1e-5) & ((smc_vis_binaries['dec'] - dec).abs() < 1e-5)
        in_smc_opt_binary = ((smc_opt_binaries['ra'] - ra).abs() < 1e-5) & ((smc_opt_binaries['dec'] - dec).abs() < 1e-5)
        in_lmc_vis_binary = ((lmc_vis_binaries['ra'] - ra).abs() < 1e-5) & ((lmc_vis_binaries['dec'] - dec).abs() < 1e-5)
        in_lmc_opt_binary = ((lmc_opt_binaries['ra'] - ra).abs() < 1e-5) & ((lmc_opt_binaries['dec'] - dec).abs() < 1e-5)
        is_binary = in_smc_vis_binary.any() or in_smc_opt_binary.any() or in_lmc_vis_binary.any() or in_lmc_opt_binary.any()
        
        # Check SIMBAD main type
        simbad_maintype = np.nan
        if target < 377:
            simbad_match = df_smc[((df_smc['ra'] - ra).abs() < 1e-5) & ((df_smc['dec'] - dec).abs() < 1e-5)]
        elif target >= 377 and target < 848:
            simbad_match = df_lmc[((df_lmc['ra'] - ra).abs() < 1e-5) & ((df_lmc['dec'] - dec).abs() < 1e-5)]
        elif target >= 848 and target < 1012:
            simbad_match = df_smc_prefinal[((df_smc_prefinal['ra'] - ra).abs() < 1e-5) & ((df_smc_prefinal['dec'] - dec).abs() < 1e-5)]
        else:  # target >= 1012
            simbad_match = df_lmc_prefinal[((df_lmc_prefinal['ra'] - ra).abs() < 1e-5) & ((df_lmc_prefinal['dec'] - dec).abs() < 1e-5)]
        
        if len(simbad_match) > 0 and 'main_type' in simbad_match.columns:
            simbad_maintype = simbad_match.iloc[0]['main_type']
            simbad_othertypes = simbad_match.iloc[0]['other_types']
        
        # Compile results
        result_dict = {
            'star_idx': target,
            'RA': ra,
            'DEC': dec,
            'overall_mean': overall_mean,
            'overall_median': overall_median,
            'overall_mean_mag': overall_mags_mean,
            'overall_median_mag': overall_mags_median,
            'offset_flag': offset_flag,
            'chi_squared_68': chi_squared_68,
            'chi2_threshold_68': chi2_threshold_68,
            'chi_flag_68': chi_flag_68,
            'chi_squared_95': chi_squared_95,
            'chi2_threshold_95': chi2_threshold_95,
            'chi_flag_95': chi_flag_95,
            'chi_squared_997': chi_squared_997,
            'chi2_threshold_997': chi2_threshold_997,
            'chi_flag_997': chi_flag_997,
            'amplitude_5': amplitude_5,
            'lower_percentile_5': lower_percentile_5,
            'upper_percentile_5': upper_percentile_5,
            'mag_amplitude_5': mag_amplitude_5,
            'amplitude_1': amplitude_1,
            'lower_percentile_1': lower_percentile_1,
            'upper_percentile_1': upper_percentile_1,
            'mag_amplitude_1': mag_amplitude_1,
            'largest_amp': largest_amp,
            'largest_amp_mag': largest_amp_mag,
            'best_period': best_period,
            'alarm_level_flag': alarm_level_flag,
            'std_amplitude': std_amp,
            'teff': teff,
            'teff_std': teff_std,
            'logT': logT,
            'logT_std': logT_std,
            'logL': logL,
            'logL_std': logL_std,
            'binary': is_binary,
            'SIMBAD_maintype': simbad_maintype,
            'SIMBAD_othertypes': simbad_othertypes
        }
        
        # Clean up memory
        del df, telescopes, lombs, result
        del overall_mean, means, overall_median, medians
        del overall_mags_mean, overall_mags_median, mags_means, mags_medians
        del amplitude_5, amplitude_1, largest_amp, std_amp
        
        return result_dict
        
    except Exception as e:
        print(f"Target {target} failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return partial result with at least coordinates and temperature data
        partial_result = {
            'star_idx': target,
            'RA': ra,
            'DEC': dec,
            'teff': teff,
            'teff_std': teff_std,
            'logT': logT,
            'logT_std': logT_std,
            'logL': logL,
            'logL_std': logL_std,
            'binary': False,  # Default value
            'SIMBAD_maintype': np.nan,
            'SIMBAD_othertypes': np.nan

        }
        return partial_result


def main():
    """Main execution function"""
    
    # Load shared data once
    shared_data = load_shared_data()
    n_targets = len(shared_data['coords'])
    
    # Determine number of cores to use
    n_cores = cpu_count()
    print(f"\nSystem has {n_cores} CPU cores available")
    # Use n_cores - 1 to leave one core free for system
    n_workers = max(1, n_cores - 1)
    print(f"Using {n_workers} workers for parallel processing")
    
    # Create column structure
    columns = ['star_idx',
        'RA', 'DEC', 'overall_mean', 'overall_median', 'overall_mean_mag', 'overall_median_mag',
        'offset_flag', 'chi_squared_68', 'chi2_threshold_68', 'chi_flag_68',
        'chi_squared_95', 'chi2_threshold_95', 'chi_flag_95', 
        'chi_squared_997', 'chi2_threshold_997', 'chi_flag_997',
        'amplitude_5', 'lower_percentile_5', 'upper_percentile_5', 'mag_amplitude_5',
        'amplitude_1', 'lower_percentile_1', 'upper_percentile_1', 'mag_amplitude_1',
        'largest_amp', 'largest_amp_mag', 'best_period', 'alarm_level_flag', 'std_amplitude', 
        'logT', 'logT_std', 'logL', 'logL_std', 'binary', 'SIMBAD_maintype', 'SIMBAD_othertypes', 'teff', 'teff_std'
    ]
    
    # Initialize results DataFrame
    results_df = pd.DataFrame(index=range(n_targets), columns=columns)
    results_df['star_idx'] = range(n_targets)
    results_df['RA'] = shared_data['coords']['ra'].values
    results_df['DEC'] = shared_data['coords']['dec'].values
    
    # Create partial function with shared_data bound
    process_func = partial(process_single_target, shared_data=shared_data)
    
    # Process targets in parallel with progress bar
    print(f"\nProcessing {n_targets} targets with {n_workers} parallel workers...")
    
    with Pool(processes=n_workers) as pool:
        # Use imap_unordered for better progress tracking
        results = list(tqdm(
            pool.imap_unordered(process_func, range(n_targets)),
            total=n_targets,
            desc="Analyzing targets"
        ))
    
    # Aggregate results into DataFrame
    print("\nAggregating results...")
    for result in results:
        if result is not None:
            target_idx = result['star_idx']
            for key, value in result.items():
                if key in columns:
                    results_df.loc[target_idx, key] = value
    
    # Save results
    output_filename = 'summary_results03162026.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"\n✓ Analysis complete! Results saved to {output_filename}")
    print(f"Shape: {results_df.shape}")
    print(f"Successful: {results_df['overall_mean'].notna().sum()} / {n_targets} targets")
    print(f"Failed: {results_df['overall_mean'].isna().sum()} targets")
    
    # Memory stats
    memory_mb = results_df.memory_usage(deep=True).sum() / 1024**2
    print(f"DataFrame memory usage: {memory_mb:.1f} MB")


if __name__ == '__main__':
    main()
