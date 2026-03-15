import numpy as np
import os
import sys
import astropy as ast
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy as sp
from astropy.timeseries import LombScargle
from astropy import units as u
import gc
import tqdm


from view_and_clean import get_telescope, find_header_line, modified_zscore, df_extract, individual_plotter, grid_plotter, offset_corrector, offset_corrector_window
from metrics import mean_med_flux, fit_plotter_flux, curve_fitter_flux,\
    largest_amplitude, offset_warning, chi, percentile_amplitude, plot_percentile_amplitude, \
        compute_lomb_scargle, plot_periodogram, plot_phase_fold, amplitude_per_period, amplitude_per_period_plot


coords = pd.read_csv('merged_smc_lmc_coords_all.csv', comment='#', sep="\\s+", names=['ra', 'dec'])
temperature_data = pd.read_csv('synth_phot_temp_estimation/ysg_temp_fitting_summary_03092026.csv')
df_lmc = pd.read_csv('ysg_candidates/final_lmc_ysgcands_allphot_simbad.csv', comment='#') # , sep="\\s+"
df_smc = pd.read_csv('ysg_candidates/final_smc_ysgcands_allphot_simbad.csv', comment='#') # , sep="\\s+"
df_lmc_prefinal = pd.read_csv('ysg_candidates/prefinal_lmc_ysgcands_allphot_simbad.csv', comment='#') # , sep="\\s+"
df_smc_prefinal = pd.read_csv('ysg_candidates/prefinal_smc_ysgcands_allphot_simbad.csv', comment='#') # , sep="\\s+"
smc_vis_binaries = pd.read_csv('ysg_candidates/original_files_from_anna/smc_vis_bin.csv', comment='#')
smc_opt_binaries = pd.read_csv('ysg_candidates/original_files_from_anna/smc_opt_bin.csv', comment='#') 
lmc_vis_binaries = pd.read_csv('ysg_candidates/original_files_from_anna/lmc_vis_bin.csv', comment='#')
lmc_opt_binaries = pd.read_csv('ysg_candidates/original_files_from_anna/lmc_opt_bin.csv', comment='#') 

# Diagnostic check
print(f"Loaded {len(coords)} coordinates from file")
print(f"Loaded {len(temperature_data)} temperature records from file")
print(f"First few rows:\n{coords.head()}")
print(f"Shape: {coords.shape}")

# Validate that files are aligned
if len(coords) != len(temperature_data):
    raise ValueError(f"File mismatch: {len(coords)} coords but {len(temperature_data)} temperature entries")
    
# Verify coordinate alignment for first few rows
for i in range(min(5, len(coords))):
    coord_ra, coord_dec = coords.iloc[i]['ra'], coords.iloc[i]['dec']
    temp_ra, temp_dec = temperature_data.iloc[i]['RA'], temperature_data.iloc[i]['DEC']
    if not (abs(coord_ra - temp_ra) < 1e-5 and abs(coord_dec - temp_dec) < 1e-5):
        raise ValueError(f"Coordinate mismatch at row {i}: coords ({coord_ra}, {coord_dec}) vs temp ({temp_ra}, {temp_dec})")
        
print("File alignment validated: coordinates match between files")

# Create DataFrame structure once before the loop
columns = ['star_idx',
    'RA', 'DEC', 'overall_mean', 'overall_median', 'overall_mean_mag', 'overall_median_mag',
    'offset_flag', 'chi_squared_68', 'chi2_threshold_68', 'chi_flag_68',
    'chi_squared_95', 'chi2_threshold_95', 'chi_flag_95', 
    'chi_squared_997', 'chi2_threshold_997', 'chi_flag_997',
    'amplitude_5', 'lower_percentile_5', 'upper_percentile_5', 'mag_amplitude_5',
    'amplitude_1', 'lower_percentile_1', 'upper_percentile_1', 'mag_amplitude_1',
    'largest_amp', 'best_period', 'alarm_level_flag', 'std_amplitude', 'logT', 'logT_std', 'logL', 'logL_std', 'binary', 'SIMBAD_maintype'
]

# Use actual number of coordinates instead of hardcoded value
n_targets = len(coords)
print(f"Initializing DataFrame for {n_targets} targets")

# Initialize DataFrame with NaN values and pre-populate coordinates
results_df = pd.DataFrame(index=range(n_targets), columns=columns, dtype=float)
results_df['RA'] = coords['ra'].values
results_df['DEC'] = coords['dec'].values
results_df['star_idx'] = range(n_targets)
results_df['teff'] = temperature_data['final_teff_mean'].values
results_df['teff_std'] = temperature_data['final_teff_std'].values
results_df['logT'] = temperature_data['final_logT_mean'].values
results_df['logT_std'] = temperature_data['final_logT_std'].values
results_df['logL'] = temperature_data['final_logL_mean'].values
results_df['logL_std'] = temperature_data['final_logL_std'].values
results_df['binary'] = False  # Default to False, will update based on binary catalogs
results_df['SIMBAD_maintype'] = np.nan

# Main analysis loop
for target in tqdm.tqdm(range(0, n_targets), desc="Analyzing targets"):
    print("Target: ", target)
    
    try:
        df, telescopes = offset_corrector(target, additive=False, show=False)
        print("loaded data successfully")

        # analysis
        overall_mean, means, overall_median, medians, overall_mags_mean, overall_mags_median, mags_means, mags_medians = mean_med_flux(target, df=df, telescopes=telescopes)
        # print("calculated mean and median successfully")
        chi_squared_95, chi2_threshold_95, dof_95, chi_flag_95 = chi(target, df=df, telescopes=telescopes, confidence=0.95)
        chi_squared_997, chi2_threshold_997, dof_997, chi_flag_997 = chi(target, df=df, telescopes=telescopes, confidence=0.997)
        chi_squared_68, chi2_threshold_68, dof_68, chi_flag_68 = chi(target, df=df, telescopes=telescopes, confidence=0.68)
        # print("calculated chi-squared 95 successfully")
        offset_flag = offset_warning(target)
        # print("calculated offset warning successfully")
        amplitude_5, lower_percentile_5, upper_percentile_5, mag_amplitude_5 = percentile_amplitude(target, df=df, telescopes=telescopes, tails=5)
        amplitude_1, lower_percentile_1, upper_percentile_1, mag_amplitude_1 = percentile_amplitude(target, df=df, telescopes=telescopes, tails=1)
        largest_amp, largest_amp_mag = largest_amplitude(target, df=df, telescopes=telescopes)
        lombs = compute_lomb_scargle(target, df=df, telescopes=telescopes, auto=False, median=False, subtract_median=False, samples_per_peak=10, report=False)
        best_period = lombs['best_period']
        alarm_level_flag = lombs['alarm_level_flag']
        result = amplitude_per_period(target, best_period, df=df, telescopes=telescopes, report=False)
        std_amp = result['std_amplitude']
        # temperature and luminosity data (validated at startup that indices align)
        # temp_row = temperature_data.iloc[target]
        # logT = temp_row['final_logT_mean']
        # logT_std = temp_row['final_logT_std']
        # logL = temp_row['final_logL_mean']
        # logL_std = temp_row['final_logL_std']

        # check if in binary catalogs
        ra, dec = results_df.loc[target, 'RA'], results_df.loc[target, 'DEC']
        in_smc_vis_binary = ((smc_vis_binaries['ra'] - ra).abs() < 1e-5) & ((smc_vis_binaries['dec'] - dec).abs() < 1e-5)
        in_smc_opt_binary = ((smc_opt_binaries['ra'] - ra).abs() < 1e-5) & ((smc_opt_binaries['dec'] - dec).abs() < 1e-5)
        in_lmc_vis_binary = ((lmc_vis_binaries['ra'] - ra).abs() < 1e-5) & ((lmc_vis_binaries['dec'] - dec).abs() < 1e-5)
        in_lmc_opt_binary = ((lmc_opt_binaries['ra'] - ra).abs() < 1e-5) & ((lmc_opt_binaries['dec'] - dec).abs() < 1e-5)
        is_binary = in_smc_vis_binary.any() or in_smc_opt_binary.any() or in_lmc_vis_binary.any() or in_lmc_opt_binary.any()
        results_df.loc[target, 'binary'] = is_binary
        # Check SIMBAD main type 
        if target < 377:
            simbad_match = df_smc[((df_smc['ra'] - ra).abs() < 1e-5) & ((df_smc['dec'] - dec).abs() < 1e-5)]
        elif target >= 377 and target < 848:
            simbad_match = df_lmc[((df_lmc['ra'] - ra).abs() < 1e-5) & ((df_lmc['dec'] - dec).abs() < 1e-5)]
        elif target >= 848 and target < 1012:
            simbad_match = df_smc_prefinal[((df_smc_prefinal['ra'] - ra).abs() < 1e-5) & ((df_smc_prefinal['dec'] - dec).abs() < 1e-5)]
        else:  # target >= 1012
            simbad_match = df_lmc_prefinal[((df_lmc_prefinal['ra'] - ra).abs() < 1e-5) & ((df_lmc_prefinal['dec'] - dec).abs() < 1e-5)]

        
        results_df.loc[target, 'overall_mean'] = overall_mean
        results_df.loc[target, 'overall_median'] = overall_median
        results_df.loc[target, 'overall_mean_mag'] = overall_mags_mean
        results_df.loc[target, 'overall_median_mag'] = overall_mags_median
        results_df.loc[target, 'offset_flag'] = offset_flag
        results_df.loc[target, 'chi_squared_68'] = chi_squared_68
        results_df.loc[target, 'chi2_threshold_68'] = chi2_threshold_68
        results_df.loc[target, 'chi_flag_68'] = chi_flag_68
        results_df.loc[target, 'chi_squared_95'] = chi_squared_95
        results_df.loc[target, 'chi2_threshold_95'] = chi2_threshold_95
        results_df.loc[target, 'chi_flag_95'] = chi_flag_95
        results_df.loc[target, 'chi_squared_997'] = chi_squared_997
        results_df.loc[target, 'chi2_threshold_997'] = chi2_threshold_997
        results_df.loc[target, 'chi_flag_997'] = chi_flag_997
        results_df.loc[target, 'amplitude_5'] = amplitude_5
        results_df.loc[target, 'lower_percentile_5'] = lower_percentile_5
        results_df.loc[target, 'upper_percentile_5'] = upper_percentile_5
        results_df.loc[target, 'mag_amplitude_5'] = mag_amplitude_5
        results_df.loc[target, 'amplitude_1'] = amplitude_1
        results_df.loc[target, 'lower_percentile_1'] = lower_percentile_1
        results_df.loc[target, 'upper_percentile_1'] = upper_percentile_1
        results_df.loc[target, 'mag_amplitude_1'] = mag_amplitude_1
        results_df.loc[target, 'largest_amp'] = largest_amp
        results_df.loc[target, 'largest_amp_mag'] = largest_amp_mag
        results_df.loc[target, 'best_period'] = best_period
        results_df.loc[target, 'alarm_level_flag'] = alarm_level_flag
        results_df.loc[target, 'std_amplitude'] = std_amp
        # results_df.loc[target, 'logT'] = logT
        # results_df.loc[target, 'logT_std'] = logT_std
        # results_df.loc[target, 'logL'] = logL
        # results_df.loc[target, 'logL_std'] = logL_std
        
        # Explicit memory cleanup to prevent accumulation
        del df, telescopes, lombs, result
        del overall_mean, means, overall_median, medians
        del overall_mags_mean, overall_mags_median, mags_means, mags_medians
        del amplitude_5, amplitude_1, largest_amp, std_amp
        
    except Exception as e:
        print(f"Target {target} failed: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback to see exactly where error occurs
        # Values remain NaN in DataFrame - no action needed
        continue
    
    # More frequent garbage collection for better memory management
    if target % 25 == 0:
        gc.collect()
        # Optional: Memory monitoring
        if target % 100 == 0:
            import psutil, os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage at target {target}: {memory_mb:.1f} MB")

# Save results
results_df.to_csv('summary_results03092026.csv', index=False)
print(f"Analysis complete! Results saved to summary_results03092026.csv")
print(f"Shape: {results_df.shape}")
print(f"Memory usage: {results_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")