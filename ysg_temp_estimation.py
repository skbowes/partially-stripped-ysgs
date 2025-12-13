#!/usr/bin/env python3
"""
YSG Temperature Fitting - Production Version
Vectorized and parallelized stellar parameter fitting for YSG candidates.
Run this files with:
python ysg_temp_estimation.py --stars 848 --cores 8 (or however many cores you want to use)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import os
import sys
from multiprocessing import Pool
import functools
import argparse
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import re

# Set up logging
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs_parallel/ysg_fitting_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('ysg_fitting')

def load_data():
    """Load all required data files"""
    logger = logging.getLogger('ysg_fitting')
    logger.info("Loading data files...")
    # Load coordinates, SMC and LMC candidate data
    coords = pd.read_csv('merged_smc_lmc_coords.csv', sep=r'\s+', comment='#', names=['ra', 'dec'])
    df_lmc = pd.read_csv('./annas_candidates/final_lmc_ysgcands.csv', comment='#') # , sep="\\s+"
    df_smc = pd.read_csv('./annas_candidates/final_smc_ysgcands.csv', comment='#') # , sep="\\s+"  
    # Load synthetic photometry models
    computed_models = pd.read_csv('synth_phot_all_models.csv')
    return coords, df_smc, df_lmc, computed_models


def rchi2_with_err(star_mags,star_err,model_mags):
    '''
    Returns the reduced chi^2, accounting for errors
    Parameters:
        star_mags: Observed magnitudes
        star_err: Uncertainty on the observed magnitudes
        model_mags: Model magnitudes
    Returns:
        rchi2: Reduced chi^2 value
    '''
    N = len(star_mags)
    z = (star_mags-model_mags)/star_err
    rchi2 = np.sum(z**2)/(N-1)
    return rchi2

def observed_sed(index, coords, df_smc, df_lmc, flux=True, show=False):
    """ 
    Plots the SED for a given index in the coords dataframe
    Modified for multiprocessing - takes dataframes as parameters
    """
    RA = coords['ra'].iloc[index]
    dec = coords['dec'].iloc[index]
    
    if index < 377:
        row = df_smc[(df_smc['ra'] == RA) & (df_smc['dec'] == dec)]
    else:
        row = df_lmc[(df_lmc['ra'] == RA) & (df_lmc['dec'] == dec)] 
    
    # Bands and their Vega zero points in erg/s/cm^2/Angstrom
    band_zeropoints = {
        'Jmag': 1.11933e-9,    # J-band
        'Hmag': 3.09069e-10,   # H-band
        'Kmag': 4.20615e-11,   # K-band
        'Umag': 4.08739e-9,    # U-band
        'Bmag': 6.21086e-9,    # B-band
        'Vmag': 3.64047e-9,    # V-band
        'Imag': 9.23651e-10,   # I-band
        'uvw1_mag': 4.02204e-9,  # UVW1
        'uvw2_mag': 5.37469e-9,  # UVW2
        'uvm2_mag': 4.66117e-9   # UVM2
    }
    
    # Effective wavelengths (in Angstroms)
    band_wavelengths = {
        'uvw2_mag': 2075.69,    # UV
        'uvm2_mag': 2246.56,    # UV
        'uvw1_mag': 2715.68,    # UV
        'Umag': 3706.29,        # U
        'Bmag': 4394.48,        # B
        'Vmag': 5438.23,        # V
        'Imag': 8568.89,        # I
        'Jmag': 12350.00,       # J
        'Hmag': 16620.00,       # H
        'Kmag': 21590.00        # K
    }
    
    df = row.iloc[0]
    wavelengths = []
    fluxes = []
    flux_errors = []
    mags = []
    mag_errors = []
    band_names = []
    
    for band in band_zeropoints.keys():
        if band in df.index and not pd.isna(df[band]):
            # Get magnitude and error
            mag = df[band]
            if band != 'uvm2_mag' and band != 'uvw1_mag' and band != 'uvw2_mag':
                error_col = f'e_{band}'
                band_names.append(band.replace('mag', ''))
            else:
                error_col = f'{band}_err'
                band_names.append(band.replace('_mag', ''))
            
            # Handle missing or invalid magnitude errors
            if error_col in df.index:
                mag_err = df[error_col]
            else:
                mag_err = None
            if mag_err is None or pd.isna(mag_err) or mag_err <= 0:
                mag_err = 0.1  # Default error

            if mag_err < 0.03:
                mag_err = 0.03  # Set minimum error to 0.03 mag

            if mag_err > 0.36:
                # drop the data point if error is too large
                band = np.nan
                continue


            # Skip if magnitude itself is invalid
            if pd.isna(mag):
                continue
            
            # Convert magnitude to flux density
            flux_jy = band_zeropoints[band] * 10**(-0.4 * mag)
            flux_err_jy = flux_jy * 0.921 * mag_err
            
            wavelengths.append(band_wavelengths[band])
            fluxes.append(flux_jy)
            flux_errors.append(flux_err_jy)
            mags.append(mag)
            mag_errors.append(mag_err)
    
    # Sort by longest to shortest wavelength
    sorted_indices = np.argsort(wavelengths)[::-1]
    wavelengths = np.array(wavelengths)[sorted_indices]
    fluxes = np.array(fluxes)[sorted_indices]
    flux_errors = np.array(flux_errors)[sorted_indices]
    mags = np.array(mags)[sorted_indices]
    mag_errors = np.array(mag_errors)[sorted_indices]
    band_names = np.array(band_names)[sorted_indices]

    return wavelengths, fluxes, flux_errors, mags, mag_errors, band_names

def process_star_chunk_vectorized(star_indices_chunk, computed_models, coords, df_smc, df_lmc, iterations=1000):
    """
    Process a chunk of stars using vectorized calculations.
    This function runs in a separate process.
    Returns both detailed results and summary statistics.
    """
    
    logger = logging.getLogger(f'worker_{star_indices_chunk[0]}')
    logger.info(f"Worker started: processing stars {star_indices_chunk[0]}-{star_indices_chunk[-1]}")
    
    standard_band_order = ['J', 'H', 'K', 'U', 'B', 'V', 'I', 'uvm2', 'uvw1', 'uvw2']
    cut_bands = ['K', 'H', 'J', 'I', 'V']
    full_bands = ['K', 'H', 'J', 'I', 'V', 'B', 'U']#, 'uvw1', 'uvw2', 'uvm2']
    
    chunk_results = []
    chunk_summaries = []
    
    # Process each star in this chunk
    for i, star_idx in enumerate(star_indices_chunk):
        # Log progress periodically
        if i % 5 == 0 or i == len(star_indices_chunk) - 1:
            logger.info(f"Chunk {star_indices_chunk[0]}-{star_indices_chunk[-1]}: processing star {i+1}/{len(star_indices_chunk)} (star_idx {star_idx})")
        
        RA = coords['ra'].iloc[star_idx]
        dec = coords['dec'].iloc[star_idx]
        output_filename = f'temp_fitting/{RA}_{dec}.csv'

        try:
            obs = observed_sed(star_idx, coords, df_smc, df_lmc, show=False)
            obs_wavelengths, obs_fluxes, obs_flux_errors, obs_mags, obs_mag_errors, obs_band_names = obs
            
            # Create dictionaries for observed data
            obs_mags_dict = dict(zip(obs_band_names, obs_mags))
            obs_errors_dict = dict(zip(obs_band_names, obs_mag_errors))
            
            # Use standard band order that matches synthetic models
            common_bands = [band for band in standard_band_order if band in obs_band_names]
            
            
            # Extract matched data arrays
            matched_obs_mags = np.array([obs_mags_dict[band] for band in common_bands])
            matched_obs_errors = np.array([obs_errors_dict[band] for band in common_bands])
            
            # Filter models by metallicity
            if star_idx < 377:
                models_to_test = computed_models[computed_models['metallicity'] == -0.75]
            else:
                models_to_test = computed_models[computed_models['metallicity'] == -0.25]


            # VECTORIZED CALCULATIONS
            n_models = len(models_to_test)
            
            # Pre-extract all model data
            all_model_mags = np.array([models_to_test[band+'_mag'].values for band in common_bands]).T # each row is a model, each column a band
            model_teffs = models_to_test['teff'].values
            model_loggs = models_to_test['logg'].values  
            model_avs = models_to_test['av'].values
            model_metallicities = models_to_test['metallicity'].values
            model_lum_unscaled = models_to_test['lum_unscaled'].values
            model_filenames = models_to_test['model'].values
            
            # Pre-compute indices
            cut_indices = np.array([j for j, band in enumerate(common_bands) if band in cut_bands])
            full_indices = np.array([h for h, band in enumerate(common_bands) if band in full_bands])
            ref_idx = common_bands.index('K')
            k_idx = common_bands.index('K')

            # Generate sampled spectra
            sampled_mags = np.zeros((len(matched_obs_mags), iterations))
            for i in range(len(matched_obs_mags)):
                sampled_mags[i,:] = np.random.normal(matched_obs_mags[i], matched_obs_errors[i], iterations)

            # Vectorized calculations
            varying_K_sigmas = np.array([0, 1, 2, -1, -2])
            
            # Track best fits during loops (much more efficient)
            best_fits_full = []  # Store best fit for each iteration/K combination
            best_fits_cut = []
            
            for it in range(iterations):
                sampled_obs_mags = sampled_mags[:,it].copy()
                best_fits_full_K = []  # Store best fits for this iteration across K variations
                best_fits_cut_K = []
                
                for K in varying_K_sigmas:
                    modified_obs_mags = sampled_obs_mags.copy()
                    modified_obs_mags[k_idx] = modified_obs_mags[k_idx] + K * matched_obs_errors[k_idx]
                    
                    # Vectorized calculations for all models
                    mag_shifts = modified_obs_mags[ref_idx] - all_model_mags[:, ref_idx]
                    model_mags_shifted = all_model_mags + mag_shifts[:, np.newaxis]
                    offsets = 10**(-0.4 * modified_obs_mags[ref_idx]) / 10**(-0.4 * all_model_mags[:, ref_idx])
                    luminosities = model_lum_unscaled * offsets
                    logLs = np.log10(luminosities / 3.826e33)
                    
                    # Vectorized chi-squared
                    diff_squared = (modified_obs_mags[np.newaxis, :] - model_mags_shifted)**2
                    # chi2_full_all = np.sum(diff_squared / (matched_obs_errors[np.newaxis, :]**2), axis=1)
                    # Change fitting here:
                    chi2_full_all = np.full(n_models, np.nan)
                    if len(full_indices) > 0:
                        diff_squared_full = diff_squared[:, full_indices]
                        errors_full = matched_obs_errors[full_indices]
                        chi2_full_all = np.sum(diff_squared_full / (errors_full[np.newaxis, :]**2), axis=1)
                    
                    chi2_cut_all = np.full(n_models, np.nan)
                    if len(cut_indices) > 0:
                        diff_squared_cut = diff_squared[:, cut_indices]
                        errors_cut = matched_obs_errors[cut_indices]
                        chi2_cut_all = np.sum(diff_squared_cut / (errors_cut[np.newaxis, :]**2), axis=1)
                    
                    # Find best fits for this iteration/K combination (EFFICIENT)
                    best_full_idx = np.argmin(chi2_full_all)
                    best_full_result_K = {
                        'iteration': it, 'K_variation': K, 'teff': model_teffs[best_full_idx],
                        'logg': model_loggs[best_full_idx], 'av': model_avs[best_full_idx],
                        'logL': np.log10(model_lum_unscaled[best_full_idx] * offsets[best_full_idx] / 3.826e33),
                        'chi2_full': chi2_full_all[best_full_idx],
                        'chi2_cut': chi2_cut_all[best_full_idx] if len(cut_indices) > 0 else np.nan,
                        'model_filename': model_filenames[best_full_idx]
                    }
                    best_fits_full_K.append(best_full_result_K)
                    
                    # Best cut fit
                    best_cut_idx = np.nanargmin(chi2_cut_all)
                    best_cut_result_K = {
                        'iteration': it, 'K_variation': K, 'teff': model_teffs[best_cut_idx],
                        'logg': model_loggs[best_cut_idx], 'av': model_avs[best_cut_idx],
                        'logL': np.log10(model_lum_unscaled[best_cut_idx] * offsets[best_cut_idx] / 3.826e33),
                        'chi2_full': chi2_full_all[best_cut_idx],
                        'chi2_cut': chi2_cut_all[best_cut_idx],
                        'model_filename': model_filenames[best_cut_idx]
                    }
                    best_fits_cut_K.append(best_cut_result_K)
                    
                    
                # Here is where I'll store the best results from iterating on K:
                # take best among K variations for this iteration
                chi2_full_values_K = [r['chi2_full'] for r in best_fits_full_K]
                best_full_K_idx = np.argmin(chi2_full_values_K)
                best_fits_full.append(best_fits_full_K[best_full_K_idx])

                chi2_cut_values_K = [r['chi2_cut'] for r in best_fits_cut_K]
                best_cut_K_idx = np.nanargmin(chi2_cut_values_K)
                best_fits_cut.append(best_fits_cut_K[best_cut_K_idx])
            # END VECTORIZED CALCULATIONS
            # Create separate _full and _cut parquet files with ALL best-fit results
            os.makedirs('temp_fitting', exist_ok=True)
            
            # Write full band results (best fit for each iteration/K combination)
            if best_fits_full:
                full_data = []
                for fit in best_fits_full:
                    full_result = {
                        'star_idx': star_idx, 'RA': RA, 'DEC': dec,
                        'iteration': fit['iteration'],
                        'K_variation': fit['K_variation'],
                        'teff': fit['teff'],
                        'logg': fit['logg'], 
                        'av': fit['av'],
                        'logL': fit['logL'],
                        'chi2_full': fit['chi2_full'],
                        'chi2_cut': fit['chi2_cut'],  # Include both chi2 values for comparison
                        'model_filename': fit['model_filename']
                    }
                    full_data.append(full_result)
                
                df_full = pd.DataFrame(full_data)
                full_filename = output_filename.replace('.csv', '_full.parquet')
                df_full.to_parquet(full_filename, index=False)
                logger.info(f"Star {star_idx}: saved {len(full_data)} full band best fits to {full_filename}")
            
            # Write cut band results (best fit for each iteration/K combination)
            if best_fits_cut:
                # Only include valid (non-NaN) cut fits
                valid_cut_fits = [r for r in best_fits_cut if not np.isnan(r['chi2_cut'])]
                if valid_cut_fits:
                    cut_data = []
                    for fit in valid_cut_fits:
                        cut_result = {
                            'star_idx': star_idx, 'RA': RA, 'DEC': dec,
                            'iteration': fit['iteration'],
                            'K_variation': fit['K_variation'],
                            'teff': fit['teff'],
                            'logg': fit['logg'],
                            'av': fit['av'], 
                            'logL': fit['logL'],
                            'chi2_full': fit['chi2_full'],  # Include both chi2 values for comparison
                            'chi2_cut': fit['chi2_cut'],
                            'model_filename': fit['model_filename']
                        }
                        cut_data.append(cut_result)
                    
                    df_cut = pd.DataFrame(cut_data)
                    cut_filename = output_filename.replace('.csv', '_cut.parquet')
                    df_cut.to_parquet(cut_filename, index=False)
                    logger.info(f"Star {star_idx}: saved {len(cut_data)} cut band best fits to {cut_filename}")
            
            # Calculate summary statistics - both distribution stats and overall best fits
            summary = {'star_idx': star_idx, 'RA': RA, 'DEC': dec}
            
            # Add best fit information to summary
            if best_fits_full:
                chi2_full_values = [r['chi2_full'] for r in best_fits_full]
                best_full_idx = np.argmin(chi2_full_values)
                best_full = best_fits_full[best_full_idx]
                summary.update({
                    'teff_full': best_full['teff'],
                    'logg_full': best_full['logg'],
                    'av_full': best_full['av'], 
                    'logL_full': best_full['logL'],
                    'chi2_full': best_full['chi2_full'],
                    'best_model_full_filename': best_full['model_filename']
                })
            
            if best_fits_cut:
                chi2_cut_values = [r['chi2_cut'] for r in best_fits_cut if not np.isnan(r['chi2_cut'])]
                if chi2_cut_values:
                    valid_cut_fits = [r for r in best_fits_cut if not np.isnan(r['chi2_cut'])]
                    best_cut_idx = np.argmin(chi2_cut_values)
                    best_cut = valid_cut_fits[best_cut_idx]
                    summary.update({
                        'teff_cut': best_cut['teff'],
                        'logg_cut': best_cut['logg'],
                        'av_cut': best_cut['av'],
                        'logL_cut': best_cut['logL'], 
                        'chi2_cut': best_cut['chi2_cut'],
                        'best_model_cut_filename': best_cut['model_filename']
                    })
            
            # Full band distribution statistics
            if best_fits_full:
                teff_values_full = [r['teff'] for r in best_fits_full]
                logL_values_full = [r['logL'] for r in best_fits_full]
                logg_values_full = [r['logg'] for r in best_fits_full]
                av_values_full = [r['av'] for r in best_fits_full]
                chi2_full_values = [r['chi2_full'] for r in best_fits_full]
                
                summary.update({
                    'n_fits_full': len(best_fits_full),
                    'teff_mean_full': np.mean(teff_values_full),
                    'teff_median_full': np.median(teff_values_full),
                    'teff_std_full': np.std(teff_values_full),
                    'teff_16perc_full': np.percentile(teff_values_full, 16),
                    'teff_50perc_full': np.percentile(teff_values_full, 50),
                    'teff_84perc_full': np.percentile(teff_values_full, 84),
                    'logT_mean_full': np.mean(np.log10(teff_values_full)),
                    'logT_median_full': np.median(np.log10(teff_values_full)),
                    'logT_std_full': np.std(np.log10(teff_values_full)),
                    'logL_mean_full': np.mean(logL_values_full),
                    'logL_median_full': np.median(logL_values_full),
                    'logL_std_full': np.std(logL_values_full),
                    'logL_16perc_full': np.percentile(logL_values_full, 16),
                    'logL_50perc_full': np.percentile(logL_values_full, 50),
                    'logL_84perc_full': np.percentile(logL_values_full, 84),
                    'logg_mean_full': np.mean(logg_values_full),
                    'logg_median_full': np.median(logg_values_full),
                    'logg_std_full': np.std(logg_values_full),
                    'logg_16perc_full': np.percentile(logg_values_full, 16),
                    'logg_50perc_full': np.percentile(logg_values_full, 50),
                    'logg_84perc_full': np.percentile(logg_values_full, 84),
                    'av_mean_full': np.mean(av_values_full),
                    'av_median_full': np.median(av_values_full),
                    'av_std_full': np.std(av_values_full),
                    'av_16perc_full': np.percentile(av_values_full, 16),
                    'av_50perc_full': np.percentile(av_values_full, 50),
                    'av_84perc_full': np.percentile(av_values_full, 84),
                    'chi2_full_mean': np.mean(chi2_full_values)
                })
            
            # Cut band distribution statistics
            if best_fits_cut:
                # Only use valid (non-NaN) cut fits
                valid_cut_fits = [r for r in best_fits_cut if not np.isnan(r['chi2_cut'])]
                if valid_cut_fits:
                    teff_values_cut = [r['teff'] for r in valid_cut_fits]
                    logL_values_cut = [r['logL'] for r in valid_cut_fits]
                    logg_values_cut = [r['logg'] for r in valid_cut_fits]
                    av_values_cut = [r['av'] for r in valid_cut_fits]
                    chi2_cut_values = [r['chi2_cut'] for r in valid_cut_fits]
                    
                    summary.update({
                        'n_fits_cut': len(valid_cut_fits),
                        'teff_mean_cut': np.mean(teff_values_cut),
                        'teff_median_cut': np.median(teff_values_cut),
                        'teff_std_cut': np.std(teff_values_cut),
                        'teff_16perc_cut': np.percentile(teff_values_cut, 16),
                        'teff_50perc_cut': np.percentile(teff_values_cut, 50),
                        'teff_84perc_cut': np.percentile(teff_values_cut, 84),
                        'logT_mean_cut': np.mean(np.log10(teff_values_cut)),
                        'logT_median_cut': np.median(np.log10(teff_values_cut)),
                        'logT_std_cut': np.std(np.log10(teff_values_cut)),
                        'logL_mean_cut': np.mean(logL_values_cut),
                        'logL_median_cut': np.median(logL_values_cut),
                        'logL_std_cut': np.std(logL_values_cut),
                        'logL_16perc_cut': np.percentile(logL_values_cut, 16),
                        'logL_50perc_cut': np.percentile(logL_values_cut, 50),
                        'logL_84perc_cut': np.percentile(logL_values_cut, 84),
                        'logg_mean_cut': np.mean(logg_values_cut),
                        'logg_median_cut': np.median(logg_values_cut),
                        'logg_std_cut': np.std(logg_values_cut),
                        'logg_16perc_cut': np.percentile(logg_values_cut, 16),
                        'logg_50perc_cut': np.percentile(logg_values_cut, 50),
                        'logg_84perc_cut': np.percentile(logg_values_cut, 84),
                        'av_mean_cut': np.mean(av_values_cut),
                        'av_median_cut': np.median(av_values_cut),
                        'av_std_cut': np.std(av_values_cut),
                        'av_16perc_cut': np.percentile(av_values_cut, 16),
                        'av_50perc_cut': np.percentile(av_values_cut, 50),
                        'av_84perc_cut': np.percentile(av_values_cut, 84),
                        'chi2_cut_mean': np.mean(chi2_cut_values)
                    })
            
            # Debug: Log some statistics for verification 
            has_full = 'teff_full' in summary
            has_cut = 'teff_cut' in summary
            n_full_saved = len(best_fits_full) if best_fits_full else 0
            n_cut_saved = len([r for r in best_fits_cut if not np.isnan(r['chi2_cut'])]) if best_fits_cut else 0
            
            if has_full and has_cut:
                logger.info(f"Star {star_idx}: Saved {n_full_saved} full fits and {n_cut_saved} cut fits. Best: Full Teff={summary['teff_full']}K, Cut Teff={summary['teff_cut']}K")
            elif has_full:
                logger.info(f"Star {star_idx}: Saved {n_full_saved} full fits only. Best Teff={summary['teff_full']}K")
            elif has_cut:
                logger.info(f"Star {star_idx}: Saved {n_cut_saved} cut fits only. Best Teff={summary['teff_cut']}K")
            else:
                logger.info(f"Star {star_idx}: No valid fits found")
            
            # Add summary to results
            if summary and len(summary) > 3:  # More than just star_idx, RA, DEC
                chunk_summaries.append(summary)


        except Exception as e:
            logger.error(f"Error processing star {star_idx}: {e}")
            continue
    
    logger.info(f"Worker completed: processed {len(star_indices_chunk)} stars")
    return chunk_results, chunk_summaries

def compute_ysgs_parallel(total_star_indices, coords, df_smc, df_lmc, computed_models, n_cores=4, iterations=1000):
    """
    Parallel version using multiprocessing.
    """
    logger = logging.getLogger('ysg_fitting')
    logger.info(f"Processing {total_star_indices} stars using {n_cores} cores...")
    
    # Split stars into chunks for parallel processing
    stars_per_core = max(1, total_star_indices // n_cores)
    star_chunks = []
    
    for i in range(0, total_star_indices, stars_per_core):
        chunk_end = min(i + stars_per_core, total_star_indices)
        star_chunks.append(list(range(i, chunk_end)))
    
    logger.info(f"Split into {len(star_chunks)} chunks: {[len(chunk) for chunk in star_chunks]}")
    
    # Process chunks in parallel
    with Pool(processes=n_cores) as pool:
        # Create partial function with fixed parameters
        process_func = functools.partial(
            process_star_chunk_vectorized,
            computed_models=computed_models,
            coords=coords,
            df_smc=df_smc,
            df_lmc=df_lmc,
            iterations=iterations
        )
        # Map the function to star chunks
        chunk_outputs = pool.map(process_func, star_chunks)
    
    logger.info("All parallel processes completed!")
    
    # Collect summaries from all chunks
    all_summaries = []
    for chunk_result, chunk_summary in chunk_outputs:
        all_summaries.extend(chunk_summary)
    
    # Write summary statistics file
    if all_summaries:
        logger.info("Writing summary statistics file...")
        summary_filename = f'ysg_temp_fitting_summary_v2.csv'
        with open(summary_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)
        logger.info(f"Summary statistics for {len(all_summaries)} stars written to {summary_filename}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='YSG Temperature Fitting')
    parser.add_argument('--stars', type=int, default=8, help='Number of stars to process')
    parser.add_argument('--cores', type=int, default=4, help='Number of CPU cores to use')
    parser.add_argument('--iterations', type=int, default=1000, help='Monte Carlo iterations')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    logger.info(f"Starting YSG fitting at {datetime.now()}")
    logger.info(f"Parameters: {args.stars} stars, {args.cores} cores, {args.iterations} iterations")
    
    try:
        # Load data
        coords, df_smc, df_lmc, computed_models = load_data()
        
        # Run the parallel processing
        start_time = datetime.now()
        compute_ysgs_parallel(
            total_star_indices=args.stars,
            coords=coords,
            df_smc=df_smc,
            df_lmc=df_lmc,
            computed_models=computed_models,
            n_cores=args.cores,
            iterations=args.iterations
        )
        end_time = datetime.now()
        
        logger.info(f"Completed at {end_time}")
        logger.info(f"Total runtime: {end_time - start_time}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()