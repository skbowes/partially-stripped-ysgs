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
    # Near-infrared (2MASS)
    'Jmag':3.0596e-10,    # J-band 3.0596e-10 # formerly was 1.11933e-9 
    'Hmag':1.11064e-10,    # H-band 1.11064e-10 # formerly was 3.09069e-10 
    'Kmag':4.17999e-11,     # K-band 4.17999e-11 # formerly was 4.20615e-11
    # Optical (Johnson-Cousins)
    'Umag':4.08739e-9,    # U-band
    'Bmag':6.21086e-9,    # B-band
    'Vmag':3.64047e-9,    # V-band
    'Imag':9.23651e-10,    # I-band
    # UV (Swift UVOT)
    'uvw1_mag':4.02204e-9,  # UVW
    'uvw2_mag':5.37469e-9,   # UVW2
    'uvm2_mag':4.66117e-9   # UVM2
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
    min_avs = pd.read_csv('ysg_candidate_extinctions.csv')

    
    standard_band_order = ['J', 'H', 'K', 'U', 'B', 'V', 'I', 'uvm2', 'uvw1', 'uvw2']
    V_redder_bands = ['K', 'H', 'J', 'I', 'V']
    B_redder_bands = ['K', 'H', 'J', 'I', 'V', 'B']
    U_redder_bands = ['K', 'H', 'J', 'I', 'V', 'B', 'U']#, 'uvw1', 'uvw2', 'uvm2']
    
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
            min_av = min_avs[(min_avs['RA'] == RA) & (min_avs['DEC'] == dec)]['av_eden'].values[0]
            
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
            
            # Pre-extract all model data, only take models with av >= min_av 
            av_mask = models_to_test['av'] >= min_av
            filtered_models = models_to_test[av_mask]
            all_model_mags = np.array([filtered_models[band+'_mag'].values for band in common_bands]).T # each row is a model, each column a band
            model_teffs = filtered_models['teff'].values
            model_loggs = filtered_models['logg'].values
            model_avs = filtered_models['av'].values
            model_metallicities = filtered_models['metallicity'].values
            model_lum_unscaled = filtered_models['lum_unscaled'].values
            model_filenames = filtered_models['model'].values
            
            # Pre-compute indices
            V_redder_indices = np.array([j for j, band in enumerate(common_bands) if band in V_redder_bands])
            B_redder_indices = np.array([k for k, band in enumerate(common_bands) if band in B_redder_bands])
            U_redder_indices = np.array([h for h, band in enumerate(common_bands) if band in U_redder_bands])
            ref_idx = common_bands.index('K')
            k_idx = common_bands.index('K')

            # Generate sampled spectra
            sampled_mags = np.zeros((len(matched_obs_mags), iterations))
            for i in range(len(matched_obs_mags)):
                sampled_mags[i,:] = np.random.normal(matched_obs_mags[i], matched_obs_errors[i], iterations)

            # Vectorized calculations
            varying_K_sigmas = np.array([0, 1, 2, -1, -2])
            
            # Track best fits during loops (much more efficient)
            best_fits_U_redder = []  # Store best fit for each iteration/K combination
            best_fits_B_redder = []
            best_fits_V_redder = []
            
            for it in range(iterations):
                sampled_obs_mags = sampled_mags[:,it].copy()
                best_fits_U_redder_K = []  # Store best fits for this iteration across K variations
                best_fits_B_redder_K = []
                best_fits_V_redder_K = []
                
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
                    chi2_U_redder_all = np.full(n_models, np.nan)
                    if len(U_redder_indices) > 0:
                        diff_squared_U_redder = diff_squared[:, U_redder_indices] # calculates (obs - model)^2 for U redder bands
                        errors_U_redder = matched_obs_errors[U_redder_indices]
                        chi2_U_redder_all = np.sum(diff_squared_U_redder / (errors_U_redder[np.newaxis, :]**2), axis=1) # sum over all models

                    chi2_B_redder_all = np.full(n_models, np.nan)
                    if len(B_redder_indices) > 0:
                        diff_squared_B_redder = diff_squared[:, B_redder_indices]
                        errors_B_redder = matched_obs_errors[B_redder_indices]
                        chi2_B_redder_all = np.sum(diff_squared_B_redder / (errors_B_redder[np.newaxis, :]**2), axis=1) # sum over all models
                    
                    chi2_V_redder_all = np.full(n_models, np.nan)
                    if len(V_redder_indices) > 0:
                        diff_squared_V_redder = diff_squared[:, V_redder_indices]
                        errors_V_redder = matched_obs_errors[V_redder_indices]
                        chi2_V_redder_all = np.sum(diff_squared_V_redder / (errors_V_redder[np.newaxis, :]**2), axis=1) # sum over all models
                    
                    # Find best fits for this iteration/K combination (EFFICIENT)
                    best_U_redder_idx = np.argmin(chi2_U_redder_all)
                    best_U_redder_result_K = {
                        'iteration': it, 'K_variation': K, 'teff': model_teffs[best_U_redder_idx],
                        'logg': model_loggs[best_U_redder_idx], 'av': model_avs[best_U_redder_idx],
                        'logL': np.log10(model_lum_unscaled[best_U_redder_idx] * offsets[best_U_redder_idx] / 3.826e33),
                        'chi2_U_redder': chi2_U_redder_all[best_U_redder_idx],
                        'chi2_B_redder': chi2_B_redder_all[best_U_redder_idx], #if len(B_redder_indices) > 0 else np.nan,
                        'chi2_V_redder': chi2_V_redder_all[best_U_redder_idx], #if len(V_redder_indices) > 0 else np.nan,
                        'model_filename': model_filenames[best_U_redder_idx]
                    }
                    best_fits_U_redder_K.append(best_U_redder_result_K)
                    
                    # Best B_redder fit
                    best_B_redder_idx = np.nanargmin(chi2_B_redder_all)
                    best_B_redder_result_K = {
                        'iteration': it, 'K_variation': K, 'teff': model_teffs[best_B_redder_idx],
                        'logg': model_loggs[best_B_redder_idx], 'av': model_avs[best_B_redder_idx],
                        'logL': np.log10(model_lum_unscaled[best_B_redder_idx] * offsets[best_B_redder_idx] / 3.826e33),
                        'chi2_U_redder': chi2_U_redder_all[best_B_redder_idx],
                        'chi2_B_redder': chi2_B_redder_all[best_B_redder_idx],
                        'chi2_V_redder': chi2_V_redder_all[best_B_redder_idx],
                        'model_filename': model_filenames[best_B_redder_idx]
                    }
                    best_fits_B_redder_K.append(best_B_redder_result_K)

                    best_V_redder_idx = np.nanargmin(chi2_V_redder_all)
                    best_V_redder_result_K = {
                        'iteration': it, 'K_variation': K, 'teff': model_teffs[best_V_redder_idx],
                        'logg': model_loggs[best_V_redder_idx], 'av': model_avs[best_V_redder_idx],
                        'logL': np.log10(model_lum_unscaled[best_V_redder_idx] * offsets[best_V_redder_idx] / 3.826e33),
                        'chi2_U_redder': chi2_U_redder_all[best_V_redder_idx],
                        'chi2_B_redder': chi2_B_redder_all[best_V_redder_idx],
                        'chi2_V_redder': chi2_V_redder_all[best_V_redder_idx],
                        'model_filename': model_filenames[best_V_redder_idx]
                    }
                    best_fits_V_redder_K.append(best_V_redder_result_K)


                # Here is where I'll store the best results from iterating on K:
                # take best among K variations for this iteration
                chi2_U_redder_values_K = [r['chi2_U_redder'] for r in best_fits_U_redder_K]
                best_U_redder_K_idx = np.argmin(chi2_U_redder_values_K)
                best_fits_U_redder.append(best_fits_U_redder_K[best_U_redder_K_idx])

                chi2_B_redder_values_K = [r['chi2_B_redder'] for r in best_fits_B_redder_K]
                best_B_redder_K_idx = np.nanargmin(chi2_B_redder_values_K)
                best_fits_B_redder.append(best_fits_B_redder_K[best_B_redder_K_idx])

                chi2_V_redder_values_K = [r['chi2_V_redder'] for r in best_fits_V_redder_K]
                best_V_redder_K_idx = np.nanargmin(chi2_V_redder_values_K)
                best_fits_V_redder.append(best_fits_V_redder_K[best_V_redder_K_idx])
            # END VECTORIZED CALCULATIONS
            # Create separate _U_redder and _B_redder and _V_redder parquet files with ALL best-fit results
            os.makedirs('temp_fitting', exist_ok=True)
            
            # Write full band results (best fit for each iteration/K combination)
            U_redder_data = []
            for fit in best_fits_U_redder:
                U_redder_result = {
                    'star_idx': star_idx, 'RA': RA, 'DEC': dec,
                    'iteration': fit['iteration'],
                    'K_variation': fit['K_variation'],
                    'teff': fit['teff'],
                    'logg': fit['logg'], 
                    'av': fit['av'],
                    'logL': fit['logL'],
                    'chi2_U_redder': fit['chi2_U_redder'],
                    'chi2_B_redder': fit['chi2_B_redder'],  # Include both chi2 values for comparison
                    'chi2_V_redder': fit['chi2_V_redder'],
                    'model_filename': fit['model_filename']
                }
                U_redder_data.append(U_redder_result)
                
            df_U_redder = pd.DataFrame(U_redder_data)
            U_redder_filename = output_filename.replace('.csv', '_U_redder.parquet')
            df_U_redder.to_parquet(U_redder_filename, index=False)
            logger.info(f"Star {star_idx}: saved {len(U_redder_data)} U redder band best fits to {U_redder_filename}")

            B_redder_data = []
            for fit in best_fits_B_redder:
                B_redder_result = {
                    'star_idx': star_idx, 'RA': RA, 'DEC': dec,
                    'iteration': fit['iteration'],
                    'K_variation': fit['K_variation'],
                    'teff': fit['teff'],
                    'logg': fit['logg'], 
                    'av': fit['av'],
                    'logL': fit['logL'],
                    'chi2_U_redder': fit['chi2_U_redder'],
                    'chi2_B_redder': fit['chi2_B_redder'],  # Include both chi2 values for comparison
                    'chi2_V_redder': fit['chi2_V_redder'],
                    'model_filename': fit['model_filename']
                }
                B_redder_data.append(B_redder_result)
            df_B_redder = pd.DataFrame(B_redder_data)
            B_redder_filename = output_filename.replace('.csv', '_B_redder.parquet')
            df_B_redder.to_parquet(B_redder_filename, index=False)
            logger.info(f"Star {star_idx}: saved {len(B_redder_data)} B redder band best fits to {B_redder_filename}")

            
            # Write cut band results (best fit for each iteration/K combination)
            # Only include valid (non-NaN) cut fits
            V_redder_data = []
            for fit in best_fits_V_redder:
                V_redder_result = {
                    'star_idx': star_idx, 'RA': RA, 'DEC': dec,
                    'iteration': fit['iteration'],
                    'K_variation': fit['K_variation'],
                    'teff': fit['teff'],
                    'logg': fit['logg'],
                    'av': fit['av'], 
                    'logL': fit['logL'],
                    'chi2_U_redder': fit['chi2_U_redder'],  # Include both chi2 values for comparison
                    'chi2_B_redder': fit['chi2_B_redder'],
                    'chi2_V_redder': fit['chi2_V_redder'],
                    'model_filename': fit['model_filename']
                }
                V_redder_data.append(V_redder_result)
                
            df_V_redder = pd.DataFrame(V_redder_data)
            V_redder_filename = output_filename.replace('.csv', '_V_redder.parquet')
            df_V_redder.to_parquet(V_redder_filename, index=False)
            logger.info(f"Star {star_idx}: saved {len(V_redder_data)} V redder band best fits to {V_redder_filename}")
        
            # Calculate summary statistics - both distribution stats and overall best fits
            summary = {'star_idx': star_idx, 'RA': RA, 'DEC': dec}
            
            # Add best fit information to summary
            if best_fits_U_redder:
                chi2_U_redder_values = [r['chi2_U_redder'] for r in best_fits_U_redder]
                best_U_redder_idx = np.argmin(chi2_U_redder_values)
                best_U_redder = best_fits_U_redder[best_U_redder_idx]
                summary.update({
                    'teff_U_redder': best_U_redder['teff'],
                    'logg_U_redder': best_U_redder['logg'],
                    'av_U_redder': best_U_redder['av'], 
                    'logL_U_redder': best_U_redder['logL'],
                    'chi2_U_redder': best_U_redder['chi2_U_redder'],
                    'best_model_U_redder_filename': best_U_redder['model_filename']
                })
            
            if best_fits_B_redder:
                chi2_B_redder_values = [r['chi2_B_redder'] for r in best_fits_B_redder if not np.isnan(r['chi2_B_redder'])]
                if chi2_B_redder_values:
                    valid_B_redder_fits = [r for r in best_fits_B_redder if not np.isnan(r['chi2_B_redder'])]
                    best_B_redder_idx = np.argmin(chi2_B_redder_values)
                    best_B_redder = valid_B_redder_fits[best_B_redder_idx]
                    summary.update({
                        'teff_B_redder': best_B_redder['teff'],
                        'logg_B_redder': best_B_redder['logg'],
                        'av_B_redder': best_B_redder['av'],
                        'logL_B_redder': best_B_redder['logL'], 
                        'chi2_B_redder': best_B_redder['chi2_B_redder'],
                        'best_model_B_redder_filename': best_B_redder['model_filename']
                    })

            if best_fits_V_redder:
                chi2_V_redder_values = [r['chi2_V_redder'] for r in best_fits_V_redder if not np.isnan(r['chi2_V_redder'])]
                if chi2_V_redder_values:
                    valid_V_redder_fits = [r for r in best_fits_V_redder if not np.isnan(r['chi2_V_redder'])]
                    best_V_redder_idx = np.argmin(chi2_V_redder_values)
                    best_V_redder = valid_V_redder_fits[best_V_redder_idx]
                    summary.update({
                        'teff_V_redder': best_V_redder['teff'],
                        'logg_V_redder': best_V_redder['logg'],
                        'av_V_redder': best_V_redder['av'],
                        'logL_V_redder': best_V_redder['logL'],
                        'chi2_V_redder': best_V_redder['chi2_V_redder'],
                        'best_model_V_redder_filename': best_V_redder['model_filename']
                    })
            
            # U redder band distribution statistics
            if best_fits_U_redder:
                teff_values_U_redder = [r['teff'] for r in best_fits_U_redder]
                logL_values_U_redder = [r['logL'] for r in best_fits_U_redder]
                logg_values_U_redder = [r['logg'] for r in best_fits_U_redder]
                av_values_U_redder = [r['av'] for r in best_fits_U_redder]
                chi2_U_redder_values = [r['chi2_U_redder'] for r in best_fits_U_redder]
                
                summary.update({
                    'n_fits_U_redder': len(best_fits_U_redder),
                    'teff_mean_U_redder': np.mean(teff_values_U_redder),
                    'teff_median_U_redder': np.median(teff_values_U_redder),
                    'teff_std_U_redder': np.std(teff_values_U_redder),
                    'teff_16perc_U_redder': np.percentile(teff_values_U_redder, 16),
                    'teff_50perc_U_redder': np.percentile(teff_values_U_redder, 50),
                    'teff_84perc_U_redder': np.percentile(teff_values_U_redder, 84),
                    'logT_mean_U_redder': np.mean(np.log10(teff_values_U_redder)),
                    'logT_median_U_redder': np.median(np.log10(teff_values_U_redder)),
                    'logT_std_U_redder': np.std(np.log10(teff_values_U_redder)),
                    'logL_mean_U_redder': np.mean(logL_values_U_redder),
                    'logL_median_U_redder': np.median(logL_values_U_redder),
                    'logL_std_U_redder': np.std(logL_values_U_redder),
                    'logL_16perc_U_redder': np.percentile(logL_values_U_redder, 16),
                    'logL_50perc_U_redder': np.percentile(logL_values_U_redder, 50),
                    'logL_84perc_U_redder': np.percentile(logL_values_U_redder, 84),
                    'logg_mean_U_redder': np.mean(logg_values_U_redder),
                    'logg_median_U_redder': np.median(logg_values_U_redder),
                    'logg_std_U_redder': np.std(logg_values_U_redder),
                    'logg_16perc_U_redder': np.percentile(logg_values_U_redder, 16),
                    'logg_50perc_U_redder': np.percentile(logg_values_U_redder, 50),
                    'logg_84perc_U_redder': np.percentile(logg_values_U_redder, 84),
                    'av_mean_U_redder': np.mean(av_values_U_redder),
                    'av_median_U_redder': np.median(av_values_U_redder),
                    'av_std_U_redder': np.std(av_values_U_redder),
                    'av_16perc_U_redder': np.percentile(av_values_U_redder, 16),
                    'av_50perc_U_redder': np.percentile(av_values_U_redder, 50),
                    'av_84perc_U_redder': np.percentile(av_values_U_redder, 84),
                    'chi2_U_redder_mean': np.mean(chi2_U_redder_values)
                })
            
            # Cut band distribution statistics
            if best_fits_B_redder:
                teff_values_B_redder = [r['teff'] for r in best_fits_B_redder]
                logL_values_B_redder = [r['logL'] for r in best_fits_B_redder]
                logg_values_B_redder = [r['logg'] for r in best_fits_B_redder]
                av_values_B_redder = [r['av'] for r in best_fits_B_redder]
                chi2_B_redder_values = [r['chi2_B_redder'] for r in best_fits_B_redder]
                    
                summary.update({
                    'n_fits_B_redder': len(best_fits_B_redder),
                    'teff_mean_B_redder': np.mean(teff_values_B_redder),
                    'teff_median_B_redder': np.median(teff_values_B_redder),
                    'teff_std_B_redder': np.std(teff_values_B_redder),
                    'teff_16perc_B_redder': np.percentile(teff_values_B_redder, 16),
                    'teff_50perc_B_redder': np.percentile(teff_values_B_redder, 50),
                    'teff_84perc_B_redder': np.percentile(teff_values_B_redder, 84),
                    'logT_mean_B_redder': np.mean(np.log10(teff_values_B_redder)),
                    'logT_median_B_redder': np.median(np.log10(teff_values_B_redder)),
                    'logT_std_B_redder': np.std(np.log10(teff_values_B_redder)),
                    'logL_mean_B_redder': np.mean(logL_values_B_redder),
                    'logL_median_B_redder': np.median(logL_values_B_redder),
                    'logL_std_B_redder': np.std(logL_values_B_redder),
                    'logL_16perc_B_redder': np.percentile(logL_values_B_redder, 16),
                    'logL_50perc_B_redder': np.percentile(logL_values_B_redder, 50),
                    'logL_84perc_B_redder': np.percentile(logL_values_B_redder, 84),
                    'logg_mean_B_redder': np.mean(logg_values_B_redder),
                    'logg_median_B_redder': np.median(logg_values_B_redder),
                    'logg_std_B_redder': np.std(logg_values_B_redder),
                    'logg_16perc_B_redder': np.percentile(logg_values_B_redder, 16),
                    'logg_50perc_B_redder': np.percentile(logg_values_B_redder, 50),
                    'logg_84perc_B_redder': np.percentile(logg_values_B_redder, 84),
                    'av_mean_B_redder': np.mean(av_values_B_redder),
                    'av_median_B_redder': np.median(av_values_B_redder),
                    'av_std_B_redder': np.std(av_values_B_redder),
                    'av_16perc_B_redder': np.percentile(av_values_B_redder, 16),
                    'av_50perc_B_redder': np.percentile(av_values_B_redder, 50),
                    'av_84perc_B_redder': np.percentile(av_values_B_redder, 84),
                    'chi2_B_redder_mean': np.mean(chi2_B_redder_values)
                })
            
            if best_fits_V_redder:
                teff_values_V_redder = [r['teff'] for r in best_fits_V_redder]
                logL_values_V_redder = [r['logL'] for r in best_fits_V_redder]
                logg_values_V_redder = [r['logg'] for r in best_fits_V_redder]
                av_values_V_redder = [r['av'] for r in best_fits_V_redder]
                chi2_V_redder_values = [r['chi2_V_redder'] for r in best_fits_V_redder]
                    
                summary.update({
                    'n_fits_V_redder': len(best_fits_V_redder),
                    'teff_mean_V_redder': np.mean(teff_values_V_redder),
                    'teff_median_V_redder': np.median(teff_values_V_redder),
                    'teff_std_V_redder': np.std(teff_values_V_redder),
                    'teff_16perc_V_redder': np.percentile(teff_values_V_redder, 16),
                    'teff_50perc_V_redder': np.percentile(teff_values_V_redder, 50),
                    'teff_84perc_V_redder': np.percentile(teff_values_V_redder, 84),
                    'logT_mean_V_redder': np.mean(np.log10(teff_values_V_redder)),
                    'logT_median_V_redder': np.median(np.log10(teff_values_V_redder)),
                    'logT_std_V_redder': np.std(np.log10(teff_values_V_redder)),
                    'logL_mean_V_redder': np.mean(logL_values_V_redder),
                    'logL_median_V_redder': np.median(logL_values_V_redder),
                    'logL_std_V_redder': np.std(logL_values_V_redder),
                    'logL_16perc_V_redder': np.percentile(logL_values_V_redder, 16),
                    'logL_50perc_V_redder': np.percentile(logL_values_V_redder, 50),
                    'logL_84perc_V_redder': np.percentile(logL_values_V_redder, 84),
                    'logg_mean_V_redder': np.mean(logg_values_V_redder),
                    'logg_median_V_redder': np.median(logg_values_V_redder),
                    'logg_std_V_redder': np.std(logg_values_V_redder),
                    'logg_16perc_V_redder': np.percentile(logg_values_V_redder, 16),
                    'logg_50perc_V_redder': np.percentile(logg_values_V_redder, 50),
                    'logg_84perc_V_redder': np.percentile(logg_values_V_redder, 84),
                    'av_mean_V_redder': np.mean(av_values_V_redder),
                    'av_median_V_redder': np.median(av_values_V_redder),
                    'av_std_V_redder': np.std(av_values_V_redder),
                    'av_16perc_V_redder': np.percentile(av_values_V_redder, 16),
                    'av_50perc_V_redder': np.percentile(av_values_V_redder, 50),
                    'av_84perc_V_redder': np.percentile(av_values_V_redder, 84),
                    'chi2_V_redder_mean': np.mean(chi2_V_redder_values)
                })
            
            # Debug: Log some statistics for verification 
            has_U_redder = 'teff_U_redder' in summary
            has_B_redder = 'teff_B_redder' in summary
            has_V_redder = 'teff_V_redder' in summary
            n_U_redder_saved = len(best_fits_U_redder) if best_fits_U_redder else 0
            n_B_redder_saved = len(best_fits_B_redder) if best_fits_B_redder else 0
            n_V_redder_saved = len(best_fits_V_redder) if best_fits_V_redder else 0
            
            if has_U_redder and has_B_redder and has_V_redder:
                logger.info(f"Star {star_idx}: Saved {n_U_redder_saved} U_redder fits, {n_B_redder_saved} B_redder fits, and {n_V_redder_saved} V_redder fits. Best: U_redder Teff={summary['teff_U_redder']}K, B_redder Teff={summary['teff_B_redder']}K, V_redder Teff={summary['teff_V_redder']}K")
            elif has_U_redder:
                logger.info(f"Star {star_idx}: Saved {n_U_redder_saved} U_redder fits only. Best Teff={summary['teff_U_redder']}K")
            elif has_B_redder:
                logger.info(f"Star {star_idx}: Saved {n_B_redder_saved} B_redder fits only. Best Teff={summary['teff_B_redder']}K")
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
        summary_filename = f'ysg_temp_fitting_summary_v3.csv'
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