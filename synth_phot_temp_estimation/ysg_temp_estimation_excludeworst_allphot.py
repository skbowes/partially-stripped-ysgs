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
    coords = pd.read_csv('../merged_smc_lmc_coords.csv', sep=r'\s+', comment='#', names=['ra', 'dec'])
    df_lmc = pd.read_csv('../annas_candidates/final_lmc_ysgcands_allphot.csv', comment='#') # , sep="\\s+"
    df_smc = pd.read_csv('../annas_candidates/final_smc_ysgcands_allphot.csv', comment='#') # , sep="\\s+"
    choose_phot = pd.read_csv('choose_photometry_v2.csv')
    # Load synthetic photometry models
    computed_models = pd.read_csv('synth_phot_all_models_allphot_gordon.csv')
    return coords, df_smc, df_lmc, choose_phot, computed_models


# def rchi2_with_err(star_mags,star_err,model_mags):
#     '''
#     Returns the reduced chi^2, accounting for errors
#     Parameters:
#         star_mags: Observed magnitudes
#         star_err: Uncertainty on the observed magnitudes
#         model_mags: Model magnitudes
#     Returns:
#         rchi2: Reduced chi^2 value
#     '''
#     N = len(star_mags)
#     z = (star_mags-model_mags)/star_err
#     rchi2 = np.sum(z**2)/(N-1)
#     return rchi2

def observed_sed_allphot(index, coords, df_smc, df_lmc, choose_phot, flux=True, show=False):
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
    'Jmag_2MASS':3.0596e-10,    # J-band 3.0596e-10 # formerly was 1.11933e-9 
    'Hmag_2MASS':1.11064e-10,    # H-band 1.11064e-10 # formerly was 3.09069e-10 
    'Kmag_2MASS':4.17999e-11,     # K-band 4.17999e-11 # formerly was 4.20615e-11
    # Optical (MCPS)
    'Umag_MCPS':4.08739e-9,    # U-band
    'Bmag_MCPS':6.21086e-9,    # B-band
    'Vmag_MCPS':3.64047e-9,    # V-band
    'Imag_MCPS':9.23651e-10,    # I-band
    # Optical (APASS)
    'Bmag_APASS':6.72553e-9,
    'Vmag_APASS':3.636e-9,
    'gmag_APASS':4.92255e-9,
    'rmag_APASS':2.85425e-9,
    'imag_APASS':1.94038e-9,
    # UV (Swift UVOT)
    'uvw1mag_SWIFT':4.02204e-9,  # UVW
    'uvw2mag_SWIFT':5.37469e-9,   # UVW2
    'uvm2mag_SWIFT':4.66117e-9   # UVM2
    }


    #####  lowercase bands are AB, uppercase are Vega, so be careful with the zeropoints! we will convert the ab mags (g, r, i) from
    ##### APASS to be in VEGA since the rest of the bands are. we would also have to do this for smash. #####
    band_AB_zeropoints = {'gmag_APASS': 4.92255e-9, # ab, g
                            'rmag_APASS': 2.85425e-9, # ab, r
                            'imag_APASS': 1.94038e-9 # ab, i
                            }

    # Effective wavelengths (in Angstroms)
    band_wavelengths = {
        'uvw2mag_SWIFT': 2075.69,    # UV
        'uvm2mag_SWIFT': 2246.56,    # UV
        'uvw1mag_SWIFT': 2715.68,    # UV
        'Umag_MCPS': 3706.29,        # U
        'Bmag_MCPS': 4394.48,        # B
        'Vmag_MCPS': 5438.23,        # V
        'Imag_MCPS': 8568.89,        # I
        'Jmag_2MASS': 12350.00,       # J
        'Hmag_2MASS': 16620.00,       # H
        'Kmag_2MASS': 21590.00,        # K
        'Bmag_APASS': 4369.53,
        'Vmag_APASS': 5467.57,
        'gmag_APASS': 4671.78,
        'rmag_APASS': 6141.12,
        'imag_APASS':7457.89
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
            mag = df[band] # grab the magnitude, but for APASS we will convert to flux and back to mag to ensure consistency with zeropoints and error handling
            error_col = f'e_{band}'
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

            # Convert magnitude to flux density
            flux_ergs = band_zeropoints[band] * 10**(-0.4 * mag)
            flux_err_ergs = flux_ergs * 0.921 * mag_err

            # Get magnitude and error by converting back from flux, to ensure consistency
            if band in band_AB_zeropoints:
                # For AB magnitudes, use the AB zero point
                mag = -2.5 * np.log10(flux_ergs / band_AB_zeropoints[band])
                if error_col in df.index:
                    mag_err = (2.5 / np.log(10)) * (flux_err_ergs / flux_ergs)  # Convert flux error to magnitude error
            else:
                mag = df[band]
            
            wavelengths.append(band_wavelengths[band])
            fluxes.append(flux_ergs)
            flux_errors.append(flux_err_ergs)
            mags.append(mag)
            mag_errors.append(mag_err)
            band_names.append(band)
    
    # Sort by longest to shortest wavelength
    sorted_indices = np.argsort(wavelengths)[::-1]
    wavelengths = np.array(wavelengths)[sorted_indices]
    fluxes = np.array(fluxes)[sorted_indices]
    flux_errors = np.array(flux_errors)[sorted_indices]
    mags = np.array(mags)[sorted_indices]
    mag_errors = np.array(mag_errors)[sorted_indices]
    # didnt have this line before:
    band_names = np.array(band_names)[sorted_indices]

    return wavelengths, fluxes, flux_errors, mags, mag_errors, band_names

def process_star_chunk_vectorized(star_indices_chunk, choose_phot, computed_models, coords, df_smc, df_lmc, iterations=1000):
    """
    Process a chunk of stars using vectorized calculations.
    This function runs in a separate process.
    Returns both detailed results and summary statistics.
    """
    
    logger = logging.getLogger(f'worker_{star_indices_chunk[0]}')
    logger.info(f"Worker started: processing stars {star_indices_chunk[0]}-{star_indices_chunk[-1]}")
    min_max_avs = pd.read_csv('ysg_candidate_extinctions.csv')

    standard_band_order = ['Kmag_2MASS', 'Hmag_2MASS', 'Jmag_2MASS', 'Umag_MCPS', 'Bmag_APASS', 'Bmag_MCPS', 'gmag_APASS', 'Vmag_MCPS', 'Vmag_APASS', 'rmag_APASS', 'imag_APASS', 'uvm2mag_SWIFT', 'uvw1mag_SWIFT', 'uvw2mag_SWIFT']
    V_redder_bands = ['Kmag_2MASS', 'Hmag_2MASS', 'Jmag_2MASS', 'Imag_MCPS', 'Vmag_MCPS', 'Vmag_APASS']
    B_redder_bands = ['Kmag_2MASS', 'Hmag_2MASS', 'Jmag_2MASS', 'Imag_MCPS', 'Vmag_MCPS', 'Vmag_APASS', 'Bmag_APASS', 'Bmag_MCPS']
    U_redder_bands = ['Kmag_2MASS', 'Hmag_2MASS', 'Jmag_2MASS', 'Imag_MCPS', 'Vmag_MCPS', 'Vmag_APASS', 'Bmag_APASS', 'Bmag_MCPS', 'Umag_MCPS']
    
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
            obs = observed_sed_allphot(star_idx, coords, df_smc, df_lmc, choose_phot, show=False)
            obs_wavelengths, obs_fluxes, obs_flux_errors, obs_mags, obs_mag_errors, obs_band_names = obs

            choose_phot_options = ['Kmag_2MASS', 'Hmag_2MASS', 'Jmag_2MASS']
            choose_phot_row = choose_phot[(choose_phot['RA'] == RA) & (choose_phot['DEC'] == dec)].iloc[0]
            if choose_phot_row['choose_MCPS'] == 1:
                choose_phot_options.extend(['Umag_MCPS', 'Bmag_MCPS', 'Vmag_MCPS', 'Imag_MCPS'])
                if choose_phot_row['choose_APASS'] == 1:
                    choose_phot_options.extend(['Bmag_APASS', 'Vmag_APASS', 'gmag_APASS', 'rmag_APASS', 'imag_APASS'])
                if choose_phot_row['choose_APASS'] == 0 and 'Bmag_MCPS' in obs_band_names:
                    if pd.isna(obs_mags[list(obs_band_names).index('Bmag_MCPS')]):
                        #could add a print statement here
                        choose_phot_options.append('Bmag_APASS')
                if choose_phot_row['choose_APASS'] == 0 and 'Vmag_MCPS' in obs_band_names: 
                    if pd.isna(obs_mags[list(obs_band_names).index('Vmag_MCPS')]):
                        #could add a print statement here
                        choose_phot_options.append('Vmag_APASS')
            elif choose_phot_row['choose_APASS'] == 1:
                choose_phot_options.extend(['Bmag_APASS', 'Vmag_APASS', 'gmag_APASS', 'rmag_APASS', 'imag_APASS'])
                 # Add MCPS U, since APASS doesn't have U and we want to keep it if APASS is chosen
                choose_phot_options.append('Umag_MCPS')
                if choose_phot_row['choose_MCPS'] == 0 and 'Bmag_APASS' not in obs_band_names:
                    choose_phot_options.append('Bmag_MCPS')
                if choose_phot_row['choose_MCPS'] == 0 and 'Vmag_APASS' not in obs_band_names: 
                    choose_phot_options.append('Vmag_MCPS')

            # Create dictionaries for observed data
            obs_mags_dict = dict(zip(obs_band_names, obs_mags))
            obs_errors_dict = dict(zip(obs_band_names, obs_mag_errors))
            
            # Use standard band order that matches synthetic models
            # common_bands = [band for band in standard_band_order if band in obs_band_names]
            ########### THIS IS WHERE THE CHOSEN PHOTOMETRY IS REALLY FILTERED OUT, THRU COMMON_BANDS ############
            common_bands = [band for band in choose_phot_options if band in obs_band_names]
            min_av = min_max_avs[(min_max_avs['RA'] == RA) & (min_max_avs['DEC'] == dec)]['av_eden'].values[0]
            max_av = min_max_avs[(min_max_avs['RA'] == RA) & (min_max_avs['DEC'] == dec)]['av_sf'].values[0]
            max_av = np.minimum(max_av, 1.0)  # limit to 1
            zh_av = min_max_avs[(min_max_avs['RA'] == RA) & (min_max_avs['DEC'] == dec)]['av_zh'].values[0]
            
            # Extract matched data arrays
            matched_obs_mags = np.array([obs_mags_dict[band] for band in common_bands])
            matched_obs_errors = np.array([obs_errors_dict[band] for band in common_bands])
            
            # Filter models by metallicity
            if star_idx < 377:
                models_to_test = computed_models[computed_models['metallicity'] == -0.75]
            else:
                models_to_test = computed_models[computed_models['metallicity'] == -0.25]


            # VECTORIZED CALCULATIONS
            
            # Pre-extract all model data, only take models with av >= min_av and av <= max_av
            av_mask = (models_to_test['av'] >= min_av) & (models_to_test['av'] <= max_av)
            if np.sum(av_mask) == 0:
                if zh_av is not np.nan and (zh_av >= min_av + 0.05):
                    av_mask = (models_to_test['av'] >= min_av) & (models_to_test['av'] <= zh_av)
                else:
                    av_mask = (models_to_test['av'] >= min_av) & (models_to_test['av'] <= 1.0)
            # else:
            #         av_mask = (models_to_test['av'] >= min_av) & (models_to_test['av'] <= 1.0)

            filtered_models = models_to_test[av_mask]
            n_models = len(filtered_models)
            all_model_mags = np.array([filtered_models[band].values for band in common_bands]).T # each row is a model, each column a band
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
            ref_idx = common_bands.index('Kmag_2MASS')
            k_idx = common_bands.index('Kmag_2MASS')

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
                        # Identify worst matching band for each model
                        # residuals = modified_obs_mags[V_redder_indices][np.newaxis, :] - model_mags_shifted[:, V_redder_indices]
                        # # Create array of indices of worst fitting points
                        # worst_match_idx = np.argmax(np.abs(residuals), axis=1)
                        # # Could exclude worst match here if desired:
                        # chi2_V_redder_all_excludeworst = chi2_V_redder_all - (residuals[np.arange(n_models), worst_match_idx]**2) / (errors_V_redder[worst_match_idx]**2)
                        # exclude_mask = chi2_V_redder_all_excludeworst < (1/10) * chi2_V_redder_all
                        # # Apply exclusion only where beneficial
                        # chi2_V_redder_all = np.where(exclude_mask, chi2_V_redder_all_excludeworst, chi2_V_redder_all)
                        # # report how many models used exclusion
                        # # if np.any(exclude_mask):
                        # #     print(f"Excluded worst band from chi2 for {np.sum(exclude_mask)} out of {n_models} models")

                    
                    # Find best fits for this iteration/K combination (EFFICIENT)
                    best_U_redder_idx = np.argmin(chi2_U_redder_all)
                    best_U_redder_result_K = {
                        'iteration': it, 'K_variation': K, 'teff': model_teffs[best_U_redder_idx],
                        'logg': model_loggs[best_U_redder_idx], 'av': model_avs[best_U_redder_idx],
                        'logL': np.log10(model_lum_unscaled[best_U_redder_idx] * offsets[best_U_redder_idx] / 3.826e33),
                        'chi2_U_redder': chi2_U_redder_all[best_U_redder_idx],
                        'chi2_B_redder': chi2_B_redder_all[best_U_redder_idx], #if len(B_redder_indices) > 0 else np.nan,
                        'chi2_V_redder': chi2_V_redder_all[best_U_redder_idx], #if len(V_redder_indices) > 0 else np.nan,
                        'model_filename': model_filenames[best_U_redder_idx],
                        'excluded_band': np.nan
                    }
                    
                    # # COMMENTED OUT: Worst-band exclusion moved to after best K selection
                    # # Check if excluding worst non-U band improves U_redder fit by factor of 100
                    # if len(U_redder_indices) > 3:  # Need at least 3 bands to exclude one (keeping U + 2 others)
                    #     residuals_best_model = modified_obs_mags[U_redder_indices] - model_mags_shifted[best_U_redder_idx, U_redder_indices]
                    #     # Find U band position in U_redder_indices
                    #     U_band_pos = None
                    #     for pos, idx in enumerate(U_redder_indices):
                    #         if common_bands[idx] == 'Umag_MCPS':
                    #             U_band_pos = pos
                    #             break
                    #     
                    #     if U_band_pos is not None:
                    #         # Get indices of non-U bands
                    #         non_U_positions = [pos for pos in range(len(U_redder_indices)) if pos != U_band_pos]
                    #         if len(non_U_positions) > 1:  # Need at least 2 non-U bands to exclude one
                    #             residuals_non_U = np.abs(residuals_best_model[non_U_positions])
                    #             worst_band_pos_in_non_U = np.argmax(residuals_non_U)
                    #             worst_band_pos = non_U_positions[worst_band_pos_in_non_U]
                    #             worst_band_idx = U_redder_indices[worst_band_pos]
                    #             
                    #             # Calculate improvement if we exclude this band
                    #             chi2_original = chi2_U_redder_all[best_U_redder_idx]
                    #             chi2_exclude_worst = chi2_original - (residuals_best_model[worst_band_pos]**2) / (matched_obs_errors[worst_band_idx]**2)
                    #             
                    #             if chi2_exclude_worst < chi2_original / 10:  # Factor of 100 improvement
                    #                 # Redo fitting excluding the worst band
                    #                 U_redder_indices_excluded = np.delete(U_redder_indices, worst_band_pos)
                    #                 
                    #                 # Recalculate chi2 for all models excluding worst band
                    #                 diff_squared_U_redder_excl = diff_squared[:, U_redder_indices_excluded]
                    #                 errors_U_redder_excl = matched_obs_errors[U_redder_indices_excluded]
                    #                 chi2_U_redder_all_excl = np.sum(diff_squared_U_redder_excl / (errors_U_redder_excl[np.newaxis, :]**2), axis=1)
                    #                 
                    #                 # Find new best fit
                    #                 best_U_redder_idx_excl = np.nanargmin(chi2_U_redder_all_excl)
                    #                 
                    #                 # Update best result if this is better
                    #                 if chi2_U_redder_all_excl[best_U_redder_idx_excl] < chi2_original:
                    #                     best_U_redder_result_K = {
                    #                         'iteration': it, 'K_variation': K, 'teff': model_teffs[best_U_redder_idx_excl],
                    #                         'logg': model_loggs[best_U_redder_idx_excl], 'av': model_avs[best_U_redder_idx_excl],
                    #                         'logL': np.log10(model_lum_unscaled[best_U_redder_idx_excl] * offsets[best_U_redder_idx_excl] / 3.826e33),
                    #                         'chi2_U_redder': chi2_U_redder_all_excl[best_U_redder_idx_excl],
                    #                         'chi2_B_redder': chi2_B_redder_all[best_U_redder_idx_excl],
                    #                         'chi2_V_redder': chi2_V_redder_all[best_U_redder_idx_excl],
                    #                         'model_filename': model_filenames[best_U_redder_idx_excl],
                    #                         'excluded_band': common_bands[worst_band_idx]
                    #                     }
                    
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
                        'model_filename': model_filenames[best_B_redder_idx],
                        'excluded_band': np.nan
                    }
                    
                    # # COMMENTED OUT: Worst-band exclusion moved to after best K selection
                    # # Check if excluding worst non-B band improves B_redder fit by factor of 100
                    # if len(B_redder_indices) > 2:  # Need at least 3 bands to exclude one (keeping B + 1 other)
                    #     residuals_best_model = modified_obs_mags[B_redder_indices] - model_mags_shifted[best_B_redder_idx, B_redder_indices]
                    #     # Find B band position in B_redder_indices
                    #     B_band_pos = None
                    #     for pos, idx in enumerate(B_redder_indices):
                    #         if common_bands[idx] == 'Bmag_MCPS' or common_bands[idx] == 'Bmag_APASS':
                    #             B_band_pos = pos
                    #             break
                    #     
                    #     if B_band_pos is not None:
                    #         # Get indices of non-B bands
                    #         non_B_positions = [pos for pos in range(len(B_redder_indices)) if pos != B_band_pos]
                    #         if len(non_B_positions) > 1:  # Need at least 2 non-B bands to exclude one
                    #             residuals_non_B = np.abs(residuals_best_model[non_B_positions])
                    #             worst_band_pos_in_non_B = np.argmax(residuals_non_B)
                    #             worst_band_pos = non_B_positions[worst_band_pos_in_non_B]
                    #             worst_band_idx = B_redder_indices[worst_band_pos]
                    #             
                    #             # Calculate improvement if we exclude this band
                    #             chi2_original = chi2_B_redder_all[best_B_redder_idx]
                    #             chi2_exclude_worst = chi2_original - (residuals_best_model[worst_band_pos]**2) / (matched_obs_errors[worst_band_idx]**2)
                    #             
                    #             if chi2_exclude_worst < chi2_original / 10:  # Factor of 100 improvement
                    #                 # Redo fitting excluding the worst band
                    #                 B_redder_indices_excluded = np.delete(B_redder_indices, worst_band_pos)
                    #                 
                    #                 # Recalculate chi2 for all models excluding worst band
                    #                 diff_squared_B_redder_excl = diff_squared[:, B_redder_indices_excluded]
                    #                 errors_B_redder_excl = matched_obs_errors[B_redder_indices_excluded]
                    #                 chi2_B_redder_all_excl = np.sum(diff_squared_B_redder_excl / (errors_B_redder_excl[np.newaxis, :]**2), axis=1)
                    #                 
                    #                 # Find new best fit
                    #                 best_B_redder_idx_excl = np.nanargmin(chi2_B_redder_all_excl)
                    #                 
                    #                 # Update best result if this is better
                    #                 if chi2_B_redder_all_excl[best_B_redder_idx_excl] < chi2_original:
                    #                     best_B_redder_result_K = {
                    #                         'iteration': it, 'K_variation': K, 'teff': model_teffs[best_B_redder_idx_excl],
                    #                         'logg': model_loggs[best_B_redder_idx_excl], 'av': model_avs[best_B_redder_idx_excl],
                    #                         'logL': np.log10(model_lum_unscaled[best_B_redder_idx_excl] * offsets[best_B_redder_idx_excl] / 3.826e33),
                    #                         'chi2_U_redder': chi2_U_redder_all[best_B_redder_idx_excl],
                    #                         'chi2_B_redder': chi2_B_redder_all_excl[best_B_redder_idx_excl],
                    #                         'chi2_V_redder': chi2_V_redder_all[best_B_redder_idx_excl],
                    #                         'model_filename': model_filenames[best_B_redder_idx_excl],
                    #                         'excluded_band': common_bands[worst_band_idx]
                    #                     }
                    
                    best_fits_B_redder_K.append(best_B_redder_result_K)

                    best_V_redder_idx = np.nanargmin(chi2_V_redder_all)
                    best_V_redder_result_K = {
                        'iteration': it, 'K_variation': K, 'teff': model_teffs[best_V_redder_idx],
                        'logg': model_loggs[best_V_redder_idx], 'av': model_avs[best_V_redder_idx],
                        'logL': np.log10(model_lum_unscaled[best_V_redder_idx] * offsets[best_V_redder_idx] / 3.826e33),
                        'chi2_U_redder': chi2_U_redder_all[best_V_redder_idx],
                        'chi2_B_redder': chi2_B_redder_all[best_V_redder_idx],
                        'chi2_V_redder': chi2_V_redder_all[best_V_redder_idx],
                        'model_filename': model_filenames[best_V_redder_idx],
                        'excluded_band': np.nan
                    }
                    
                    # # COMMENTED OUT: Worst-band exclusion moved to after best K selection
                    # # Check if excluding worst non-V band improves V_redder fit by factor of 100
                    # if len(V_redder_indices) > 3:  # Need at least 3 bands to exclude one (keeping V + 2 others)
                    #     residuals_best_model = modified_obs_mags[V_redder_indices] - model_mags_shifted[best_V_redder_idx, V_redder_indices]
                    #     # Find V band position in V_redder_indices
                    #     V_band_pos = None
                    #     for pos, idx in enumerate(V_redder_indices):
                    #         if common_bands[idx] == 'Vmag_MCPS' or common_bands[idx] == 'Vmag_APASS':
                    #             V_band_pos = pos
                    #             break
                    #     
                    #     if V_band_pos is not None:
                    #         # Get indices of non-V bands
                    #         non_V_positions = [pos for pos in range(len(V_redder_indices)) if pos != V_band_pos]
                    #         if len(non_V_positions) > 1:  # Need at least 2 non-V bands to exclude one
                    #             residuals_non_V = np.abs(residuals_best_model[non_V_positions])
                    #             worst_band_pos_in_non_V = np.argmax(residuals_non_V)
                    #             worst_band_pos = non_V_positions[worst_band_pos_in_non_V]
                    #             worst_band_idx = V_redder_indices[worst_band_pos]
                    #             
                    #             # Calculate improvement if we exclude this band
                    #             chi2_original = chi2_V_redder_all[best_V_redder_idx]
                    #             chi2_exclude_worst = chi2_original - (residuals_best_model[worst_band_pos]**2) / (matched_obs_errors[worst_band_idx]**2)
                    #             
                    #             if chi2_exclude_worst < chi2_original / 10:  # Factor of 100 improvement
                    #                 # Redo fitting excluding the worst band
                    #                 V_redder_indices_excluded = np.delete(V_redder_indices, worst_band_pos)
                    #                 
                    #                 # Recalculate chi2 for all models excluding worst band
                    #                 diff_squared_V_redder_excl = diff_squared[:, V_redder_indices_excluded]
                    #                 errors_V_redder_excl = matched_obs_errors[V_redder_indices_excluded]
                    #                 chi2_V_redder_all_excl = np.sum(diff_squared_V_redder_excl / (errors_V_redder_excl[np.newaxis, :]**2), axis=1)
                    #                 
                    #                 # Find new best fit
                    #                 best_V_redder_idx_excl = np.nanargmin(chi2_V_redder_all_excl)
                    #                 
                    #                 # Update best result if this is better
                    #                 if chi2_V_redder_all_excl[best_V_redder_idx_excl] < chi2_original:
                    #                     best_V_redder_result_K = {
                    #                         'iteration': it, 'K_variation': K, 'teff': model_teffs[best_V_redder_idx_excl],
                    #                         'logg': model_loggs[best_V_redder_idx_excl], 'av': model_avs[best_V_redder_idx_excl],
                    #                         'logL': np.log10(model_lum_unscaled[best_V_redder_idx_excl] * offsets[best_V_redder_idx_excl] / 3.826e33),
                    #                         'chi2_U_redder': chi2_U_redder_all[best_V_redder_idx_excl],
                    #                         'chi2_B_redder': chi2_B_redder_all[best_V_redder_idx_excl],
                    #                         'chi2_V_redder': chi2_V_redder_all_excl[best_V_redder_idx_excl],
                    #                         'model_filename': model_filenames[best_V_redder_idx_excl],
                    #                         'excluded_band': common_bands[worst_band_idx]
                    #                     }
                    
                    best_fits_V_redder_K.append(best_V_redder_result_K)
                

                # Here is where I'll store the best results from iterating on K:
                # take best among K variations for this iteration
                chi2_U_redder_values_K = [r['chi2_U_redder'] for r in best_fits_U_redder_K]
                best_U_redder_K_idx = np.argmin(chi2_U_redder_values_K)
                best_U_redder_result = best_fits_U_redder_K[best_U_redder_K_idx]

                chi2_B_redder_values_K = [r['chi2_B_redder'] for r in best_fits_B_redder_K]
                best_B_redder_K_idx = np.nanargmin(chi2_B_redder_values_K)
                best_B_redder_result = best_fits_B_redder_K[best_B_redder_K_idx]

                chi2_V_redder_values_K = [r['chi2_V_redder'] for r in best_fits_V_redder_K]
                best_V_redder_K_idx = np.nanargmin(chi2_V_redder_values_K)
                best_V_redder_result = best_fits_V_redder_K[best_V_redder_K_idx]

                # NOW CHECK WORST BAND EXCLUSION FOR BEST K RESULTS
                # For each redder type, recalculate with the best K value and check if excluding worst band helps
                
                # U_redder worst band exclusion
                best_K_U = best_U_redder_result['K_variation']
                modified_obs_mags_U = sampled_obs_mags.copy()
                modified_obs_mags_U[k_idx] = modified_obs_mags_U[k_idx] + best_K_U * matched_obs_errors[k_idx]
                
                # Recalculate model shifts and chi2 for best K
                mag_shifts_U = modified_obs_mags_U[ref_idx] - all_model_mags[:, ref_idx]
                model_mags_shifted_U = all_model_mags + mag_shifts_U[:, np.newaxis]
                offsets_U = 10**(-0.4 * modified_obs_mags_U[ref_idx]) / 10**(-0.4 * all_model_mags[:, ref_idx])
                diff_squared_U = (modified_obs_mags_U[np.newaxis, :] - model_mags_shifted_U)**2
                
                if len(U_redder_indices) > 3:
                    diff_squared_U_redder = diff_squared_U[:, U_redder_indices]
                    errors_U_redder = matched_obs_errors[U_redder_indices]
                    chi2_U_redder_all_U = np.sum(diff_squared_U_redder / (errors_U_redder[np.newaxis, :]**2), axis=1)
                    
                    # Find best model index for this result
                    best_U_model_idx = np.where((model_teffs == best_U_redder_result['teff']) & 
                                                 (model_loggs == best_U_redder_result['logg']) & 
                                                 (model_avs == best_U_redder_result['av']))[0][0]
                    
                    residuals_best_model = modified_obs_mags_U[U_redder_indices] - model_mags_shifted_U[best_U_model_idx, U_redder_indices]
                    U_band_pos = None
                    for pos, idx in enumerate(U_redder_indices):
                        if common_bands[idx] == 'Umag_MCPS':
                            U_band_pos = pos
                            break
                    
                    if U_band_pos is not None:
                        non_U_positions = [pos for pos in range(len(U_redder_indices)) if pos != U_band_pos]
                        if len(non_U_positions) > 1:
                            residuals_non_U = np.abs(residuals_best_model[non_U_positions])
                            worst_band_pos_in_non_U = np.argmax(residuals_non_U)
                            worst_band_pos = non_U_positions[worst_band_pos_in_non_U]
                            worst_band_idx = U_redder_indices[worst_band_pos]
                            
                            chi2_original = chi2_U_redder_all_U[best_U_model_idx]
                            chi2_exclude_worst = chi2_original - (residuals_best_model[worst_band_pos]**2) / (matched_obs_errors[worst_band_idx]**2)
                            
                            if chi2_exclude_worst < chi2_original / 5: # change factor here
                                U_redder_indices_excluded = np.delete(U_redder_indices, worst_band_pos)
                                diff_squared_U_redder_excl = diff_squared_U[:, U_redder_indices_excluded]
                                errors_U_redder_excl = matched_obs_errors[U_redder_indices_excluded]
                                chi2_U_redder_all_excl = np.sum(diff_squared_U_redder_excl / (errors_U_redder_excl[np.newaxis, :]**2), axis=1)
                                
                                best_U_redder_idx_excl = np.nanargmin(chi2_U_redder_all_excl)
                                
                                if chi2_U_redder_all_excl[best_U_redder_idx_excl] < chi2_original:
                                    # Recalculate other chi2 values for the new best model
                                    if len(B_redder_indices) > 0:
                                        diff_squared_B_redder_new = diff_squared_U[:, B_redder_indices]
                                        errors_B_redder_new = matched_obs_errors[B_redder_indices]
                                        chi2_B_redder_new = np.sum(diff_squared_B_redder_new[best_U_redder_idx_excl] / (errors_B_redder_new**2))
                                    else:
                                        chi2_B_redder_new = np.nan
                                    
                                    if len(V_redder_indices) > 0:
                                        diff_squared_V_redder_new = diff_squared_U[:, V_redder_indices]
                                        errors_V_redder_new = matched_obs_errors[V_redder_indices]
                                        chi2_V_redder_new = np.sum(diff_squared_V_redder_new[best_U_redder_idx_excl] / (errors_V_redder_new**2))
                                    else:
                                        chi2_V_redder_new = np.nan
                                    
                                    best_U_redder_result = {
                                        'iteration': it, 'K_variation': best_K_U, 'teff': model_teffs[best_U_redder_idx_excl],
                                        'logg': model_loggs[best_U_redder_idx_excl], 'av': model_avs[best_U_redder_idx_excl],
                                        'logL': np.log10(model_lum_unscaled[best_U_redder_idx_excl] * offsets_U[best_U_redder_idx_excl] / 3.826e33),
                                        'chi2_U_redder': chi2_U_redder_all_excl[best_U_redder_idx_excl],
                                        'chi2_B_redder': chi2_B_redder_new,
                                        'chi2_V_redder': chi2_V_redder_new,
                                        'model_filename': model_filenames[best_U_redder_idx_excl],
                                        'excluded_band': common_bands[worst_band_idx]
                                    }
                
                # B_redder worst band exclusion
                best_K_B = best_B_redder_result['K_variation']
                modified_obs_mags_B = sampled_obs_mags.copy()
                modified_obs_mags_B[k_idx] = modified_obs_mags_B[k_idx] + best_K_B * matched_obs_errors[k_idx]
                
                mag_shifts_B = modified_obs_mags_B[ref_idx] - all_model_mags[:, ref_idx]
                model_mags_shifted_B = all_model_mags + mag_shifts_B[:, np.newaxis]
                offsets_B = 10**(-0.4 * modified_obs_mags_B[ref_idx]) / 10**(-0.4 * all_model_mags[:, ref_idx])
                diff_squared_B = (modified_obs_mags_B[np.newaxis, :] - model_mags_shifted_B)**2
                
                if len(B_redder_indices) > 2:
                    diff_squared_B_redder = diff_squared_B[:, B_redder_indices]
                    errors_B_redder = matched_obs_errors[B_redder_indices]
                    chi2_B_redder_all_B = np.sum(diff_squared_B_redder / (errors_B_redder[np.newaxis, :]**2), axis=1)
                    
                    best_B_model_idx = np.where((model_teffs == best_B_redder_result['teff']) & 
                                                 (model_loggs == best_B_redder_result['logg']) & 
                                                 (model_avs == best_B_redder_result['av']))[0][0]
                    
                    residuals_best_model = modified_obs_mags_B[B_redder_indices] - model_mags_shifted_B[best_B_model_idx, B_redder_indices]
                    B_band_pos = None
                    for pos, idx in enumerate(B_redder_indices):
                        if common_bands[idx] == 'Bmag_MCPS' or common_bands[idx] == 'Bmag_APASS':
                            B_band_pos = pos
                            break
                    
                    if B_band_pos is not None:
                        non_B_positions = [pos for pos in range(len(B_redder_indices)) if pos != B_band_pos]
                        if len(non_B_positions) > 1:
                            residuals_non_B = np.abs(residuals_best_model[non_B_positions])
                            worst_band_pos_in_non_B = np.argmax(residuals_non_B)
                            worst_band_pos = non_B_positions[worst_band_pos_in_non_B]
                            worst_band_idx = B_redder_indices[worst_band_pos]
                            
                            chi2_original = chi2_B_redder_all_B[best_B_model_idx]
                            chi2_exclude_worst = chi2_original - (residuals_best_model[worst_band_pos]**2) / (matched_obs_errors[worst_band_idx]**2)
                            
                            if chi2_exclude_worst < chi2_original / 5: # change factor here
                                B_redder_indices_excluded = np.delete(B_redder_indices, worst_band_pos)
                                diff_squared_B_redder_excl = diff_squared_B[:, B_redder_indices_excluded]
                                errors_B_redder_excl = matched_obs_errors[B_redder_indices_excluded]
                                chi2_B_redder_all_excl = np.sum(diff_squared_B_redder_excl / (errors_B_redder_excl[np.newaxis, :]**2), axis=1)
                                
                                best_B_redder_idx_excl = np.nanargmin(chi2_B_redder_all_excl)
                                
                                if chi2_B_redder_all_excl[best_B_redder_idx_excl] < chi2_original:
                                    if len(U_redder_indices) > 0:
                                        diff_squared_U_redder_new = diff_squared_B[:, U_redder_indices]
                                        errors_U_redder_new = matched_obs_errors[U_redder_indices]
                                        chi2_U_redder_new = np.sum(diff_squared_U_redder_new[best_B_redder_idx_excl] / (errors_U_redder_new**2))
                                    else:
                                        chi2_U_redder_new = np.nan
                                    
                                    if len(V_redder_indices) > 0:
                                        diff_squared_V_redder_new = diff_squared_B[:, V_redder_indices]
                                        errors_V_redder_new = matched_obs_errors[V_redder_indices]
                                        chi2_V_redder_new = np.sum(diff_squared_V_redder_new[best_B_redder_idx_excl] / (errors_V_redder_new**2))
                                    else:
                                        chi2_V_redder_new = np.nan
                                    
                                    best_B_redder_result = {
                                        'iteration': it, 'K_variation': best_K_B, 'teff': model_teffs[best_B_redder_idx_excl],
                                        'logg': model_loggs[best_B_redder_idx_excl], 'av': model_avs[best_B_redder_idx_excl],
                                        'logL': np.log10(model_lum_unscaled[best_B_redder_idx_excl] * offsets_B[best_B_redder_idx_excl] / 3.826e33),
                                        'chi2_U_redder': chi2_U_redder_new,
                                        'chi2_B_redder': chi2_B_redder_all_excl[best_B_redder_idx_excl],
                                        'chi2_V_redder': chi2_V_redder_new,
                                        'model_filename': model_filenames[best_B_redder_idx_excl],
                                        'excluded_band': common_bands[worst_band_idx]
                                    }
                
                # V_redder worst band exclusion
                best_K_V = best_V_redder_result['K_variation']
                modified_obs_mags_V = sampled_obs_mags.copy()
                modified_obs_mags_V[k_idx] = modified_obs_mags_V[k_idx] + best_K_V * matched_obs_errors[k_idx]
                
                mag_shifts_V = modified_obs_mags_V[ref_idx] - all_model_mags[:, ref_idx]
                model_mags_shifted_V = all_model_mags + mag_shifts_V[:, np.newaxis]
                offsets_V = 10**(-0.4 * modified_obs_mags_V[ref_idx]) / 10**(-0.4 * all_model_mags[:, ref_idx])
                diff_squared_V = (modified_obs_mags_V[np.newaxis, :] - model_mags_shifted_V)**2
                
                if len(V_redder_indices) > 3:
                    diff_squared_V_redder = diff_squared_V[:, V_redder_indices]
                    errors_V_redder = matched_obs_errors[V_redder_indices]
                    chi2_V_redder_all_V = np.sum(diff_squared_V_redder / (errors_V_redder[np.newaxis, :]**2), axis=1)
                    
                    best_V_model_idx = np.where((model_teffs == best_V_redder_result['teff']) & 
                                                 (model_loggs == best_V_redder_result['logg']) & 
                                                 (model_avs == best_V_redder_result['av']))[0][0]
                    
                    residuals_best_model = modified_obs_mags_V[V_redder_indices] - model_mags_shifted_V[best_V_model_idx, V_redder_indices]
                    V_band_pos = None
                    for pos, idx in enumerate(V_redder_indices):
                        if common_bands[idx] == 'Vmag_MCPS' or common_bands[idx] == 'Vmag_APASS':
                            V_band_pos = pos
                            break
                    
                    if V_band_pos is not None:
                        non_V_positions = [pos for pos in range(len(V_redder_indices)) if pos != V_band_pos]
                        if len(non_V_positions) > 1:
                            residuals_non_V = np.abs(residuals_best_model[non_V_positions])
                            worst_band_pos_in_non_V = np.argmax(residuals_non_V)
                            worst_band_pos = non_V_positions[worst_band_pos_in_non_V]
                            worst_band_idx = V_redder_indices[worst_band_pos]
                            
                            chi2_original = chi2_V_redder_all_V[best_V_model_idx]
                            chi2_exclude_worst = chi2_original - (residuals_best_model[worst_band_pos]**2) / (matched_obs_errors[worst_band_idx]**2)
                            
                            if chi2_exclude_worst < chi2_original / 5: # change factor here
                                V_redder_indices_excluded = np.delete(V_redder_indices, worst_band_pos)
                                diff_squared_V_redder_excl = diff_squared_V[:, V_redder_indices_excluded]
                                errors_V_redder_excl = matched_obs_errors[V_redder_indices_excluded]
                                chi2_V_redder_all_excl = np.sum(diff_squared_V_redder_excl / (errors_V_redder_excl[np.newaxis, :]**2), axis=1)
                                
                                best_V_redder_idx_excl = np.nanargmin(chi2_V_redder_all_excl)
                                
                                if chi2_V_redder_all_excl[best_V_redder_idx_excl] < chi2_original:
                                    if len(U_redder_indices) > 0:
                                        diff_squared_U_redder_new = diff_squared_V[:, U_redder_indices]
                                        errors_U_redder_new = matched_obs_errors[U_redder_indices]
                                        chi2_U_redder_new = np.sum(diff_squared_U_redder_new[best_V_redder_idx_excl] / (errors_U_redder_new**2))
                                    else:
                                        chi2_U_redder_new = np.nan
                                    
                                    if len(B_redder_indices) > 0:
                                        diff_squared_B_redder_new = diff_squared_V[:, B_redder_indices]
                                        errors_B_redder_new = matched_obs_errors[B_redder_indices]
                                        chi2_B_redder_new = np.sum(diff_squared_B_redder_new[best_V_redder_idx_excl] / (errors_B_redder_new**2))
                                    else:
                                        chi2_B_redder_new = np.nan
                                    
                                    best_V_redder_result = {
                                        'iteration': it, 'K_variation': best_K_V, 'teff': model_teffs[best_V_redder_idx_excl],
                                        'logg': model_loggs[best_V_redder_idx_excl], 'av': model_avs[best_V_redder_idx_excl],
                                        'logL': np.log10(model_lum_unscaled[best_V_redder_idx_excl] * offsets_V[best_V_redder_idx_excl] / 3.826e33),
                                        'chi2_U_redder': chi2_U_redder_new,
                                        'chi2_B_redder': chi2_B_redder_new,
                                        'chi2_V_redder': chi2_V_redder_all_excl[best_V_redder_idx_excl],
                                        'model_filename': model_filenames[best_V_redder_idx_excl],
                                        'excluded_band': common_bands[worst_band_idx]
                                    }
                
                # Append the final (potentially updated) results
                best_fits_U_redder.append(best_U_redder_result)
                best_fits_B_redder.append(best_B_redder_result)
                best_fits_V_redder.append(best_V_redder_result)


                


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
                    'model_filename': fit['model_filename'],
                    'excluded_band': fit.get('excluded_band', np.nan)
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
                    'model_filename': fit['model_filename'],
                    'excluded_band': fit.get('excluded_band', np.nan)
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
                    'model_filename': fit['model_filename'],
                    'excluded_band': fit.get('excluded_band', np.nan)
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
                
                # Exclusion statistics for U_redder
                excluded_bands_U_redder = [r.get('excluded_band') for r in best_fits_U_redder if pd.notna(r.get('excluded_band'))]
                n_excluded_U_redder = len(excluded_bands_U_redder)
                most_common_excluded_U_redder = pd.Series(excluded_bands_U_redder).mode().iloc[0] if excluded_bands_U_redder else np.nan
                
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
                    'chi2_U_redder_mean': np.mean(chi2_U_redder_values),
                    'n_excluded_U_redder': n_excluded_U_redder,
                    'most_common_excluded_U_redder': most_common_excluded_U_redder
                })
            
            # Cut band distribution statistics
            if best_fits_B_redder:
                teff_values_B_redder = [r['teff'] for r in best_fits_B_redder]
                logL_values_B_redder = [r['logL'] for r in best_fits_B_redder]
                logg_values_B_redder = [r['logg'] for r in best_fits_B_redder]
                av_values_B_redder = [r['av'] for r in best_fits_B_redder]
                chi2_B_redder_values = [r['chi2_B_redder'] for r in best_fits_B_redder]
                
                # Exclusion statistics for B_redder
                excluded_bands_B_redder = [r.get('excluded_band') for r in best_fits_B_redder if pd.notna(r.get('excluded_band'))]
                n_excluded_B_redder = len(excluded_bands_B_redder)
                most_common_excluded_B_redder = pd.Series(excluded_bands_B_redder).mode().iloc[0] if excluded_bands_B_redder else np.nan
                    
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
                    'chi2_B_redder_mean': np.mean(chi2_B_redder_values),
                    'n_excluded_B_redder': n_excluded_B_redder,
                    'most_common_excluded_B_redder': most_common_excluded_B_redder
                })
            
            if best_fits_V_redder:
                teff_values_V_redder = [r['teff'] for r in best_fits_V_redder]
                logL_values_V_redder = [r['logL'] for r in best_fits_V_redder]
                logg_values_V_redder = [r['logg'] for r in best_fits_V_redder]
                av_values_V_redder = [r['av'] for r in best_fits_V_redder]
                chi2_V_redder_values = [r['chi2_V_redder'] for r in best_fits_V_redder]
                
                # Exclusion statistics for V_redder
                excluded_bands_V_redder = [r.get('excluded_band') for r in best_fits_V_redder if pd.notna(r.get('excluded_band'))]
                n_excluded_V_redder = len(excluded_bands_V_redder)
                most_common_excluded_V_redder = pd.Series(excluded_bands_V_redder).mode().iloc[0] if excluded_bands_V_redder else np.nan
                    
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
                    'chi2_V_redder_mean': np.mean(chi2_V_redder_values),
                    'n_excluded_V_redder': n_excluded_V_redder,
                    'most_common_excluded_V_redder': most_common_excluded_V_redder
                })
            
            # Overall exclusion statistics across all band sets
            all_excluded_bands = []
            # total_exclusions = 0
            if best_fits_U_redder:
                excluded_U = [r.get('excluded_band') for r in best_fits_U_redder if pd.notna(r.get('excluded_band'))]
                all_excluded_bands.extend(excluded_U)
                total_exclusions_U_redder = len(excluded_U)
            if best_fits_B_redder:
                excluded_B = [r.get('excluded_band') for r in best_fits_B_redder if pd.notna(r.get('excluded_band'))]
                all_excluded_bands.extend(excluded_B)
                total_exclusions_B_redder = len(excluded_B)
            if best_fits_V_redder:
                excluded_V = [r.get('excluded_band') for r in best_fits_V_redder if pd.notna(r.get('excluded_band'))]
                all_excluded_bands.extend(excluded_V)
                total_exclusions_V_redder = len(excluded_V)
            
            most_common_excluded_overall = pd.Series(all_excluded_bands).mode().iloc[0] if all_excluded_bands else np.nan
            
            summary.update({
                'total_exclusions_U_redder': total_exclusions_U_redder,
                'total_exclusions_B_redder': total_exclusions_B_redder,
                'total_exclusions_V_redder': total_exclusions_V_redder,
                'most_common_excluded_overall': most_common_excluded_overall,
                'bands_used_for_fitting': ','.join(choose_phot_options)  # Save which bands were used
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

def compute_ysgs_parallel(total_star_indices, coords, df_smc, df_lmc, choose_phot, computed_models, n_cores=4, iterations=1000):
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
            choose_phot=choose_phot,
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
        summary_filename = f'ysg_temp_fitting_summary_v8.csv'
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
        coords, df_smc, df_lmc, choose_phot, computed_models = load_data()
        
        # Run the parallel processing
        start_time = datetime.now()
        compute_ysgs_parallel(
            total_star_indices=args.stars,
            coords=coords,
            df_smc=df_smc,
            df_lmc=df_lmc,
            choose_phot=choose_phot,
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