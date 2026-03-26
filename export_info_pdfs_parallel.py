#!/usr/bin/env python3
"""Parallel export of one info PDF per star index.

This version appends an SED fitting figure by using code copied from the
notebook directly in this file (no notebook execution at runtime).
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Avoid GUI backends during batch plotting.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import ascii
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

_MODULE_DIR = Path(__file__).parent
_FIT_CONTEXT = None

# Notebook-style globals used by copied plotting functions.
coords = None
df_lmc = None
df_smc = None
df_lmc_prefinal = None
df_smc_prefinal = None
smc_vis_binaries = None
smc_opt_binaries = None
lmc_vis_binaries = None
lmc_opt_binaries = None
choose_surveys = None
computed_models = None
gaia_bprp = None
temp_stats = None
lmc_wave = None
lmc_AlamAv = None
smc_wave = None
smc_AlamAv = None
wave = None
tmass_j = None
tmass_h = None
tmass_k = None
mcps_U = None
mcps_B = None
mcps_V = None
mcps_I = None
swift_uvm1 = None
swift_uvw1 = None
swift_uvw2 = None
apass_B = None
apass_V = None
apass_g = None
apass_r = None
apass_i = None
gaia_U = None
gaia_B = None
gaia_V = None
gaia_I = None
smash_U = None
smash_G = None
smash_R = None
smash_I = None
smash_Z = None
band_wavelengths = None


def _initialize_fit_context():
    global _FIT_CONTEXT
    global coords, df_lmc, df_smc, df_lmc_prefinal, df_smc_prefinal
    global smc_vis_binaries, smc_opt_binaries, lmc_vis_binaries, lmc_opt_binaries
    global choose_surveys, computed_models, gaia_bprp, temp_stats
    global lmc_wave, lmc_AlamAv, smc_wave, smc_AlamAv, wave
    global tmass_j, tmass_h, tmass_k, mcps_U, mcps_B, mcps_V, mcps_I
    global swift_uvm1, swift_uvw1, swift_uvw2
    global apass_B, apass_V, apass_g, apass_r, apass_i
    global gaia_U, gaia_B, gaia_V, gaia_I
    global smash_U, smash_G, smash_R, smash_I, smash_Z
    global band_wavelengths

    if _FIT_CONTEXT is not None:
        return _FIT_CONTEXT

    synth_dir = _MODULE_DIR / "synth_phot_temp_estimation"

    ctx = {
        "coords": pd.read_csv(_MODULE_DIR / "merged_smc_lmc_coords_all.csv", comment="#", sep="\\s+", names=["RA", "DEC"]),
        "df_lmc": pd.read_csv(_MODULE_DIR / "ysg_candidates/final_lmc_ysgcands_allphot_simbad.csv", comment="#"),
        "df_smc": pd.read_csv(_MODULE_DIR / "ysg_candidates/final_smc_ysgcands_allphot_simbad.csv", comment="#"),
        "df_lmc_prefinal": pd.read_csv(_MODULE_DIR / "ysg_candidates/prefinal_lmc_ysgcands_allphot_simbad.csv", comment="#"),
        "df_smc_prefinal": pd.read_csv(_MODULE_DIR / "ysg_candidates/prefinal_smc_ysgcands_allphot_simbad.csv", comment="#"),
        "smc_vis_binaries": pd.read_csv(_MODULE_DIR / "ysg_candidates/original_files_from_anna/smc_vis_bin.csv", comment="#"),
        "smc_opt_binaries": pd.read_csv(_MODULE_DIR / "ysg_candidates/original_files_from_anna/smc_opt_bin.csv", comment="#"),
        "lmc_vis_binaries": pd.read_csv(_MODULE_DIR / "ysg_candidates/original_files_from_anna/lmc_vis_bin.csv", comment="#"),
        "lmc_opt_binaries": pd.read_csv(_MODULE_DIR / "ysg_candidates/original_files_from_anna/lmc_opt_bin.csv", comment="#"),
        "choose_surveys": pd.read_csv(synth_dir / "choose_surveys_v4.csv", comment="#"),
        "computed_models": pd.read_csv(synth_dir / "synth_phot_all_models_allphot_gordon_gaia.csv"),
        "gaia_bprp": pd.read_csv(synth_dir / "gaia_bprp_synthphot_allbands.csv", sep=";", comment="#"),
        "temp_stats": pd.read_csv(synth_dir / "ysg_temp_fitting_summary_03152026.csv"),
    }

    lmc_gordon = ascii.read(
        synth_dir / "dustmaps_data/extinction_curves/lmc_avg_ext.dat",
        comment=";",
        names=["x", "AlamAv", "unc"],
    )
    smc_gordon = ascii.read(
        synth_dir / "dustmaps_data/extinction_curves/smc_bar_avg.dat",
        comment=";",
        names=["x", "AlamAv", "unc"],
    )

    ctx["lmc_wave"] = np.flip((1.0 / lmc_gordon["x"]) * 1e4)
    ctx["lmc_AlamAv"] = np.flip(lmc_gordon["AlamAv"])
    ctx["smc_wave"] = np.flip((1.0 / smc_gordon["x"]) * 1e4)
    ctx["smc_AlamAv"] = np.flip(smc_gordon["AlamAv"])

    ctx["wave"] = ascii.read(
        synth_dir / "pysynphot_data/grid/bosz/r2000/bosz2024_wave_r2000.txt",
        names=["wave"],
        data_start=0,
    )

    col_names = ["lam", "flux"]
    filter_paths = {
        "tmass_j": "pysynphot_data/grid/bosz/2MASS_2MASS.J_v3.0596e-10_ab7.08741e-10_eff12350.dat",
        "tmass_h": "pysynphot_data/grid/bosz/2MASS_2MASS.H_v1.11064e-10_ab4.00078e-10_eff16620.dat",
        "tmass_k": "pysynphot_data/grid/bosz/2MASS_2MASS.Ks_v4.17999e-11_ab2.32482e-10_eff21590.dat",
        "mcps_U": "pysynphot_data/grid/bosz/Misc_MCPS.U_v4.08739e-9_ab8.23894e-9_eff3706.dat",
        "mcps_B": "pysynphot_data/grid/bosz/Misc_MCPS.B_v6.21086e-9_ab5.60999e-9_eff4394.dat",
        "mcps_V": "pysynphot_data/grid/bosz/Misc_MCPS.V_v3.64047e-9_ab3.63812e-9_eff5438.dat",
        "mcps_I": "pysynphot_data/grid/bosz/Misc_MCPS.I_v9.23651e-10_ab1.45234e-9_eff8568.dat",
        "swift_uvm1": "pysynphot_data/grid/bosz/Swift_UVOT.UVM2_trn_v4.66117e-9_ab2.15291e-8_eff2246.dat",
        "swift_uvw1": "pysynphot_data/grid/bosz/Swift_UVOT.UVW1_trn_v4.02204e-9_ab1.57569e-8_eff2715.dat",
        "swift_uvw2": "pysynphot_data/grid/bosz/Swift_UVOT.UVW2_trn_v5.37469e-9_2.59051e-8_eff2075.dat",
        "apass_B": "pysynphot_data/grid/bosz/Generic_Johnson.B.dat",
        "apass_V": "pysynphot_data/grid/bosz/Generic_Johnson.V.dat",
        "apass_g": "pysynphot_data/grid/bosz/SLOAN_SDSS.g.dat",
        "apass_r": "pysynphot_data/grid/bosz/SLOAN_SDSS.r.dat",
        "apass_i": "pysynphot_data/grid/bosz/SLOAN_SDSS.i.dat",
        "gaia_U": "pysynphot_data/grid/bosz/Generic_Johnson.U.dat",
        "gaia_B": "pysynphot_data/grid/bosz/Generic_Johnson.B.dat",
        "gaia_V": "pysynphot_data/grid/bosz/Generic_Johnson.V.dat",
        "gaia_I": "pysynphot_data/grid/bosz/Generic_Johnson.I.dat",
        "smash_U": "pysynphot_data/grid/bosz/CTIO_DECam.u.dat",
        "smash_G": "pysynphot_data/grid/bosz/CTIO_DECam.g.dat",
        "smash_R": "pysynphot_data/grid/bosz/CTIO_DECam.r.dat",
        "smash_I": "pysynphot_data/grid/bosz/CTIO_DECam.i.dat",
        "smash_Z": "pysynphot_data/grid/bosz/CTIO_DECam.z.dat",
    }

    for key, rel_path in filter_paths.items():
        ctx[key] = ascii.read(synth_dir / rel_path, names=col_names, data_start=0)

    ctx["band_wavelengths"] = {
        "uvw2mag_SWIFT": 2075.69,
        "uvm2mag_SWIFT": 2246.56,
        "uvw1mag_SWIFT": 2715.68,
        "Umag_MCPS": 3706.29,
        "Bmag_MCPS": 4394.48,
        "Vmag_MCPS": 5438.23,
        "Imag_MCPS": 8568.89,
        "Jmag_2MASS": 12350.00,
        "Hmag_2MASS": 16620.00,
        "Kmag_2MASS": 21590.00,
        "Bmag_APASS": 4369.53,
        "Vmag_APASS": 5467.57,
        "gmag_APASS": 4671.78,
        "rmag_APASS": 6141.12,
        "imag_APASS": 7457.89,
        "Umag_GAIA": 3551.05,
        "Bmag_GAIA": 4369.53,
        "Vmag_GAIA": 5467.57,
        "Imag_GAIA": 8568.89,
    }

    # Expose notebook-style global names so copied functions stay unchanged.
    coords = ctx["coords"]
    df_lmc = ctx["df_lmc"]
    df_smc = ctx["df_smc"]
    df_lmc_prefinal = ctx["df_lmc_prefinal"]
    df_smc_prefinal = ctx["df_smc_prefinal"]
    smc_vis_binaries = ctx["smc_vis_binaries"]
    smc_opt_binaries = ctx["smc_opt_binaries"]
    lmc_vis_binaries = ctx["lmc_vis_binaries"]
    lmc_opt_binaries = ctx["lmc_opt_binaries"]
    choose_surveys = ctx["choose_surveys"]
    computed_models = ctx["computed_models"]
    gaia_bprp = ctx["gaia_bprp"]
    temp_stats = ctx["temp_stats"]
    lmc_wave = ctx["lmc_wave"]
    lmc_AlamAv = ctx["lmc_AlamAv"]
    smc_wave = ctx["smc_wave"]
    smc_AlamAv = ctx["smc_AlamAv"]
    wave = ctx["wave"]
    tmass_j = ctx["tmass_j"]
    tmass_h = ctx["tmass_h"]
    tmass_k = ctx["tmass_k"]
    mcps_U = ctx["mcps_U"]
    mcps_B = ctx["mcps_B"]
    mcps_V = ctx["mcps_V"]
    mcps_I = ctx["mcps_I"]
    swift_uvm1 = ctx["swift_uvm1"]
    swift_uvw1 = ctx["swift_uvw1"]
    swift_uvw2 = ctx["swift_uvw2"]
    apass_B = ctx["apass_B"]
    apass_V = ctx["apass_V"]
    apass_g = ctx["apass_g"]
    apass_r = ctx["apass_r"]
    apass_i = ctx["apass_i"]
    gaia_U = ctx["gaia_U"]
    gaia_B = ctx["gaia_B"]
    gaia_V = ctx["gaia_V"]
    gaia_I = ctx["gaia_I"]
    smash_U = ctx["smash_U"]
    smash_G = ctx["smash_G"]
    smash_R = ctx["smash_R"]
    smash_I = ctx["smash_I"]
    smash_Z = ctx["smash_Z"]
    band_wavelengths = ctx["band_wavelengths"]

    _FIT_CONTEXT = ctx
    return _FIT_CONTEXT


def synth_flux(filter_name,model_lam,model_flux):
    '''
    Calculates the synthetic photometric flux for the input model with the
    input filter transmission file.

    Parameters:
    filter_name: The filter file containing the wavelength (Angstroms) 
                    and intensity (erg/s/cm^2/A)
    model_lam: The model wavelengths in Angstroms
    model_flux: The model flux in erg/s/cm^2/A

    Returns:
    flux: The synthetic flux in erg/s/cm^2/A
    '''
    filter_lam = filter_name.columns[0]
    filter_flux = filter_name.columns[1]
    
    f_lam=np.zeros(len(filter_lam))
    for i in range(len(f_lam)):
        # This interpolates the model flux along the same wavelengths as the
        # transmission curve. This needs to be done for the integration to work
        f_lam[i]=np.interp(filter_lam[i],model_lam,model_flux)
    
    # Multiply the (interpolated) model flux by the filter intensity, and integrate
    top = np.trapezoid(np.multiply(f_lam,filter_flux),x=filter_lam)
    # Integrate the filter intensity alone
    bottom = np.trapezoid(filter_flux,x=filter_lam)
    # Divide!
    flux = top/bottom
    
    return flux


def gordon_redden(wave, flux, Av, wave_gordon, AlamAv):
    """Will redden an input spectrum based on the Gordon's Extinction Curve:

    Parameters:
    wave (angstrom): wavelengths of input spectrum in Angstrom
    flux (erg/s/cm/Ang): flux of spectrum; scaled version of this flux are fine.
    Av: Amount to de-redden by
    wave_gordon (Angstrom): wavelengths associated with the Gordon Curve
    AlamAv: associated array of Alambda/Av values for Gordon curve.

    Returns: a new flux array in same units as input
    """

    # Interpolate Gordone to the wavelengths of Spectrum:
    AlamAv_interp = np.interp(wave,wave_gordon,AlamAv)
    A_lambda = Av * AlamAv_interp

    #############
    NewSpec = flux * 10.0 ** (-0.4 * A_lambda)

    return NewSpec





def smash(index):
    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]
    
    if index < 377:
        row = df_smc[(df_smc['ra'] == RA) & (df_smc['dec'] == dec)]
    elif index >= 377 and index < 848:
        row = df_lmc[(df_lmc['ra'] == RA) & (df_lmc['dec'] == dec)] 
    elif index >= 848 and index < 1012:
        row = df_smc_prefinal[(df_smc_prefinal['ra'] == RA) & (df_smc_prefinal['dec'] == dec)] 
    elif index >= 1012:
        row = df_lmc_prefinal[(df_lmc_prefinal['ra'] == RA) & (df_lmc_prefinal['dec'] == dec)]
    else:
        raise ValueError(f"Index {index} out of expected range")

    smashmags = []
    smashmag_errs = []
    smashflux_jy = []
    smashflux_err_jy = []

    # Bands and their AB ZEROPOINTS in erg/s/cm^2/Angstrom (in UGRIZ order)
    smashband_zeropoints = {'Umag_SMASH': 7.48186e-9,
                       'Gmag_SMASH': 4.70792e-9,
                       'Rmag_SMASH': 2.64299e-9,
                       'Imag_SMASH': 1.78253e-9,
                       'Zmag_SMASH': 1.29484e-9}

    # Effective wavelengths (in Angstroms) (in UGRIZ order)
    smashband_wavelength = [3856.88,
                       4769.90,
                       6370.44,
                       7774.30,
                       9154.88]
    
    smash_band_names = ['Umag_SMASH', 'Gmag_SMASH', 'Rmag_SMASH', 'Imag_SMASH', 'Zmag_SMASH']

    # do this for UGRIZ mags
    for i, band in enumerate(smash_band_names):
        mag_col = f'{band}'
        err_col = f'e_{band}'
        
        # Check if the SMASH columns exist in the dataframe (they don't exist for prefinal stars)
        if len(row) > 0 and mag_col in row.columns and not pd.isna(row[mag_col].values[0]):
            magvalue = row[mag_col].values[0]
            magerr = row[err_col].values[0]
            
            # Convert magnitude to flux density
            flux_val = smashband_zeropoints[band] * 10**(-0.4 * magvalue)
            flux_err_val = flux_val * 0.921 * magerr
            
            smashmags.append(magvalue)
            smashmag_errs.append(magerr)
            smashflux_jy.append(flux_val)
            smashflux_err_jy.append(flux_err_val)
        else:
            smashmags.append(np.nan)
            smashmag_errs.append(np.nan)
            smashflux_jy.append(np.nan)
            smashflux_err_jy.append(np.nan)
    

    return smashband_wavelength, smashflux_jy, smashflux_err_jy, smashmags, smashmag_errs, smash_band_names, 


def observed_sed_allphot(index, coords, df_smc, df_lmc, df_smc_prefinal, df_lmc_prefinal, smc_vis_binaries, smc_opt_binaries, lmc_vis_binaries, lmc_opt_binaries, gaia_bprp, choose_surveys, flux=True, show=False):
    """ 
    Plots the SED for a given index in the coords dataframe
    Modified for multiprocessing - takes dataframes as parameters
    """
    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]
    binary = False

    if index < 377:
        row = df_smc[(df_smc['ra'] == RA) & (df_smc['dec'] == dec)]
            # Check if star is a binary based on Anna's binary files
        if ((smc_vis_binaries['ra'] == RA) & (smc_vis_binaries['dec'] == dec)).any() or ((smc_opt_binaries['ra'] == RA) & (smc_opt_binaries['dec'] == dec)).any():
            binary = True
        
    elif index >= 377 and index < 848:
        row = df_lmc[(df_lmc['ra'] == RA) & (df_lmc['dec'] == dec)]
        if ((lmc_vis_binaries['ra'] == RA) & (lmc_vis_binaries['dec'] == dec)).any() or ((lmc_opt_binaries['ra'] == RA) & (lmc_opt_binaries['dec'] == dec)).any():
            binary = True
    elif index >= 848 and index < 1012:
        row = df_smc_prefinal[(df_smc_prefinal['ra'] == RA) & (df_smc_prefinal['dec'] == dec)] 
    elif index >= 1012:
        row = df_lmc_prefinal[(df_lmc_prefinal['ra'] == RA) & (df_lmc_prefinal['dec'] == dec)]
    else:
        raise ValueError(f"Index {index} out of expected range")
    gaia_row = gaia_bprp[(gaia_bprp['ra'] == RA) & (gaia_bprp['dec'] == dec)]
    
    # Check if we found a match
    if len(row) == 0:
        raise ValueError(f"No matching star found for index {index} (RA={RA}, Dec={dec})")
    
    
    df = row.iloc[0]
    gaia_df = gaia_row.iloc[0] if len(gaia_row) > 0 else None

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
    'uvm2mag_SWIFT':4.66117e-9,   # UVM2

  # Optical (Gaia synthetic photometry) in VEGA
    'Umag_GAIA':3.49719e-9,    # U-band
    'Bmag_GAIA':6.72553e-9,    # B-band
    'Vmag_GAIA':3.5833e-9,    # V-band
    'Imag_GAIA':9.23651e-10 #,    # I-band
    # 'gmag_GAIA':5.45476e-9,    # g-band
    # 'rmag_GAIA':2.49767e-9	
    }

    band_AB_zeropoints_gaia = {    # AB zero points for Gaia bands
    'rmag_GAIA':2.85425e-9,    # r-band
    'gmag_GAIA':4.92255e-9,    # g-band
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
        'imag_APASS':7457.89,
        'Umag_GAIA': 3551.05,        # U gaia
        'Bmag_GAIA': 4369.53,        # B gaia
        'Vmag_GAIA': 5467.57,        # V gaia
        'Imag_GAIA': 8568.89 #,        # I gaia
        # 'rmag_GAIA': 6141.12,        # r gaia
        # 'gmag_GAIA': 4671.78         # g gaia
    }
    
    wavelengths = []
    fluxes = []
    flux_errors = []
    mags = []
    mag_errors = []
    band_names = []
    true_phot_band_counter = 0
    for band in band_zeropoints.keys():
        if band in df.index and not pd.isna(df[band]): #and band not in ['Umag_GAIA', 'Bmag_GAIA', 'Vmag_GAIA', 'Imag_GAIA', 'gmag_GAIA', 'rmag_GAIA']:
            # if band not in ['uvw2mag_SWIFT', 'uvm2mag_SWIFT', 'uvw1mag_SWIFT']:
            #     true_phot_band_counter += 1
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
                # true_phot_band_counter -= 1
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

        elif band in ['Umag_GAIA', 'Bmag_GAIA', 'Vmag_GAIA', 'Imag_GAIA', 'gmag_GAIA', 'rmag_GAIA']:
            if gaia_df is not None and band in gaia_df.index and not pd.isna(gaia_df[band]):
                mag = gaia_df[band]
                error_col = f'e_{band}'
                mag_err = gaia_df[error_col] if error_col in gaia_df.index and not pd.isna(gaia_df[error_col]) else None
                if mag_err is None or pd.isna(mag_err) or mag_err <= 0:
                    mag_err = 0.1
                if mag_err < 0.03:
                    mag_err = 0.03
                if mag_err > 0.36:
                    continue
                flux_ergs = band_zeropoints[band] * 10**(-0.4 * mag)
                flux_err_ergs = flux_ergs * 0.921 * mag_err
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

    return wavelengths, fluxes, flux_errors, mags, mag_errors, band_names, true_phot_band_counter, binary


def fit_models_to_star_flux(star_idx): 
    """
    Fits the best model spectra (the median) to the observed SED of a given star index,
    comparing models based on median parameters from temp_stats.
    Optionally plots the observed SED and best-fit models.
    """  
    _initialize_fit_context()
    from metrics import largest_amplitude

    RA = coords['RA'].iloc[star_idx]
    dec = coords['DEC'].iloc[star_idx]
    ######### grab the observed SED for the star
    obs = observed_sed_allphot(star_idx, coords, df_smc, df_lmc, df_smc_prefinal, df_lmc_prefinal, smc_vis_binaries, smc_opt_binaries, lmc_vis_binaries, lmc_opt_binaries, gaia_bprp, choose_surveys, show=False)
    obs_wavelengths, obs_fluxes, obs_flux_errors, obs_mags, obs_mag_errors, obs_band_names, true_phot_band_counter, binary = obs
    # obs = observed_sed_allphot(star_idx, coords, df_smc, df_lmc, choose_phot, flux=True, show=False)
    # obs_wavelengths, obs_fluxes, obs_flux_errors, obs_mags, obs_mag_errors, obs_band_names = obs

    smash_wavelengths, smash_fluxes, smash_flux_errors, smash_mags, smash_mag_errors, smash_band_names = smash(star_idx)
    valid_smash_indices = [i for i, flux in enumerate(smash_fluxes) if not np.isnan(flux)]
    smash_band_names_valid = [smash_band_names[i] for i in valid_smash_indices]
    smash_wavelengths_valid = [smash_wavelengths[i] for i in valid_smash_indices]

    available_bands = list(obs_band_names)  # Convert to list to ensure .index() method works
    band_to_filter = {
    'Jmag_2MASS': tmass_j, 'Hmag_2MASS': tmass_h, 'Kmag_2MASS': tmass_k,
    'Umag_MCPS': mcps_U, 'Bmag_MCPS': mcps_B, 'Vmag_MCPS': mcps_V, 'Imag_MCPS': mcps_I,
    'Bmag_APASS': apass_B, 'Vmag_APASS': apass_V, 'gmag_APASS': apass_g, 
    'rmag_APASS': apass_r, 'imag_APASS': apass_i,
    'uvm2mag_SWIFT': swift_uvm1, 'uvw1mag_SWIFT': swift_uvw1, 'uvw2mag_SWIFT': swift_uvw2,
    'Umag_SMASH': smash_U, 'Gmag_SMASH': smash_G, 'Rmag_SMASH': smash_R, 
   'Imag_SMASH': smash_I, 'Zmag_SMASH': smash_Z,
   'Umag_GAIA': gaia_U,'Bmag_GAIA': gaia_B,'Vmag_GAIA': gaia_V,'Imag_GAIA': gaia_I,
    }

    ########## determine the best fitting model to test based on medians in ysg_temp_fitting_summary_vX.csv
    # Best fitting U-redder parameters:
    median_teff_U_redder = temp_stats['teff_median_U_redder'][star_idx]
    median_logg_U_redder_raw = temp_stats['logg_median_U_redder'][star_idx]
    median_logg_U_redder = round(np.round(median_logg_U_redder_raw / 0.5) * 0.5, 2)  # Round to nearest 0.5 for model lookup
    median_av_U_redder_raw = temp_stats['av_median_U_redder'][star_idx]  # Keep raw value for display
    median_av_U_redder = round(np.round(median_av_U_redder_raw / 0.05) * 0.05, 2)  # Round to nearest 0.05 for model lookup
    median_metallicity = -0.75 if RA < 40 else -0.25 # SMC if index < 377, else LMC
    
    # Best fitting B-redder parameters:
    median_teff_B_redder = temp_stats['teff_median_B_redder'][star_idx]
    median_logg_B_redder_raw = temp_stats['logg_median_B_redder'][star_idx]
    median_logg_B_redder = round(np.round(median_logg_B_redder_raw / 0.5) * 0.5, 2)  # Round to nearest 0.5 for model lookup
    median_av_B_redder_raw = temp_stats['av_median_B_redder'][star_idx]  # Keep raw value for display
    median_av_B_redder = round(np.round(median_av_B_redder_raw / 0.05) * 0.05, 2)  # Round to nearest 0.05 for model lookup

    #Best fitting V-redder parameters:
    median_teff_V_redder = temp_stats['teff_median_V_redder'][star_idx]
    median_logg_V_redder_raw = temp_stats['logg_median_V_redder'][star_idx]
    median_logg_V_redder = round(np.round(median_logg_V_redder_raw / 0.5) * 0.5, 2)  # Round to nearest 0.5 for model lookup
    median_av_V_redder_raw = temp_stats['av_median_V_redder'][star_idx]  # Keep raw value for display
    median_av_V_redder = round(np.round(median_av_V_redder_raw / 0.05) * 0.05, 2)  # Round to nearest 0.05 for model lookup

    # Filter models based on median parameters - separate the two model sets properly
    # Use rounded Av values for exact model lookup
    models_U_redder = computed_models[
        (computed_models['teff'] == median_teff_U_redder) &
        (computed_models['logg'] == median_logg_U_redder) &
        (computed_models['av'] == median_av_U_redder) &
        (computed_models['metallicity'] == median_metallicity)
    ]
    models_B_redder = computed_models[
        (computed_models['teff'] == median_teff_B_redder) &
        (computed_models['logg'] == median_logg_B_redder) &
        (computed_models['av'] == median_av_B_redder) &
        (computed_models['metallicity'] == median_metallicity)
    ]
    models_V_redder = computed_models[
        (computed_models['teff'] == median_teff_V_redder) &
        (computed_models['logg'] == median_logg_V_redder) &
        (computed_models['av'] == median_av_V_redder) &
        (computed_models['metallicity'] == median_metallicity)
    ]
    plot_data = []
    
    # Process all models (full, B-redder and V-redder)
    for model_type, models in [('V_redder', models_V_redder), ('B_redder', models_B_redder), ('U_redder', models_U_redder)]:
        if len(models) == 0:
            print(f"No models found for {model_type}")
            print(f"  Parameters: Teff={median_teff_V_redder if model_type=='V_redder' else (median_teff_B_redder if model_type=='B_redder' else median_teff_U_redder)}, logg={median_logg_V_redder if model_type=='V_redder' else (median_logg_B_redder if model_type=='B_redder' else median_logg_U_redder)}, Av={median_av_V_redder if model_type=='V_redder' else (median_av_B_redder if model_type=='B_redder' else median_av_U_redder)}, metallicity={median_metallicity}")
            continue
        # take the first (should be only) model
        model = models.iloc[0]
        
        # load model spectrum
        model_spectrum = ascii.read(model['model'], names=['flux','cont'], data_start=0)
        # model_flux = Cardelli_redden(wave['wave'], model_spectrum['flux'], Av=model['av'])
        if star_idx < 377:
            model_flux = gordon_redden(wave['wave'], model_spectrum['flux'], Av=model['av'], wave_gordon=smc_wave, AlamAv=smc_AlamAv)
            simbad_maintype = df_smc['main_type'][(df_smc['ra'] == RA) & (df_smc['dec'] == dec)].values[0]
            simbad_spectype = df_smc['sp_type'][(df_smc['ra'] == RA) & (df_smc['dec'] == dec)].values[0]
            distance = df_smc['angDist'][(df_smc['ra'] == RA) & (df_smc['dec'] == dec)].values[0]
        elif star_idx >= 377 and star_idx < 848:
            model_flux = gordon_redden(wave['wave'], model_spectrum['flux'], Av=model['av'], wave_gordon=lmc_wave, AlamAv=lmc_AlamAv)
            simbad_maintype = df_lmc['main_type'][(df_lmc['ra'] == RA) & (df_lmc['dec'] == dec)].values[0]
            simbad_spectype = df_lmc['sp_type'][(df_lmc['ra'] == RA) & (df_lmc['dec'] == dec)].values[0]
            distance = df_lmc['angDist'][(df_lmc['ra'] == RA) & (df_lmc['dec'] == dec)].values[0]
        elif star_idx >= 848 and star_idx < 1012:
            model_flux = gordon_redden(wave['wave'], model_spectrum['flux'], Av=model['av'], wave_gordon=smc_wave, AlamAv=smc_AlamAv)
            simbad_maintype = df_smc_prefinal['main_type'][(df_smc_prefinal['ra'] == RA) & (df_smc_prefinal['dec'] == dec)].values[0]
            simbad_spectype = df_smc_prefinal['sp_type'][(df_smc_prefinal['ra'] == RA) & (df_smc_prefinal['dec'] == dec)].values[0]
            distance = df_smc_prefinal['angDist'][(df_smc_prefinal['ra'] == RA) & (df_smc_prefinal['dec'] == dec)].values[0]
        elif star_idx >= 1012:
            model_flux = gordon_redden(wave['wave'], model_spectrum['flux'], Av=model['av'], wave_gordon=lmc_wave, AlamAv=lmc_AlamAv)
            simbad_maintype = df_lmc_prefinal['main_type'][(df_lmc_prefinal['ra'] == RA) & (df_lmc_prefinal['dec'] == dec)].values[0]
            simbad_spectype = df_lmc_prefinal['sp_type'][(df_lmc_prefinal['ra'] == RA) & (df_lmc_prefinal['dec'] == dec)].values[0]
            distance = df_lmc_prefinal['angDist'][(df_lmc_prefinal['ra'] == RA) & (df_lmc_prefinal['dec'] == dec)].values[0]

        
        # calculate synthetic fluxes for available bands using the model
        model_fluxes = np.array([synth_flux(band_to_filter[band], wave['wave'], model_flux) for band in available_bands])
        smash_model_fluxes = np.array([synth_flux(band_to_filter[band], wave['wave'], model_flux) for band in smash_band_names_valid])
        
        # Determine reference band for flux scaling (prefer K, then H)
        if 'Kmag_2MASS' in available_bands:
            ref_band = 'Kmag_2MASS'
        else:
            print('No K band found in observed bands.')
            return
        
        ref_idx = available_bands.index(ref_band)
        
        # Calculate scaling factor using reference band
        flux_scale = obs_fluxes[ref_idx] / model_fluxes[ref_idx]
        
        # Scale model fluxes and spectrum
        scaled_model_fluxes = model_fluxes * flux_scale
        scaled_smash_model_fluxes = smash_model_fluxes * flux_scale
        scaled_model_spectrum = model_flux * flux_scale
        
        # Calculate luminosity from scaling
        luminosity = model['lum_unscaled'] * flux_scale
        logL = np.log10(luminosity / 3.826e33)  # solar lum in erg/s
        
        # Get wavelengths for plotting
        plot_wavelengths = [band_wavelengths[band] for band in available_bands]
        plot_smash_wavelengths = smash_wavelengths_valid
        
        # Store data for plotting
        plot_data.append({
            'model_type': model_type,
            'model': model,
            'scaled_model_fluxes': scaled_model_fluxes,
            'scaled_model_spectrum': scaled_model_spectrum,
            'plot_wavelengths': plot_wavelengths,
            # 'scaled_smash_model_fluxes': scaled_smash_model_fluxes,
            'plot_smash_wavelengths': plot_smash_wavelengths,
            # 'obs_fluxes': obs_fluxes,
            'available_bands': available_bands,
            'logL': logL,
            'luminosity': luminosity,
            'scaled_smash_model_fluxes': scaled_smash_model_fluxes
        })

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # inset axes in the lower right corner
    axins = inset_axes(ax, width="35%", height="35%", loc='upper right')

    bands_used = temp_stats['bands_used_for_fitting'].iloc[star_idx].split(',')
    print(f'bands used: {bands_used}')
    print(f'chi2 values: U-redder={temp_stats["chi2_U_redder_mean"].iloc[star_idx]:.3f}, B-redder={temp_stats["chi2_B_redder_mean"].iloc[star_idx]:.3f}, V-redder={temp_stats["chi2_V_redder_mean"].iloc[star_idx]:.3f}')

    # In your plotting loop
    for i, (wl, flux, errs, band) in enumerate(zip(obs_wavelengths, obs_fluxes, obs_flux_errors, obs_band_names)):
        if band in bands_used:
            color = 'lightblue'
        else:
            color = 'gray'
        ax.errorbar(wl, flux,  yerr=errs, mfc=color,ecolor=color, ms=10,  fmt='*', mec='black', zorder = 12)
    
    # Plot V-redder synthetic photometry
    data_V_redder = plot_data[0]
    ax.plot(data_V_redder['plot_wavelengths'], data_V_redder['scaled_model_fluxes'],
            'o', ms=10, mec='k', mfc='red', alpha=0.9, label='V-redder Synth Phot, Teff={:.0f}K'.format(data_V_redder['model']['teff']), zorder=8)
    ax.plot(data_V_redder['plot_smash_wavelengths'], data_V_redder['scaled_smash_model_fluxes'],
            's', ms=8, mec='k', mfc='red', alpha=0.9, label='SMASH Model Points', zorder=8)

    # Plot B-redder synthetic photometry
    data_B_redder = plot_data[1]
    ax.plot(data_B_redder['plot_wavelengths'], data_B_redder['scaled_model_fluxes'],
            'o', ms=10, mec='k', mfc='yellow', alpha=0.9, label='B-redder Synth Phot, Teff={:.0f}K'.format(data_B_redder['model']['teff']), zorder=8)
    ax.plot(data_B_redder['plot_smash_wavelengths'], data_B_redder['scaled_smash_model_fluxes'],
            's', ms=8, mec='k', mfc='yellow', alpha=0.9, label='SMASH Model Points', zorder=8)
    
    # Plot U-redder synthetic photometry
    data_U_redder = plot_data[2] 
    ax.plot(data_U_redder['plot_wavelengths'], data_U_redder['scaled_model_fluxes'],
            'o', ms=10, mec='k', mfc='blue', alpha=0.9, label='U-redder Synth Phot, Teff={:.0f}K'.format(data_U_redder['model']['teff']), zorder=8)
    ax.plot(data_U_redder['plot_smash_wavelengths'], data_U_redder['scaled_smash_model_fluxes'],
            's', ms=8, mec='k', mfc='blue', alpha=0.9, label='SMASH Model Points', zorder=8)

    # band labels for observed points
    for i, (wl, flux, band) in enumerate(zip(obs_wavelengths, obs_fluxes, obs_band_names)):
        ax.annotate(band, (wl, flux), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
        
    for i, (wl, flux, band) in enumerate(zip(obs_wavelengths, obs_fluxes, obs_band_names)):
        axins.annotate(band, (wl, flux), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    # model spectra in background
    ax.plot(wave['wave'], data_V_redder['scaled_model_spectrum'],
            '-', color='red', alpha=0.6, label='V-redder Spectrum', zorder=5)
    ax.plot(wave['wave'], data_B_redder['scaled_model_spectrum'],
            '-', color='yellow', alpha=0.6, label='B-redder Model Spectrum', zorder=5)
    ax.plot(wave['wave'], data_U_redder['scaled_model_spectrum'],
            '-', color='blue', alpha=0.6, label='U-redder Model Spectrum', zorder=5)
    ax.grid(True, alpha=0.3)

    for i, (wl, flux, errs, band) in enumerate(zip(obs_wavelengths, obs_fluxes, obs_flux_errors, obs_band_names)):
        if band in bands_used:
            color = 'lightblue'
        else:
            color = 'gray'
        axins.errorbar(wl, flux,  yerr=errs, mfc=color, ms=10,  fmt='*', mec='black', zorder = 12)
    
    axins.plot(data_V_redder['plot_wavelengths'], data_V_redder['scaled_model_fluxes'],
            'o', ms=8, mec='k', mfc='red', alpha=0.7, zorder=8)
    axins.plot(data_V_redder['plot_smash_wavelengths'], data_V_redder['scaled_smash_model_fluxes'],
            's', ms=6, mec='k', mfc='red', alpha=0.7, zorder=8)

    axins.plot(data_B_redder['plot_wavelengths'], data_B_redder['scaled_model_fluxes'],
        'o', ms=8, mec='k', mfc='yellow', alpha=0.7, zorder=8)
    axins.plot(data_B_redder['plot_smash_wavelengths'], data_B_redder['scaled_smash_model_fluxes'],
            's', ms=6, mec='k', mfc='yellow', alpha=0.7, zorder=8)

    axins.plot(data_U_redder['plot_wavelengths'], data_U_redder['scaled_model_fluxes'],
        'o', ms=8, mec='k', mfc='blue', alpha=0.7, zorder=8)
    axins.plot(data_U_redder['plot_smash_wavelengths'], data_U_redder['scaled_smash_model_fluxes'],
            's', ms=6, mec='k', mfc='blue', alpha=0.7, zorder=8)

    axins.plot(wave['wave'], data_V_redder['scaled_model_spectrum'],
            '-', color='red', alpha=0.6, label='V-redder Model Spectrum', zorder=5)
    axins.plot(wave['wave'], data_B_redder['scaled_model_spectrum'],
            '-', color='yellow', alpha=0.6, label='B-redder Model Spectrum', zorder=5)
    axins.plot(wave['wave'], data_U_redder['scaled_model_spectrum'],
            '-', color='blue', alpha=0.6, label='U-redder Model Spectrum', zorder=5)

    # Set zoom limits around U band
    axins.set_xlim(3500, 4800) 
    # Get U band flux values specifically - with error handling for missing bands
    try:
        u_idx_V_redder = data_V_redder['available_bands'].index('Vmag_MCPS')
        u_idx_B_redder = data_B_redder['available_bands'].index('Bmag_MCPS')
        u_idx_U_redder = data_U_redder['available_bands'].index('Umag_MCPS')
        u_obs_match = np.where(obs_band_names == 'Umag_MCPS')[0]
        
        if len(u_obs_match) > 0:
            u_idx_obs = u_obs_match[0]
            
            u_fluxes = [
                data_V_redder['scaled_model_fluxes'][u_idx_V_redder],
                data_B_redder['scaled_model_fluxes'][u_idx_B_redder],
                data_U_redder['scaled_model_fluxes'][u_idx_U_redder],
                obs_fluxes[u_idx_obs]
            ]
            
            minimum, maximum = np.min(u_fluxes) * 0.5, np.max(u_fluxes) * 2.0  
            axins.set_ylim(minimum, maximum)
        else:
            # If U band not in observations, use all available flux values for zoom
            all_visible_fluxes = [f for f in obs_fluxes if 10**(-15) < f < 10**(-12)]
            if len(all_visible_fluxes) > 0:
                minimum, maximum = np.min(all_visible_fluxes) * 0.5, np.max(all_visible_fluxes) * 2.0
                axins.set_ylim(minimum, maximum)
    except (ValueError, IndexError):
        # If any required bands are missing, set reasonable default limits
        axins.set_ylim(10**(-15), 10**(-13))
    
    axins.set_yscale('log')
    axins.set_xscale('log')
    # Remove all tick labels and ticks - must be done AFTER setting scales
    axins.set_xticks([])
    axins.set_yticks([])
    axins.tick_params(axis='both', which='both', bottom=False, top=False, 
                        left=False, right=False, labelbottom=False, labeltop=False,
                        labelleft=False, labelright=False, length=0)
    # Add title to inset
    axins.set_xlabel('U and B zoom-in', fontsize=12, labelpad=5) 
    
    # Plot additional models within +/-500K range
    for model_type, main_data, color in [('V_redder', data_V_redder, 'gold'), ('B_redder', data_B_redder, 'violet'), ('U_redder', data_U_redder, 'cyan')]:
        best_teff = main_data['model']['teff']
        best_logg = main_data['model']['logg']
        best_av = main_data['model']['av']
        best_metallicity = main_data['model']['metallicity']
        
        # Determine temperature increment based on the model grid: 250K below 12000K, 500K at/above 12000K
        if best_teff < 12000:
            temp_increment = 250
        else:
            temp_increment = 500
        
        # Find temperature variants: one below and one above the best fit
        temp_low = best_teff - temp_increment
        temp_high = best_teff + temp_increment
        
        # Determine allowed logg values based on temperature
        # At 12000K: logg can be 2.5 or 3.0
        # At 12500K and above: only logg = 3.0 is possible
        allowed_logg = [best_logg]  # Always include the best fit logg
        
        if temp_low == 12000 or temp_high == 12000:
            # At 12000K, allow both 2.5 and 3.0
            if 2.5 not in allowed_logg:
                allowed_logg.append(2.5)
            if 3.0 not in allowed_logg:
                allowed_logg.append(3.0)
        elif temp_high >= 12500:
            # At 12500K and above, only logg=3.0
            allowed_logg = [3.0]
        
        temp_models = computed_models[
            ((computed_models['teff'] == temp_low) | (computed_models['teff'] == temp_high)) &
            (computed_models['logg'].isin(allowed_logg)) &  # Surface gravity in allowed range
            (computed_models['av'] == best_av) &       # Same extinction
            (computed_models['metallicity'] == best_metallicity)
        ]

    ######### NEW PHOTOMETRY FROM SMASH

    ax.errorbar(smash_wavelengths, smash_fluxes, yerr=smash_flux_errors,
                ms=10, fmt='*', mec='palevioletred', mfc='lightpink', ecolor='k', label='SMASH', zorder=10)
    axins.errorbar(smash_wavelengths, smash_fluxes, yerr=smash_flux_errors,
                ms=10, fmt='*', mec='palevioletred', mfc='lightpink', ecolor='k', label='SMASH', zorder=10)
    
    # Check if star is in binary candidate files
    tolerance = 1e-5  # Small tolerance for coordinate matching
    in_optsmc = ((np.abs(smc_opt_binaries['ra'] - RA) < tolerance) & 
                (np.abs(smc_opt_binaries['dec'] - dec) < tolerance)).any()
    in_optlmc = ((np.abs(lmc_opt_binaries['ra'] - RA) < tolerance) & 
                (np.abs(lmc_opt_binaries['dec'] - dec) < tolerance)).any()
    in_vissmc = ((np.abs(smc_vis_binaries['ra'] - RA) < tolerance) & 
                (np.abs(smc_vis_binaries['dec'] - dec) < tolerance)).any()
    in_vislmc = ((np.abs(lmc_vis_binaries['ra'] - RA) < tolerance) & 
                (np.abs(lmc_vis_binaries['dec'] - dec) < tolerance)).any()
    in_binary_files = "yes" if (in_optsmc or in_vissmc or in_optlmc or in_vislmc) else "no"

    amp = largest_amplitude(star_idx)[1]
    if temp_stats.iloc[star_idx]['use_U'] == True:
        fitmodel_to_use = 'U-redder'
    elif temp_stats.iloc[star_idx]['use_B'] == True:
        fitmodel_to_use = 'B-redder'
    elif temp_stats.iloc[star_idx]['use_V'] == True:
        fitmodel_to_use = 'V-redder'

    # Add annotation with both model parameters - use raw (unrounded) Av values for display
    annotation_text = ('V-redder Model: Teff={:.0f}K, Mean Teff={:.0f} $\\pm$ {:.0f}K, logL={:.2f}, logg={:.1f}, Av={:.3f}\n'
                        'B-redder Model: Teff={:.0f}K, Mean Teff={:.0f} $\\pm$ {:.0f}K, logL={:.2f}, logg={:.1f}, Av={:.3f}\n'
                        'U-redder Model: Teff={:.0f}K, Mean Teff={:.0f} $\\pm$ {:.0f}K, logL={:.2f}, logg={:.1f}, Av={:.3f}\n'
                        'SIMBAD: Main Type = {}, Spec Type {}, Distance = {:.2f} arcsec\n'
                        'Model chosen: {}\n'
                        'Variability Amplitude: {:.2f} mag\n'
                        'In binary files: {}').format(
                        data_V_redder['model']['teff'], temp_stats.iloc[star_idx]['teff_mean_V_redder'], temp_stats.iloc[star_idx]['teff_std_V_redder'], data_V_redder['logL'], data_V_redder['model']['logg'], median_av_V_redder_raw,
                        data_B_redder['model']['teff'], temp_stats.iloc[star_idx]['teff_mean_B_redder'], temp_stats.iloc[star_idx]['teff_std_B_redder'], data_B_redder['logL'], data_B_redder['model']['logg'], median_av_B_redder_raw,
                        data_U_redder['model']['teff'], temp_stats.iloc[star_idx]['teff_mean_U_redder'], temp_stats.iloc[star_idx]['teff_std_U_redder'], data_U_redder['logL'], data_U_redder['model']['logg'], median_av_U_redder_raw,
                        simbad_maintype, simbad_spectype, distance,
                        fitmodel_to_use,
                        amp,
                        in_binary_files)
    
    ax.annotate(annotation_text, xy=(0.01, 0.18), xycoords='axes fraction', 
                va='top', color='black', fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), zorder=13)
    chi2_annotation = (f'$\\chi^2$ Values:\n'
                       f'V-redder: {temp_stats.iloc[star_idx]["chi2_V_redder"]:.2f}\n'
                       f'B-redder: {temp_stats.iloc[star_idx]["chi2_B_redder"]:.2f}\n'
                       f'U-redder: {temp_stats.iloc[star_idx]["chi2_U_redder"]:.2f}')
    ax.annotate(chi2_annotation, xy=(0.87, 0.01), xycoords='axes fraction', #  xy=(0.82, 0.87) 
            va='bottom', color='black', fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(r'Wavelength ($\AA$)')
    ax.set_ylabel(r'Flux (erg/s/cm$^{2}$/$\AA$)')
    ax.set_title(f'{RA}, {dec}; {star_idx} - Median Model Comparison: V-redder vs B-redder vs U-redder Fitting')
    ax.set_xlim(1000, 25000)
    ax.set_yscale('log')
    y_min = min(np.min(data_V_redder['scaled_model_fluxes']), np.min(data_B_redder['scaled_model_fluxes']), np.min(data_U_redder['scaled_model_fluxes']))
    ymax = max(np.max(data_V_redder['scaled_model_fluxes']), np.max(data_B_redder['scaled_model_fluxes']), np.max(data_U_redder['scaled_model_fluxes']))
    # ax.set_ylim(bottom=np.log10(y_min)*1E4, top=np.log10(1))  # Set y-limits to show 4 orders of magnitude above the minimum flux
    if ymax > 5*10**(-14):
        ax.set_ylim(10**(-16), 10**(-12))
    else:
        ax.set_ylim(10**(-16), 10**(-13))
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.show()
    return fig


def _build_pdf_for_index(index, g=True, change_period=None, output_dir=None,
                         include_hr_highlight=True, summary_file="summary_results03162026.csv",
                         include_sed_figure=True):
    from metrics import _capture_info_figures, coords

    if output_dir is None:
        out_dir = _MODULE_DIR / "figs" / "star_reports"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ra = coords["RA"].iloc[index]
    dec = coords["DEC"].iloc[index]
    pdf_name = f"{index}_{ra}{dec}_info_plots.pdf"
    pdf_path = out_dir / pdf_name

    figures = _capture_info_figures(
        index=index,
        g=g,
        change_period=change_period,
        include_hr_highlight=include_hr_highlight,
        summary_file=summary_file,
        var_df=None,
    )

    if include_sed_figure:
        sed_fig = fit_models_to_star_flux(index)
        if sed_fig is not None:
            figures.append(sed_fig)

    if len(figures) == 0:
        return None

    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")

    for fig in figures:
        plt.close(fig)

    return pdf_path


def _worker(args):
    idx, g, change_period, output_dir, include_hr_highlight, summary_file, include_sed_figure = args
    try:
        _initialize_fit_context()
        pdf_path = _build_pdf_for_index(
            index=idx,
            g=g,
            change_period=change_period,
            output_dir=output_dir,
            include_hr_highlight=include_hr_highlight,
            summary_file=summary_file,
            include_sed_figure=include_sed_figure,
        )
        if pdf_path is None:
            return idx, None, "No figures generated"
        return idx, str(pdf_path), None
    except Exception as exc:
        return idx, None, str(exc)


def run_parallel(indices, g=True, change_period=None, output_dir=None,
                 include_hr_highlight=True, summary_file="summary_results03162026.csv",
                 max_workers=None, include_sed_figure=True):
    indices = list(indices)
    if len(indices) == 0:
        return [], []

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 1)

    worker_args = [
        (idx, g, change_period, output_dir, include_hr_highlight, summary_file, include_sed_figure)
        for idx in indices
    ]

    succeeded = []
    failed = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, args): args[0] for args in worker_args}
        for n, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]
            try:
                result_idx, path_str, err = future.result()
                if err is None and path_str is not None:
                    succeeded.append(Path(path_str))
                else:
                    failed.append((result_idx, err))
                    print(f"[FAIL] {result_idx}: {err}")
            except Exception as exc:
                failed.append((idx, str(exc)))
                print(f"[CRASH] {idx}: {exc}")

            if n % 25 == 0:
                print(f"Progress: {n}/{len(indices)} completed")

    print(
        f"Parallel export finished: {len(succeeded)} succeeded, "
        f"{len(failed)} failed, workers={max_workers}"
    )
    return succeeded, failed


def parse_args():
    parser = argparse.ArgumentParser(description="Export info PDFs for many stars in parallel")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--stop", type=int, default=1270, help="Stop index (exclusive)")
    parser.add_argument("--max-workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--g", action="store_true", default=True, help="Use g-band (default: True)")
    parser.add_argument("--change-period", type=float, default=None, help="Override period in days")
    parser.add_argument("--output-dir", type=str, default="figs/star_reports", help="Output directory for per-star PDFs")
    parser.add_argument("--summary-file", type=str, default="summary_results03162026.csv", help="Summary CSV used for HR_highlight")
    parser.add_argument("--no-sed-figure", action="store_true", help="Disable adding fit_models_to_star_flux page to each PDF")
    parser.add_argument("--no-hr-highlight", action="store_true", help="Disable HR_highlight page in the output PDFs")
    return parser.parse_args()


def main():
    args = parse_args()
    run_parallel(
        indices=range(args.start, args.stop),
        g=args.g,
        change_period=args.change_period,
        output_dir=args.output_dir,
        include_hr_highlight=not args.no_hr_highlight,
        summary_file=args.summary_file,
        max_workers=args.max_workers,
        include_sed_figure=not args.no_sed_figure,
    )


if __name__ == "__main__":
    main()
