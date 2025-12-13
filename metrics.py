from view_and_clean import df_extract, df_extract_v2
from view_and_clean import telescope_separator,offset_corrector_window, offset_corrector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from astropy.timeseries import LombScargle
from matplotlib import gridspec
from scipy.signal import find_peaks

offsets_df = pd.read_csv('offsets.csv')
coords = pd.read_csv('merged_smc_lmc_coords.csv', comment='#', sep="\\s+", names=['RA', 'DEC'])
df_lmc = pd.read_csv('final_lmc_ysgcands.csv', comment='#') # , sep="\\s+"
df_smc = pd.read_csv('final_smc_ysgcands.csv', comment='#') # , sep="\\s+"


def mean_med_flux(index, df=None, telescopes=None, g=True, correct_offsets=True):
    """
    Calculate mean and median fluxes for each telescope.
    Args:
        index (int): The index of the object to analyze.
        df (DataFrame, optional): The DataFrame containing the light curve data.
        telescopes (list, optional): The list of telescopes to consider.
        g (bool, optional): If True, use g-band data; if False, use V-band data.
        correct_offsets (bool, optional): If True, apply offset corrections.
    Returns:
        tuple: A tuple containing the overall mean, means, overall median, medians, overall magnitudes mean, overall magnitudes median, magnitudes means, and magnitudes medians.
    """
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    means = []
    medians = []
    mags_means = []
    mags_medians = []
    for scope in telescopes:
        mean = df[df['telescope'] == scope]['flux_(mJy)'].mean()
        means.append(mean)
        mags_mean = df[df['telescope'] == scope]['mag'].mean()
        mags_means.append(mags_mean)

        median = df[df['telescope'] == scope]['flux_(mJy)'].median()
        medians.append(median)
        mags_median = df[df['telescope'] == scope]['mag'].median()
        mags_medians.append(mags_median)

    overall_mean = df['flux_(mJy)'].mean()
    overall_median = df['flux_(mJy)'].median()
    overall_mags_mean = df['mag'].mean()
    overall_mags_median = df['mag'].median()

    return overall_mean, means, overall_median, medians, overall_mags_mean, overall_mags_median, mags_means, mags_medians

def mean_med_flux_v2(index, df=None, telescopes=None, filematcher_result=None, g=True, correct_offsets=True):
    """
    Same as mean_med_flux but for v2 data structure.
    """
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    means = []
    medians = []
    mags_means = []
    mags_medians = []
    for scope in telescopes:
        mean = df[df['telescope'] == scope]['flux_(mJy)'].mean()
        means.append(mean)
        mags_mean = df[df['telescope'] == scope]['mag'].mean()
        mags_means.append(mags_mean)

        median = df[df['telescope'] == scope]['flux_(mJy)'].median()
        medians.append(median)
        mags_median = df[df['telescope'] == scope]['mag'].median()
        mags_medians.append(mags_median)

    overall_mean = df['flux_(mJy)'].mean()
    overall_median = df['flux_(mJy)'].median()
    overall_mags_mean = df['mag'].mean()
    overall_mags_median = df['mag'].median()

    return overall_mean, means, overall_median, medians, overall_mags_mean, overall_mags_median, mags_means, mags_medians


def fit_plotter_flux(index, df=None, telescopes=None, tail=1, g=True, correct_offsets=True):
    """
    Fit and plot the flux data for a given index.
    Args:
        index (int): The index of the object to analyze.
        df (DataFrame, optional): The DataFrame containing the light curve data.
        telescopes (list, optional): The list of telescopes to consider.
        tail (int, optional): The percentile tail to consider for outlier detection.
        g (bool, optional): If True, use g-band data; if False, use V-band data.
        correct_offsets (bool, optional): If True, apply offset corrections.
    """
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    mean_med = mean_med_flux(index, df=df, telescopes=telescopes, g=g)
    mean = mean_med[0]
    median = mean_med[2]

    mean_fit = mean * np.ones_like(df['HJD'])  # Create a constant fit line based on the mean flux
    median_fit = median * np.ones_like(df['HJD'])  # Create a constant fit line based on the median flux
    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    # colors = ['b', 'b', 'b', 'b', 'b', 'b', 'b']

    fluxes = df['flux_(mJy)'].values
    lower_percentile = np.percentile(fluxes, 0+tail)
    upper_percentile = np.percentile(fluxes, 100-tail)
    t_n1 = telescopes[0]
    for t in telescopes:
        # DEPENDENT ON LENGTH OF SMC/LMC INDICES
        if index <= 376:
            mask = (df['telescope'] == t) & (df['HJD'] > 2459325.00) & (df['HJD'] < 2459650.00)
            len_window = len(df[mask])
            mask_t_n1 = (df['telescope'] == t_n1) & (df['HJD'] > 2459325.00) & (df['HJD'] < 2459650.00)
            if len(df[mask]) >= len(df[mask_t_n1]):
                most_data_telescope = t
        if index > 376:
            mask = (df['telescope'] == t) & (df['HJD'] > 2459400.00) & (df['HJD'] < 2459725.00)
            len_window = len(df[mask])
            mask_t_n1 = (df['telescope'] == t_n1) & (df['HJD'] > 2459400.00) & (df['HJD'] < 2459725.00)
            if len(df[mask]) >= len(df[mask_t_n1]):
                most_data_telescope = t
        print(f"telescope {t}", len_window, len_window / len(df[df['telescope'] == t]))
        t_n1 = t

    print(f'Most data telescope in window: {most_data_telescope}')
    plt.rcParams['font.family'] = 'serif'
    fig, ax1 = plt.subplots(figsize=(18, 6))
    for j, tel in enumerate(telescopes):
        mask = df['telescope'] == tel
        ax1.errorbar(df[mask]['HJD'], df[mask]['flux_(mJy)'], yerr=df[mask]['flux_err'],
                     color=colors[j % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {tel}')
    # ax1.plot(df['HJD'], mean_fit, color='black', label='Mean Flux', zorder=10)
    # ax1.plot(df['HJD'], median_fit, color='grey', label='Median Flux', zorder=10)
    # ax1.axhline(lower_percentile, color='red', linestyle='--', label=f'{tail}% percentile')
    # ax1.axhline(upper_percentile, color='red', linestyle='--')
    ax1.set_xlabel('HJD', fontsize=14)
    ax1.set_ylabel('Flux (mJy)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=14)
    ax1.grid(True)

    RA = coords['RA'][index]
    dec = coords['DEC'][index]
    print(f'RA: {RA}, DEC: {dec}')
    # plt.title(f'{RA} {dec} (g-band)', fontsize=14)
    # plt.savefig(f'lc_plots_2025/{RA}{dec}_lcg.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def fit_plotter_flux_v2(index, df=None, telescopes=None, tail=1, filematcher_result=None, g=True, correct_offsets=True):
    """
    Same as fit_plotter_flux but for v2 data structure.
    """
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    mean_med = mean_med_flux_v2(index, df=df, telescopes=telescopes, g=g, filematcher_result=filematcher_result)
    mean = mean_med[0]
    median = mean_med[2]
    mean_fit = mean * np.ones_like(df['jd'])  # Create a constant fit line based on the mean flux
    median_fit = median * np.ones_like(df['jd'])  # Create a constant fit line based on the median flux
    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
    # colors = ['b', 'b', 'b', 'b', 'b', 'b', 'b']

    fluxes = df['flux'].values
    lower_percentile = np.percentile(fluxes, 0+tail)
    upper_percentile = np.percentile(fluxes, 100-tail)
    plt.rcParams['font.family'] = 'serif'

    fig, ax1 = plt.subplots(figsize=(12, 6))
    for j, tel in enumerate(telescopes):
        mask = df['telescope'] == tel
        ax1.errorbar(df[mask]['jd'], df[mask]['flux'], yerr=df[mask]['flux_err'],
                     color=colors[j % len(colors)], capsize=5, markeredgecolor='black', fmt='o')# , label=f'Telescope {tel}')
    ax1.plot(df['jd'], mean_fit, color='black', label='Mean Flux', zorder=10)
    ax1.plot(df['jd'], median_fit, color='grey', label='Median Flux', zorder=10)
    ax1.axhline(lower_percentile, color='red', linestyle='--', label=f'{tail}% percentile')
    ax1.axhline(upper_percentile, color='red', linestyle='--')
    ax1.set_xlabel('jd', fontsize=14)
    ax1.set_ylabel('Flux (mJy)', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    RA = coords['RA'][index]
    dec = coords['DEC'][index]
    print(f'RA: {RA}, DEC: {dec}')
    plt.title(f'{RA} {dec} (g-band)', fontsize=14)
    plt.show()
    plt.close(fig)

def curve_fitter_flux(index, df=None, telescopes=None, g=True, mean=False, correct_offsets=True):
    """
    Fit a constant model to the flux data for a given index.
    Args:
        index (int): The index of the object to analyze.
        df (DataFrame, optional): The DataFrame containing the light curve data.
        telescopes (list, optional): The list of telescopes to consider.
        g (bool, optional): If True, use g-band data; if False, use V-band data.
        mean (bool, optional): If True, use mean flux as initial guess; if False, use median flux.
        correct_offsets (bool, optional): If True, apply offset corrections.
    Returns:
        popt (ndarray): The optimal values for the parameters.
        pcov (ndarray): The covariance matrix of the parameters.
    """
    def constant_model(x, c):
        return np.full_like(x, c)
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    x = np.array(df['HJD'])
    y = np.array(df['flux_(mJy)'])
    mean, means, median, medians = mean_med_flux(index, df=df, telescopes=telescopes, g=g)
    if mean:
        mean_fit = mean * np.ones_like(x) 
        # mean = constant_model(x, mean)
        popt, pcov = sp.optimize.curve_fit(constant_model, x, y, sigma=df['flux_err'], absolute_sigma=True, p0=[mean])
    else:
        median_fit = median * np.ones_like(x)
        popt, pcov = sp.optimize.curve_fit(constant_model, x, y, sigma=df['flux_err'], absolute_sigma=True)
    return popt, pcov

def largest_amplitude(index, df=None, telescopes=None, g=True, correct_offsets=True):
    """
    Calculate the largest amplitude of variability for a given index.
    """
    # Calculate amplitude of variability
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    # mean, means, median, medians = mean_med_flux(index, g=g)
    
    # Calculate the amplitude as the difference between max and min flux
    max_flux = df['flux_(mJy)'].max()
    min_flux = df['flux_(mJy)'].min()
    amplitude = max_flux - min_flux

    max_mag = df['mag'].max()
    min_mag = df['mag'].min()
    amplitude_mag = max_mag - min_mag 
    
    print(f"\tLargest amplitude of variability (g-band: {g}): {amplitude:.2f} mJy, {amplitude_mag:.2f} mags")
    return amplitude


def offset_warning(index, g=True):
    """Deals with UNCORRECTED dataframe - offsets have not been corrected
    Finds the mean flux for each telescope and compares them to find offsets. Hard-coded threshold of 1.0 mJy for significant offset.
    """
    df, telescopes = df_extract(index, g=g)
    means= mean_med_flux(index, df=None, g=g)[1]
    offset_flag = 0
    offsets = {}
    for i, current_mean in enumerate(means):
        for j, comparison_mean in enumerate(means):
            if i != j:  # Avoid comparing the same telescope
                offset = current_mean - comparison_mean
                offsets[f"{telescopes[i]}_vs_{telescopes[j]}"] = offset
    print(f"Offsets for object {index} (g-band: {g}):")
    print(offsets)
    for scope, offset in offsets.items():
    ### Set threshold for significant offset
        if abs(offset) > 1.0: #2.0
            print(f"Warning: {scope} has a significant offset of {offset:.4f} mJy")
            offset_flag = 1
    return offset_flag
    # return offsets

def offset_warning_v2(index, filematcher_result=None, g=True):
    """Deals with UNCORRECTED dataframe - offsets have not been corrected
    Finds the mean flux for each telescope and compares them to find offsets. Hard-coded threshold of 1.0 mJy for significant offset.
    """
    df, telescopes = df_extract_v2(index, filematcher_result=filematcher_result, g=g, v2=True)
    means = mean_med_flux_v2(index, filematcher_result=filematcher_result, g=g)[1]
    print(means)
    offset_flag = 0
    offsets = {}
    for i, current_mean in enumerate(means):
        for j, comparison_mean in enumerate(means):
            if i != j:  # Avoid comparing the same telescope
                offset = current_mean - comparison_mean
                offsets[f"{telescopes[i]}_vs_{telescopes[j]}"] = offset
    print(f"Offsets for object {index} (g-band: {g}):")
    print(offsets)
    for scope, offset in offsets.items():
        ### Set threshold for significant offset
        if abs(offset) > 1.0:
            print(f"Warning: {scope} has a significant offset of {offset:.4f} mJy")
            offset_flag = 1
    return offset_flag


def chi(index, df=None, telescopes=None, confidence=0.95, g=True, correct_offsets=True, show=False):
    """Calculate the chi-squared statistic for a given target.
    Determines if the target is variable based on chi-squared test against the mean flux.
    Args:
        index (int): The index of the object to analyze.
        df (DataFrame, optional): The DataFrame containing the light curve data.
        telescopes (list, optional): The list of telescopes to consider.
        confidence (float, optional): The confidence level for the chi-squared test.
        g (bool, optional): If True, use g-band data; if False, use V-band data.
        correct_offsets (bool, optional): If True, apply offset corrections.
        show (bool, optional): If True, print the results.
    Returns:
        chi_squared (float): The calculated chi-squared statistic.
        chi2_threshold (float): The chi-squared threshold for the given confidence level.
        dof (int): The degrees of freedom.
        chi_flag (int): 0 if variable, 1 if not variable."""
    # uses the mean
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    x = np.array(df['HJD'])
    y = np.array(df['flux_(mJy)'])
    mean = mean_med_flux(index, df=df, telescopes=telescopes, g=g)[0]
    mean_fit = mean * np.ones_like(x)  # Create a constant fit line based on the mean magnitude

    # Calculate residuals
    residuals = y - mean_fit
    chi_squared = np.sum((residuals / df['flux_err']) ** 2)
    # chi_squared = np.sum(residuals **2 / mean_fit)
    
    # Degrees of freedom
    dof = len(y) - 1  # Number of data points minus number of parameters (1 for the mean)
    chi_flag = 0
    chi2_threshold = sp.stats.chi2.ppf(confidence, dof)  # 95% confidence level
    if chi_squared > chi2_threshold:
        if show:
            print(f"Object {index} is likely variable (chi-squared = {chi_squared:.2f}, threshold = {chi2_threshold:.2f}); less than {(1-confidence)*100:.2f} % chance of non-variability.")
    else:
        chi_flag = 1
        if show:
            print(f"Object {index} does not have a significant deviation from the mean flux (chi-squared = {chi_squared:.2f}, threshold = {chi2_threshold:.2f}).")
    return chi_squared, chi2_threshold, dof, chi_flag


def percentile_amplitude(index, df=None, telescopes=None, tails=5, g=True, correct_offsets=True):
    # Calculate amplitude of variability using percentiles
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    fluxes = df['flux_(mJy)'].values  # Convert to numpy array to use positional indexing
    lower_percentile = np.percentile(fluxes, 0+tails)
    upper_percentile = np.percentile(fluxes, 100-tails)
    amplitude = upper_percentile - lower_percentile

    mags = df['mag'].values  # Convert to numpy array to use positional indexing
    lower_mag_percentile = np.percentile(mags, 0+tails)
    upper_mag_percentile = np.percentile(mags, 100-tails)
    mag_amplitude = upper_mag_percentile - lower_mag_percentile
    print("Amplitude metrics:")
    print(f"\tLower {tails}% percentile: {lower_percentile:.2f} mJy, {lower_mag_percentile:.2f} mags")
    print(f"\tUpper {tails}% percentile: {upper_percentile:.2f} mJy, {upper_mag_percentile:.2f} mags")
    print(f"\tPercentile amplitude of variability (excluding the brightest and dimmest {tails}% of data) (g-band: {g}): {amplitude:.2f} mJy, {mag_amplitude:.2f} mags")
    return amplitude, lower_percentile, upper_percentile, mag_amplitude

def percentile_amplitude_v2(index, df=None, telescopes=None, filematcher_result=None, tails=5, g=True, correct_offsets=True):
    """Does not correct offsets (if so, need to change offsets_corrector to work with v2)"""
    # Calculate amplitude of variability using percentiles
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    fluxes = df['flux'].values  # Convert to numpy array to use positional indexing
    lower_percentile = np.percentile(fluxes, 0+tails)
    upper_percentile = np.percentile(fluxes, 100-tails)
    amplitude = upper_percentile - lower_percentile

    mags = df['mag'].values  # Convert to numpy array to use positional indexing
    lower_mag_percentile = np.percentile(mags, 0+tails)
    upper_mag_percentile = np.percentile(mags, 100-tails)
    mag_amplitude = upper_mag_percentile - lower_mag_percentile
    print(f"Lower {tails}% percentile: {lower_percentile:.2f} mJy, {lower_mag_percentile:.2f}")
    print(f"Upper {tails}% percentile: {upper_percentile:.2f} mJy, {upper_mag_percentile:.2f}")
    print(f"Percentile amplitude of variability (g-band: {g}): {amplitude:.2f} mJy, {mag_amplitude:.2f} mags")
    return amplitude, lower_percentile, upper_percentile, mag_amplitude

def plot_percentile_amplitude(index, df=None, telescopes=None, tails=5, g=True, correct_offsets=True):
    # Plot the flux values and the percentiles
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    fluxes = df['flux_(mJy)'].values  # Convert to numpy array to use positional indexing
    lower_percentile = np.percentile(fluxes, 0+tails)
    upper_percentile = np.percentile(fluxes, 100-tails)

    plt.figure(figsize=(10, 6))
    plt.plot(df['HJD'], fluxes, 'o', label='Flux values')
    plt.axhline(lower_percentile, color='red', linestyle='--', label=f'Lower {tails}% percentile')
    plt.axhline(upper_percentile, color='green', linestyle='--', label=f'Upper {tails}% percentile')
    # plt.title(f'Flux Values and Percentiles for Object {index} (g-band: {g})')
    plt.xlabel('HJD')
    plt.ylabel('Flux (mJy)')
    plt.legend()
    plt.show()

def aliasing(index, df=None, telescopes=None, g=True, auto=False, median=True, samples_per_peak=5, show=True, correct_offsets=True):
    """
    Finds aliases for removal.
    Parameters:
    -----------
    index : int
        Index of the object in the dataset
    auto : bool
        If True, use autopower to automatically determine frequency range and resolution
    median : bool
        If True, use the median flux for the Lomb-Scargle analysis; otherwise, use zero
    samples_per_peak : int
        Number of samples per peak in the periodogram
    show : bool
        If True, plot the Lomb-Scargle periodogram
    Returns:
    --------
    dict : Dictionary containing frequency, power, best frequency, best period, maximum power, false alarm probability, and peaks
    """
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    time = np.array(df['HJD']) 
    if median:
        values = np.full_like(df['flux_(mJy)'], df['flux_(mJy)'].mean())
    else:  
        values = np.zeros_like(df['flux_(mJy)'])
    errors = np.array(df['flux_err'])  

    lsalias = LombScargle(time, values, dy=errors)

    # LS for real values
    # real_values = np.array(df['flux_(mJy)'])  # Store real values for aliasing check
    # ls = LombScargle(time, real_values, dy=errors)
    # probabilities = 0.01
    # alarm_level = ls.false_alarm_level(probabilities)
    # noise_level = alarm_level.item()
    # findpeaks = find_peaks(power, height= noise_level, distance=samples_per_peak)

    min_freq = 2.0 / (time.max() - time.min()) # same range as in compute_lomb_scargle
    max_freq = 0.9

    if auto is not True:
        frequency = np.linspace(min_freq, max_freq, 
                              int((max_freq - min_freq) * len(time) * samples_per_peak))
        power = lsalias.power(frequency)
    else:
        frequency, power = lsalias.autopower(samples_per_peak=samples_per_peak)
    # Find best period
    best_freq = frequency[np.argmax(power)]
    best_period = 1.0 / best_freq
    max_power = np.max(power)

    probabilities = 0.01
    alarm_level = lsalias.false_alarm_level(probabilities)
    noise_level = alarm_level.item()  # Convert to Python scalar
    # peaks = find_peaks(power, height=max_power*0.1, distance=10)  # Find peaks with a minimum height of 10% of max power
    peaks = find_peaks(power, height=max_power*0.1, distance=10)
    if show:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(frequency, power, color='tab:blue', linestyle='-', linewidth=1)
        plt.xlim(0, max_freq)
        plt.xlabel('Frequency (1/day)')
        plt.ylabel('Power')
        plt.ylim(0, np.max(power) * 1.1)
        plt.show()
        plt.close(fig)    
    return frequency, power, peaks

def compute_lomb_scargle(index, df=None, telescopes=None, g=True, auto=False, median=True, samples_per_peak=5, report=True, show=False, subtract_median=True, correct_offsets=True):
    """
    Compute the Lomb-Scargle periodogram for a given object index.  
    Parameters:
    -----------
    index : int
        Index of the object in the dataset
    auto : bool
        If True, use autopower to automatically determine frequency range and resolution
    samples_per_peak : int
        Number of samples per peak in the periodogram
    median : bool
        If True, use the median flux for aliasing; otherwise, do not remove aliases
    report : bool
        If True, print the best frequency, period, and maximum power
    subtract_median : bool
        If True, subtract the median flux from the values before computing the periodogram
    Returns:
    --------
    dict : Dictionary containing frequency, power, best frequency, best period, maximum power, false alarm probability, and peaks
    """
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    time = np.array(df['HJD'])
    values = np.array(df['flux_(mJy)']) 
    errors = np.array(df['flux_err'])  
    if subtract_median:
        values -= np.median(values)
    
    ls = LombScargle(time, values, dy=errors)
    peaks = []
    peak_powers = []
    min_freq = 2.0 / (time.max() - time.min())  # Longest period is half the observation period
    max_freq = 0.9 

    # alias
    # lsalias = LombScargle(time, np.full_like(df['flux_(mJy)'], df['flux_(mJy)'].median()), dy=errors)

    # Calculate periodogram
    if auto is not True:
        frequency = np.linspace(min_freq, max_freq, 
                              int((max_freq - min_freq) * len(time) * samples_per_peak))
        power = ls.power(frequency)
    else:
        frequency, power = ls.autopower(samples_per_peak=samples_per_peak)

    probabilities = 0.01
    alarm_level = ls.false_alarm_level(probabilities)
    noise_level = alarm_level.item()  # Convert to Python scalar
    # alias_alarm_level = lsalias.false_alarm_level(probabilities).item()
    # if report:
    #     print(f"FAL: {alarm_level:.8f}, Alias FAL: {alias_alarm_level:.8f}")



    # alias = aliasing(index, auto=auto, median = median, samples_per_peak=samples_per_peak, show=False)
    findpeaks = find_peaks(power, height= noise_level, distance=samples_per_peak) 
    peaks = frequency[findpeaks[0]]
    peak_powers = findpeaks[1]['peak_heights']

    # Convert alias peak indices to frequencies
    # alias_frequencies = alias[0][alias[2][0]] if len(alias[2][0]) > 0 else [] 

    # this part is to normalize the alias powers to match the top peak of the original periodogram (top peak without removing aliases)
    if len(peak_powers) > 0:
        max_power = np.max(peak_powers) # max power of the original periodogram BEFORE alias removal
    else:
        max_power = np.max(power)
    # alias_power_normalized = alias[1] * (max_power / np.max(alias[1]))
    # alias = (alias[0], alias_power_normalized, alias[2])
    # alias_FAP = lsalias.false_alarm_probability(max_power)  # Calculate FAP for alias
    # if report:
    #     print(f"Alias false alarm probability: {alias_FAP:.3e}")

    # print(f"Peaks before aliasing check: {len(peaks)}")
    peaks_list = list(peaks)
    peak_powers_list = list(peak_powers)
    # print(len(peaks_list), len(peak_powers_list))
    
    # if len(peaks_list) > 0 and len(alias_frequencies) > 0: # if there are peaks and alias frequencies
    #     # Keep track of peaks to remove to avoid duplicates
    #     peaks_to_remove = []
    #     for alias_peak in alias_frequencies:
    #         for i, peak in enumerate(peaks_list):
    #             if abs(alias_peak - peak)/(alias_peak**2) < 0.5 and i not in peaks_to_remove:
    #                 if report:
    #                     print(f"Warning: Peak at {peak:.8f} 1/day is aliased (close to {alias_peak:.8f} 1/day).")
    #                 peaks_to_remove.append(i)
    #                 break  # only remove each peak once
        
        # # remove peaks in reverse order to maintain correct indices
        # for i in sorted(peaks_to_remove, reverse=True):
        #     # if i == 
        #     peaks_list.pop(i)
        #     peak_powers_list.pop(i)

        # peaks = np.array(peaks_list)
        # peak_powers = np.array(peak_powers_list)
        # # print(f"Peaks after aliasing check: {len(peaks)}")

    # if there are no peaks above false alarm level (nonvariable):
    if len(peaks) == 0:
        findpeaks = find_peaks(power, height=0, distance=samples_per_peak)
        peaks = frequency[findpeaks[0]]
        peak_powers = findpeaks[1]['peak_heights']
        best_freq = peaks[np.argmax(peak_powers)] # find peak with highest power, even below FA level
        best_period = 1.0 / best_freq
        max_power = np.max(peak_powers)
        alarm = ls.false_alarm_probability(max_power) 
    # if there are peaks above false alarm level (variable):
    else:
        best_freq = peaks[np.argmax(peak_powers)]
        best_period = 1.0 / best_freq
        max_power = np.max(peak_powers)
        alarm = ls.false_alarm_probability(max_power)

    alarm_level_flag = 0
    # if highest peak is under false alarm level, set flag for non-variability
    if max_power < alarm_level:
        alarm_level_flag = 1
    
    if report:
        print(f"Best frequency: {best_freq:.6f} 1/day, Best period: {best_period:.2f} days, Max power: {max_power:.3f}, False alarm probability: {alarm:.3e}")
        print(f"False alarm level for {probabilities*100:.1f}%: {alarm_level:.3e}")

    # # normalize alias powers to match top peak of original periodogram
    # alias_power_normalized = alias[1] * (max_power / np.max(alias[1]))
    # alias = (alias[0], alias_power_normalized, alias[2])

    if show:
        fig = plt.figure(figsize=(10, 6))
        # plt.plot(alias[0], alias[1], color='tab:blue', linestyle='-', linewidth=1)
        plt.plot(frequency, power, color='tab:blue', linestyle='-', linewidth=1, label = 'Original Periodogram')
        # plt.plot(alias[0], alias[1], color='tab:orange', linestyle='-', alpha = 0.8, linewidth=1, label = 'Aliases')
        plt.xlim(0, max_freq)
        if best_freq > 0:
            plt.axvline(x=best_freq, color='red', linestyle='--', alpha=0.7,
                        label=f'Best Period after alias removal = {1/best_freq:.2f} days')
        plt.axhline(y=alarm_level, color='grey', linestyle='--', label = 'FA level of original periodogram')
        # plt.axhline(y=alias_alarm_level, color='orange', linestyle='--', label = 'FA level of aliases')
        plt.xlim(0,0.06)
        plt.xlabel('Frequency (1/day)')
        plt.ylabel('Power')
        plt.legend()
        # plt.ylim(0, np.max(power) * 1.1)
        plt.show()
        plt.close(fig)

    return {
        'frequency': frequency,
        'power': power,
        'best_frequency': best_freq,
        'best_period': best_period,
        'max_power': max_power,
        'false_alarm_prob': alarm,
        'false_alarm_level': alarm_level,
        'peaks': peaks,
        'peak_powers': peak_powers,
        'observation_period': time.max() - time.min(),
        'alarm_level_flag': alarm_level_flag,
        'findpeaks': findpeaks
    }


def compute_lomb_scargle_alias(index, df=None, telescopes=None, g=True, auto=False, median=True, samples_per_peak=5, report=True, show=False, subtract_median=True, correct_offsets=True):
    """
    Compute the Lomb-Scargle periodogram for a given object index.  
    Parameters:
    -----------
    index : int
        Index of the object in the dataset
    auto : bool
        If True, use autopower to automatically determine frequency range and resolution
    samples_per_peak : int
        Number of samples per peak in the periodogram
    median : bool
        If True, use the median flux for aliasing; otherwise, do not remove aliases
    report : bool
        If True, print the best frequency, period, and maximum power
    subtract_median : bool
        If True, subtract the median flux from the values before computing the periodogram
    Returns:
    --------
    dict : Dictionary containing frequency, power, best frequency, best period, maximum power, false alarm probability, and peaks
    """
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    time = np.array(df['HJD'])
    values = np.array(df['flux_(mJy)']) 
    errors = np.array(df['flux_err'])  
    if subtract_median:
        values -= np.median(values)
    
    ls = LombScargle(time, values, dy=errors)
    peaks = []
    peak_powers = []
    min_freq = 2.0 / (time.max() - time.min())  # Longest period is half the observation period
    max_freq = 0.9 

    # alias
    lsalias = LombScargle(time, np.full_like(df['flux_(mJy)'], df['flux_(mJy)'].median()), dy=errors)

    # Calculate periodogram
    if auto is not True:
        frequency = np.linspace(min_freq, max_freq, 
                              int((max_freq - min_freq) * len(time) * samples_per_peak))
        power = ls.power(frequency)
    else:
        frequency, power = ls.autopower(samples_per_peak=samples_per_peak)

    probabilities = 0.01
    alarm_level = ls.false_alarm_level(probabilities)
    noise_level = alarm_level.item()  # Convert to Python scalar
    alias_alarm_level = lsalias.false_alarm_level(probabilities).item()
    if report:
        print(f"FAL: {alarm_level:.8f}, Alias FAL: {alias_alarm_level:.8f}")



    alias = aliasing(index, df=df, telescopes=telescopes, auto=auto, median=median, samples_per_peak=samples_per_peak, show=False)
    findpeaks = find_peaks(power, height=noise_level, distance=samples_per_peak)
    peaks = frequency[findpeaks[0]]
    peak_powers = findpeaks[1]['peak_heights']

    # Convert alias peak indices to frequencies
    alias_frequencies = alias[0][alias[2][0]] if len(alias[2][0]) > 0 else [] 

    # this part is to normalize the alias powers to match the top peak of the original periodogram (top peak without removing aliases)
    if len(peak_powers) > 0:
        max_power = np.max(peak_powers) # max power of the original periodogram BEFORE alias removal
    else:
        max_power = np.max(power)
    alias_power_normalized = alias[1] * (max_power / np.max(alias[1]))
    alias = (alias[0], alias_power_normalized, alias[2])
    alias_FAP = lsalias.false_alarm_probability(max_power)  # Calculate FAP for alias
    if report:
        print(f"Alias false alarm probability: {alias_FAP:.3e}")

    # print(f"Peaks before aliasing check: {len(peaks)}")
    peaks_list = list(peaks)
    peak_powers_list = list(peak_powers)
    # print(len(peaks_list), len(peak_powers_list))
    
    if len(peaks_list) > 0 and len(alias_frequencies) > 0: # if there are peaks and alias frequencies
        # Keep track of peaks to remove to avoid duplicates
        peaks_to_remove = []
        for alias_peak in alias_frequencies:
            for i, peak in enumerate(peaks_list):
                if abs(alias_peak - peak)/(alias_peak**2) < 0.5 and i not in peaks_to_remove:
                    if report:
                        print(f"Warning: Peak at {peak:.8f} 1/day is aliased (close to {alias_peak:.8f} 1/day).")
                    peaks_to_remove.append(i)
                    break  # only remove each peak once
        
        # remove peaks in reverse order to maintain correct indices
        for i in sorted(peaks_to_remove, reverse=True):
            # if i == 
            peaks_list.pop(i)
            peak_powers_list.pop(i)

        peaks = np.array(peaks_list)
        peak_powers = np.array(peak_powers_list)
        # print(f"Peaks after aliasing check: {len(peaks)}")

    # if there are no peaks above false alarm level (nonvariable):
    if len(peaks) == 0:
        findpeaks = find_peaks(power, height=0, distance=samples_per_peak)
        peaks = frequency[findpeaks[0]]
        peak_powers = findpeaks[1]['peak_heights']
        best_freq = peaks[np.argmax(peak_powers)] # find peak with highest power, even below FA level
        best_period = 1.0 / best_freq
        max_power = np.max(peak_powers)
        alarm = ls.false_alarm_probability(max_power) 
    # if there are peaks above false alarm level (variable):
    else:
        best_freq = peaks[np.argmax(peak_powers)]
        best_period = 1.0 / best_freq
        max_power = np.max(peak_powers)
        alarm = ls.false_alarm_probability(max_power)

    alarm_level_flag = 0
    # if highest peak is under false alarm level, set flag for non-variability
    if max_power < alarm_level:
        alarm_level_flag = 1
    

    if report:
        print(f"Best frequency: {best_freq:.6f} 1/day, Best period: {best_period:.2f} days, Max power: {max_power:.3f}, False alarm probability: {alarm:.3e}")
        print(f"False alarm level for {probabilities*100:.1f}%: {alarm_level:.3e}")

    # # normalize alias powers to match top peak of original periodogram
    # alias_power_normalized = alias[1] * (max_power / np.max(alias[1]))
    # alias = (alias[0], alias_power_normalized, alias[2])

    if show:
        fig = plt.figure(figsize=(10, 6))
        # plt.plot(alias[0], alias[1], color='tab:blue', linestyle='-', linewidth=1)
        plt.plot(frequency, power, color='tab:blue', linestyle='-', linewidth=1, label = 'Original Periodogram')
        plt.plot(alias[0], alias[1], color='tab:orange', linestyle='-', alpha = 0.8, linewidth=1, label = 'Aliases')
        plt.xlim(0, max_freq)
        if best_freq > 0:
            plt.axvline(x=best_freq, color='red', linestyle='--', alpha=0.7,
                        label=f'Best Period after alias removal = {1/best_freq:.2f} days')
        plt.axhline(y=alarm_level, color='grey', linestyle='--', label = 'FA level of original periodogram')
        plt.axhline(y=alias_alarm_level, color='orange', linestyle='--', label = 'FA level of aliases')
        plt.xlim(0,0.06)
        plt.xlabel('Frequency (1/day)')
        plt.ylabel('Power')
        plt.legend()
        # plt.ylim(0, np.max(power) * 1.1)
        plt.show() 
        plt.close(fig)

    return {
        'frequency': frequency,
        'power': power,
        'best_frequency': best_freq,
        'best_period': best_period,
        'max_power': max_power,
        'false_alarm_prob': alarm,
        'false_alarm_level': alarm_level,
        'peaks': peaks,
        'peak_powers': peak_powers,
        'observation_period': time.max() - time.min(),
        'alarm_level_flag': alarm_level_flag,
        'findpeaks': findpeaks
    }

def plot_periodogram(index, frequency, power, best_freq, false_alarm_level, title="Lomb-Scargle Periodogram", auto=False, show_best=True):
    """
    Plot the Lomb-Scargle periodogram.
    
    Parameters:
    -----------
    frequency : array-like
        Frequency values
    power : array-like
        Power values
    title : str
        Plot title
    freq_range : tuple or None
        (min_freq, max_freq) for x-axis limits
    show_best : bool
        Whether to mark the highest power peak
    """
    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]
    print(f"Grid sampled at {len(frequency)} points")
    fig = plt.figure(figsize=(10, 6))
    plt.plot(frequency, power, color='tab:blue', linestyle='-', linewidth=1)
    # plt.ylim(0, np.max(power) * 1.1)  # Set y-axis limit to 10% above max power
    
    if show_best:
        best_idx = np.where(frequency == best_freq)[0][0]  # Find index in frequency array
        plt.axvline(x=frequency[best_idx], color='red', linestyle='--', alpha=0.7, 
                   label=f'Best Period = {1/frequency[best_idx]:.2f} days')
        plt.plot(frequency[best_idx], power[best_idx], 'ro', markersize=8)
        # for peak in peaks:
        #     if peak != frequency[best_idx]:  # Exclude the tallest peak
        #         plt.axvline(x=peak, color='green', linestyle='--', alpha=0.5,
        #                    label=f'Peak at {1/peak:.2f} days')
    
    if auto is not True:
        plt.xlim(0, 0.9)
        # plt.xlim(0, 0.4)
    
    if show_best:
        plt.legend(fontsize=14)
    plt.axhline(y=false_alarm_level, color='grey', linestyle='--',
                label=f'False Alarm Level ({100*0.01:.1f}%): {false_alarm_level:.3e}')
    plt.xlabel('Frequency (1/day)', fontsize=16)
    plt.ylabel('Power', fontsize=16)
    plt.tick_params(axis='both', which='major', direction ='in', labelsize=16)
    plt.grid(True, alpha=0.3)
    # plt.title(f'{RA} {dec} periodogram', fontsize=14)
    # plt.tight_layout()
    # plt.savefig(f'lc_plots/{RA}{dec}_periodogramg.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'lc_plots_2025/{RA}{dec}_periodogramg.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_phase_fold(index, best_period, df=None, telescopes=None, g=True, phase_bins=2, correct_offsets=True, mag_space=False):
    """
    Phase fold time series data using a given period.
    
    Parameters:
    -----------
    time : array-like
        Time values
    period : float
        Period to fold by
    values : array-like or None
        Data values to fold
    errors : array-like or None
        Uncertainties in data values
    phase_bins : float
        Number of phase cycles to show (e.g., 2 shows 0-2 phases)
        
    Returns:
    --------
    dict : Dictionary containing phased data
    """
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]
    time = np.array(df['HJD'])  # Time in HJD
    flux = np.array(df['flux_(mJy)'])  # Flux in mJy
    mags = np.array(df['mag'])  # Magnitudes
    flux_errors = np.array(df['flux_err'])  # Flux uncertainties in mJy
    mag_errors = np.array(df['mag_err'])  # Magnitude uncertainties
    # period = lombs['best_period']  # Best period from Lomb-Scargle analysis
    # period = 30.89
    if best_period > (time.max() - time.min()):
        return None
    # Calculate phase
    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
    phased_time = (time / best_period) % phase_bins

    plt.figure(figsize=(10, 6))
    # colour by telescope:
    for i, telescope in enumerate(telescopes):
        mask = df['telescope'] == telescope
        if mag_space:
            plt.errorbar(phased_time[mask], mags[mask], yerr=mag_errors[mask], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {telescope}', color=colors[i % len(colors)])
        else:
            plt.errorbar(phased_time[mask], flux[mask], yerr=flux_errors[mask], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {telescope}', color=colors[i % len(colors)])
    plt.legend(fontsize=12)
    # single colour:
    # plt.errorbar(phased_time, flux, yerr=errors, fmt='o', alpha=0.7, markersize=4)
    plt.xlabel('Phase', fontsize=14)
    if mag_space:
        plt.ylabel('Magnitude', fontsize=14)
        plt.gca().invert_yaxis()  # Invert y-axis for magnitudes
    else:
        plt.ylabel('Flux (mJy)', fontsize=14)
    plt.tick_params(axis='both', which='major', direction ='in', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.title(f'{RA} {dec} (2 phases with period = {best_period:.2f} days)', fontsize=14)
    plt.tight_layout()
    # plt.savefig(f'lc_plots/{RA}{dec}_phaseg.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'lc_plots_2025/{RA}{dec}_phaseg.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Bin and find amplitude of narrow bin
    num_bins = int(np.ceil(len(phased_time) / 10))  # e.g., ~10 points per bin
    bins = np.linspace(0, phase_bins, num_bins + 1)
    bin_indices = np.digitize(phased_time, bins) - 1

    amplitudes = []
    for i in range(num_bins):
        bin_flux = flux[bin_indices == i]
        if len(bin_flux) > 1:
            amplitude = np.max(bin_flux) - np.min(bin_flux)
            amplitudes.append(amplitude)
    average_amplitude = np.mean(amplitudes)
    print(f"Average scatter (amplitude) per 10 point phase bin: {average_amplitude:.3f} mJy")
    return average_amplitude


def amplitude_per_period(index, best_period, df=None, telescopes=None, report=True, g=True, correct_offsets=True):
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    # lombs = compute_lomb_scargle(index, df=df, telescopes=telescopes, auto=False, samples_per_peak=10, report = False)  # freq_range=(0.0, 0.05),
    period = best_period
    if period > (df['HJD'].max() - df['HJD'].min()):
        return None
    # print(f"Best period found: {period:.2f} days")
    time = np.array(df['HJD'])
    flux = np.array(df['flux_(mJy)'])
    
    # Create period bins
    period_bins = np.floor(time / period).astype(int)
    unique_periods = np.unique(period_bins)
    
    amplitudes, period_numbers, bin_centers, bin_counts = [], [], [], []
    if report:
        print(f"Found {len(unique_periods)} period bins with period = {period:.2f} days")
    
    # Calculate amplitude for each period bin
    for period_num in unique_periods:
        mask = period_bins == period_num
        bin_flux = flux[mask]
        bin_time = time[mask]
        
        # Skip bins with too few data points
        # Adjust 0.66 factor
        if len(bin_flux) < 0.66*(len(df) / len(unique_periods)):
            if report:
                print(f"Skipping period {period_num} with {len(bin_flux)} points (too few data points)")
            continue
            

        for i in range(1, len(bin_time)):
            if bin_time[i] - bin_time[i-1] > 0.33 * period:
                if report:
                    print(f"Skipping period {period_num} with {len(bin_flux)} points (gap in time > 0.33 * period)")
                continue
        amplitude = np.max(bin_flux) - np.min(bin_flux) # Calculate amplitude for this bin
        
        # Store results
        amplitudes.append(amplitude)
        period_numbers.append(period_num)
        bin_centers.append(np.mean(bin_time))  # Center time of the bin
        bin_counts.append(len(bin_flux))  # Number of data points in bin
        if report:
            print(f"Period {period_num}: {len(bin_flux)} points, amplitude = {amplitude:.3f} mJy")
    
    # Convert to numpy arrays for easier handling
    amplitudes = np.array(amplitudes)
    avg_amplitude = np.mean(amplitudes)
    std_amplitude = np.std(amplitudes)
    print(f"\tAverage amplitude across all periods: {avg_amplitude:.3f} mJy")
    print(f"\tStandard deviation of amplitudes: {std_amplitude:.3f} mJy")

    if avg_amplitude < 5* std_amplitude:
        print(f"\tWarning: average amplitude ({avg_amplitude:.3f} mJy) is less than 5 sigma ({std_amplitude:.3f} mJy). This indicates significant variability in the amplitudes across periods.")
    else:
        print(f"\tAverage amplitude ({avg_amplitude:.3f} mJy) is greater than 5 sigma ({std_amplitude:.3f} mJy). This indicates consistent variability across periods.")
    period_numbers = np.array(period_numbers)
    bin_centers = np.array(bin_centers)
    bin_counts = np.array(bin_counts)
    
    # Return the results as a dictionary
    return {
        'amplitudes': amplitudes,
        'period_numbers': period_numbers,
        'bin_centers': bin_centers,
        'bin_counts': bin_counts,
        'period': period,
        'mean_amplitude': np.mean(amplitudes),
        'std_amplitude': np.std(amplitudes)
    }

def amplitude_per_period_plot(index, result, df=None, telescopes=None, g=True, correct_offsets=True):
    if df is None and correct_offsets == False:
        df, telescopes = df_extract(index, g=g)
    if df is None and correct_offsets == True:
        df, telescopes = offset_corrector(index, additive=False, show=False)
    # days_since_start = df['HJD'] - df['HJD'].min()
    plt.figure(figsize=(20, 6))
    plt.xlabel('Time (HJD)', fontdict={'fontsize': 18})
    plt.ylabel('Flux (mJy)', fontdict={'fontsize': 18})
    plt.plot(df['HJD'], df['flux_(mJy)'], '-')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # ticks going inward
    plt.tick_params(axis='both', which='major', labelsize=18, direction='in')

    bin_centers = result['bin_centers']
    amplitudes = result['amplitudes']
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

    # Amplitude vs. Time plot
    ax0 = plt.subplot(gs[0])
    ax0.plot(bin_centers, amplitudes, 'o', markersize=6)
    ax0.set_xlabel('Time (HJD)', fontdict={'fontsize': 14})
    ax0.set_ylabel('Amplitude (mJy)', fontdict={'fontsize': 14})
    ax0.grid(True, alpha=0.3)
    # ax0.set_ylim(4.2, 8.2) 
    ax0.tick_params(axis='both', which='major', labelsize=14, direction='in')

    # Histogram, sharing y-axis with amplitude plot
    ax1 = plt.subplot(gs[1], sharey=ax0)
    ax1.hist(amplitudes, bins=5, orientation='horizontal', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Number of Periods', fontdict={'fontsize': 14})
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.show()





def sed_plotter(index):
    """ Plots the SED for a given index in the coords dataframe
    """
    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]
    if index < 377:
        row = df_smc[(df_smc['ra'] == RA) & (df_smc['dec'] == dec)]
    else:
        row = df_lmc[(df_lmc['ra'] == RA) & (df_lmc['dec'] == dec)]
    if len(row) == 0:
        print("No matching row found in dataframe.")
        return
    # zeropoints in Jy for each band in Vega system
    band_zeropoints = {
    # Near-infrared (2MASS)
    'Jmag': 1594.0,    # J-band
    'Hmag': 1024.0,    # H-band  
    'Kmag': 666.80,     # K-band
    # Optical (Johnson-Cousins)
    'Umag': 1801.36,    # U-band
    'Bmag': 4019.91,    # B-band
    'Vmag': 3633.35,    # V-band
    'Imag': 2309.23,    # I-band
    # UV (Swift UVOT)
    'uvw1_mag': 926.84,  # UVW1
    'uvw2_mag': 753.35,   # UVW2  
    'uvm2_mag': 786.13   # UVM2
    }

    #bands and their Vega zero points in erg/s/cm^2/Angstrom
    # band_zeropoints = {
    # # Near-infrared (2MASS)
    # 'Jmag':3.0596e-10,    # J-band
    # 'Hmag':3.09069e-10,    # H-band
    # 'Kmag':4.20615e-11,     # K-band
    # # Optical (Johnson-Cousins)
    # 'Umag':4.08739e-9,    # U-band
    # 'Bmag':6.21086e-9,    # B-band
    # 'Vmag':3.64047e-9,    # V-band
    # 'Imag':9.23651e-10,    # I-band
    # # UV (Swift UVOT)
    # 'uvw1_mag':4.02204e-9,  # UVW
    # 'uvw2_mag':5.37469e-9,   # UVW2
    # 'uvm2_mag':4.66117e-9   # UVM2
    # }
    
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
            mag_err = df[error_col] # if error_col in df.index and not pd.isna(df[error_col]) else 0.1
            
            # Convert magnitude to flux density (in Jy)
            # Flux = zeropoint * 10^(-0.4 * mag)
            # print(f"band zeropoints:{band_zeropoints[band]}, mag: {mag}")
            flux_jy = band_zeropoints[band] * 10**(-0.4 * mag)
            # flux_mjy = flux_jy * 1000

            # Convert magnitude error to flux error
            # dF/F = 0.4 * ln(10) * dmag ≈ 0.921 * dmag
            flux_err_jy = flux_jy * 0.921 * mag_err
            # flux_err_mjy = flux_mjy * 0.921 * mag_err
            
            wavelengths.append(band_wavelengths[band])
            fluxes.append(flux_jy)
            flux_errors.append(flux_err_jy)
            # band_names.append(band.replace('mag', ''))
    
    # Sort by longest to shortest wavelength:
    sorted_indices = np.argsort(wavelengths)[::-1]
    wavelengths = np.array(wavelengths)[sorted_indices]
    fluxes = np.array(fluxes)[sorted_indices]
    flux_errors = np.array(flux_errors)[sorted_indices]
    band_names = np.array(band_names)[sorted_indices]

    fig = plt.figure()
    # plt.scatter(wavelengths, fluxes, color='blue')
    # plt.errorbar(wavelengths, fluxes, yerr=flux_errors, fmt='o', color='blue', label='Flux')
    for i in range(len(wavelengths)):
        plt.errorbar(wavelengths[i], fluxes[i], yerr=flux_errors[i], fmt='o', label=band_names[i])
    plt.xlabel('Wavelength ($\\AA$)')
    plt.ylabel('Flux (Jy)')
    # plt.ylabel('Flux (erg/s/cm$^2$/Å)')
    # plt.yscale('log')
    plt.title(f'SED for Target {index}')
    plt.legend()
    plt.grid()
    plt.show()


def info(index, g=True, change_period=None):
        # plots
        # individual_plotter(index, seeoutliers=True)
        df, telescopes = offset_corrector(index, g=g, mag_space=True, show_window=False)
        try:
                lombs = compute_lomb_scargle(index, df=df, telescopes=telescopes, g=g, auto=False, samples_per_peak=10)
                if change_period is None:
                        best_period = lombs['best_period']
                else:
                        best_period = change_period
                plot_periodogram(index, lombs['frequency'], lombs['power'], lombs['best_frequency'], lombs['false_alarm_level'], lombs['peaks'], lombs['observation_period'], show_best=True)
                plot_phase_fold(index, best_period, phase_bins=2, mag_space=True)
                sed_plotter(index)
                # computations
                overall_mean, means, overall_median, medians, overall_mags_mean, overall_mags_median, mags_means, mags_medians = mean_med_flux(index, df=df, telescopes=telescopes)
                chi_squared_95, chi2_threshold_95, dof_95, chi_flag_95 = chi(index, df=df, telescopes=telescopes, g=g, confidence=0.95)
                chi_squared_997, chi2_threshold_997, dof_997, chi_flag_997 = chi(index, df=df, telescopes=telescopes, g=g, confidence=0.997)
                chi_squared_68, chi2_threshold_68, dof_68, chi_flag_68 = chi(index, df=df, telescopes=telescopes, g=g, confidence=0.68)
                # offset_flag = offset_warning(index)
                lombs = compute_lomb_scargle(index, df=df, telescopes=telescopes, g=g, auto=False, median=False, subtract_median=False, samples_per_peak=10, report=False)
                alarm_level_flag = lombs['alarm_level_flag']
                print(lombs['peaks'])
                amplitude_5, lower_percentile_5, upper_percentile_5, mag_amplitude_5 = percentile_amplitude(index, df=df, telescopes=telescopes, g=g, tails=5)
                amplitude_1, lower_percentile_1, upper_percentile_1, mag_amplitude_1 = percentile_amplitude(index, df=df, telescopes=telescopes, g=g, tails=1)
                largest_amp = largest_amplitude(index, df=df, telescopes=telescopes, g=g)
                result = amplitude_per_period(index, best_period, df=df, telescopes=telescopes, g=g, report=False)

        except Exception as e:
                print(f"An error occurred while processing index {index}: {e}")
                return


def observed_sed(index, flux=True, show=False):
    """ Plots the SED for a given index in the coords dataframe
    """
    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]
    if index < 377:
        row = df_smc[(df_smc['ra'] == RA) & (df_smc['dec'] == dec)]
    else:
        row = df_lmc[(df_lmc['ra'] == RA) & (df_lmc['dec'] == dec)]
    if len(row) == 0:
        print("No matching row found in dataframe.")
        return
    # bands and their Vega zero points (in Jy)
    # band_zeropoints = {
    # # Near-infrared (2MASS)
    # 'Jmag': 1594.0,    # J-band
    # 'Hmag': 1024.0,    # H-band  
    # 'Kmag': 666.80,     # K-band
    # # Optical (Johnson-Cousins)
    # 'Umag': 1801.36,    # U-band
    # 'Bmag': 4019.91,    # B-band
    # 'Vmag': 3633.35,    # V-band
    # 'Imag': 2309.23,    # I-band
    # # UV (Swift UVOT)
    # 'uvw1_mag': 926.84,  # UVW1
    # 'uvw2_mag': 753.35,   # UVW2  
    # 'uvm2_mag': 786.13   # UVM2
    # }

    #bands and their Vega zero points in erg/s/cm^2/Angstrom
    band_zeropoints = {
    # Near-infrared (2MASS)
    'Jmag':1.11933e-9,    # J-band
    'Hmag':3.09069e-10,    # H-band
    'Kmag':4.20615e-11,     # K-band
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
            
            # FIX: Handle missing or invalid magnitude errors
            if error_col in df.index:
                mag_err = df[error_col]
            else:
                mag_err = None
            if mag_err is None or pd.isna(mag_err) or mag_err <= 0:
                print(f"WARNING: Missing or invalid error for {band} (error={mag_err}), using default 0.1")
                mag_err = 0.1  # Default error if missing, NaN, or zero/negative

            if mag_err < 0.03:
                mag_err = 0.03  # Set minimum error to 0.03 mag

            if mag_err > 0.36:
                # drop the data point if error is too large
                band = np.nan
                continue


            # Skip if magnitude itself is invalid
            if pd.isna(mag):
                print(f"WARNING: Invalid magnitude for {band}, skipping")
                continue
            
            # Convert magnitude to flux density (in Jy)
            # Flux = zeropoint * 10^(-0.4 * mag)
            flux_jy = band_zeropoints[band] * 10**(-0.4 * mag)
            # flux_mjy = flux_jy * 1000

            # Convert magnitude error to flux error: dF/F = 0.4 * ln(10) * dmag ≈ 0.921 * dmag
            flux_err_jy = flux_jy * 0.921 * mag_err
            # flux_err_mjy = flux_mjy * 0.921 * mag_err
            
            wavelengths.append(band_wavelengths[band])
            fluxes.append(flux_jy)
            flux_errors.append(flux_err_jy)
            mags.append(mag)
            mag_errors.append(mag_err)
            # band_names.append(band.replace('mag', ''))
    
    # Sort by longest to shortest wavelength:
    sorted_indices = np.argsort(wavelengths)[::-1]
    wavelengths = np.array(wavelengths)[sorted_indices]
    fluxes = np.array(fluxes)[sorted_indices]
    flux_errors = np.array(flux_errors)[sorted_indices]
    mags = np.array(mags)[sorted_indices]
    mag_errors = np.array(mag_errors)[sorted_indices]
    band_names = np.array(band_names)[sorted_indices]

    if show:
        fig = plt.figure()
        # plt.scatter(wavelengths, fluxes, color='blue')
        # plt.errorbar(wavelengths, fluxes, yerr=flux_errors, fmt='o', color='blue', label='Flux')
        for i in range(len(wavelengths)):
            # plt.errorbar(wavelengths[i], fluxes[i], yerr=flux_errors[i], fmt='o', label=band_names[i])
            if flux:
                plt.errorbar(wavelengths[i], fluxes[i], yerr=flux_errors[i], fmt='o', label=band_names[i])
            else:
                plt.errorbar(wavelengths[i], mags[i], yerr=mag_errors[i], fmt='o', label=band_names[i])
        plt.xlabel('Wavelength ($\\AA$)')
        if flux:
            plt.ylabel('Flux (Jy)')
            # plt.yscale('log')
        else:
            plt.ylabel('Magnitudes')
            plt.gca().invert_yaxis()
        plt.title(f'{RA}, {dec} SED')
        plt.legend()
        plt.grid()
        plt.show()

    return wavelengths, fluxes, flux_errors, mags, mag_errors, band_names

# def reduced_chi(index, g=True):
#     # uses the mean
#     df, telescopes = df_extract(index, g=g)
#     x = np.array(df['HJD'])
#     y = np.array(df['flux_(mJy)'])
#     mean, means, median, medians = mean_med_flux(index, g=g)
#     mean_fit = mean * np.ones_like(x)  # Create a constant fit line based on the mean magnitude

#     # Calculate residuals
#     residuals = y - mean_fit
#     chi_squared = np.sum((residuals / df['flux_err']) ** 2)
    
#     # Degrees of freedom
#     dof = len(y) - 1  # Number of data points minus number of parameters (1 for the mean)
    
#     # Reduced chi-squared
#     reduced_chi_squared = chi_squared / dof
#     if reduced_chi_squared > 10:
#         print(f"Reduced chi-squared for object {index} (g-band: {g}): \n{reduced_chi_squared:.2f} - Potential variability detected")
#     print(f"Degrees of freedom for object {index} (g-band: {g}): {dof}")
#     print(f"Chi-squared for object {index} (g-band: {g}): {chi_squared:.2f}")
#     return chi_squared, dof, reduced_chi_squared








# def local_peak_differences(index, g=True):

#     df, telescopes = df_extract(index, g=g)
#     fluxes = df['flux_(mJy)'].values  # Convert to numpy array to use positional indexing
#     # fluxes = np.asarray(fluxes)
#     # peaks = []
#     # troughs = []
#     extrema = []
#     # Adjusted range to prevent accessing invalid indices
#     for i in range(2, len(fluxes)-2):  # Changed from range(1, len(fluxes)-1)
#         if fluxes[i] > fluxes[i-1] and fluxes[i] > fluxes[i+1] and fluxes[i] > fluxes[i-2] and fluxes[i] > fluxes[i+2]:
#             extrema.append(i)
#         if fluxes[i] < fluxes[i-1] and fluxes[i] < fluxes[i+1] and fluxes[i] < fluxes[i-2] and fluxes[i] < fluxes[i+2]:
#             extrema.append(i)
#     # Merge and sort peaks and troughs for adjacent pairs
#     # extrema = sorted(peaks + troughs)
#     # print(extrema  )
#     diffs = []
#     for i in range(1, len(extrema)):
#         diff = abs(fluxes[extrema[i]] - fluxes[extrema[i-1]])
#         diffs.append(diff)
#     avg_diff = np.mean(diffs) if diffs else np.nan
#     return diffs, avg_diff

# # Example usage:
# diffs, avg_diff = local_peak_differences(target)
# print("Differences:", diffs)
# print("Average difference:", avg_diff)

## doesnt work well even for a very regular lightcurve.
## maybe could do in segments over 1.5 periods

def compute_lomb_scargle_old(index, auto=False, samples_per_peak=5, report = True, subtract_median=True, correct_offsets=True):
    """
    Compute Lomb-Scargle periodogram for given time series data.
    
    Parameters:
    -----------
    time : array-like
        Time values (phase)
    values : array-like
        Data values (magnitudes or normalized magnitudes)
    errors : array-like
        Uncertainties in data values
    freq_range : tuple or None
        (min_freq, max_freq) - if None, uses autopower defaults
    samples_per_peak : int
        Number of samples per peak for frequency grid
        
    Returns:
    --------
    dict : Dictionary containing frequency, power arrays and best period info
    """
    df, telescopes = df_extract(index, g=True)
    # if correct_offsets == True:
    #     df = offset_corrector(index, additive=False, show=False)
    time = np.array(df['HJD'])  # Time in HJD
    values = np.array(df['flux_(mJy)'])  # Flux in mJy
    errors = np.array(df['flux_err'])  # Flux uncertainties in mJy
    if subtract_median:
        values -= np.median(values)
    
    # Create Lomb-Scargle object
    ls = LombScargle(time, values, dy=errors)
    peaks = []
    peak_powers = []
    min_freq = 2.0 / (time.max() - time.min())  # Minimum frequency based on observation period
    max_freq = 0.9
    # min_freq = 0.001
    # max_freq = 1.5
    # Calculate periodogram
    if auto is not True:
        # frequency = np.linspace(freq_range[0], freq_range[1], 
                            #   int((freq_range[1] - freq_range[0]) * len(time) * samples_per_peak)) 
        frequency = np.linspace(min_freq, max_freq, 
                              int((max_freq - min_freq) * len(time) * samples_per_peak))
        power = ls.power(frequency)
    else:
        frequency, power = ls.autopower()#(samples_per_peak=samples_per_peak)

    best_freq = frequency[np.argmax(power)] 
    best_period = 1.0 / best_freq
    max_power = np.max(power)

    alarm = ls.false_alarm_probability(max_power)
    probabilities = 0.01
    alarm_level = ls.false_alarm_level(probabilities)

    if best_period > (time.max() - time.min()):
        print(f"Warning: Best period {best_period:.2f} days exceeds the time range of the data ({time.max() - time.min():.2f} days).")
    for i in range(len(power)+1 - 1):
        if power[i] > alarm_level and power[i] > power[i-1] and power[i] > power[i+1]:
            if power[i] == max_power:  
                continue # Skip the maximum power peak
            peaks.append(frequency[i])
            peak_powers.append(power[i])
    plt.figure(figsize=(10, 6))
    plt.plot(frequency, power, color='tab:blue', linestyle='-', linewidth=1)
    plt.show()
    if report:
        print(f"Best frequency: {best_freq:.6f} 1/day, Best period: {best_period:.2f} days, Max power: {max_power:.3f}, False alarm probability: {alarm:.3e}")
        # print(f"False alarm levels: {', '.join([f'{p*100:.1f}%: {level:.3e}' for p, level in zip(probabilities, alarm_level)])}")
        print(f"False alarm level for {probabilities*100:.1f}%: {alarm_level:.3e}")
    alarm_level_flag = 0
    if max_power < alarm_level:
        alarm_level_flag = 1
    return {
        'frequency': frequency,
        'power': power,
        'best_frequency': best_freq,
        'best_period': best_period,
        'max_power': max_power,
        'false_alarm_prob': alarm,
        'false_alarm_level': alarm_level,
        'peaks': peaks,
        'peak_powers': peak_powers,
        'observation_period': time.max() - time.min(),
        'alarm_level_flag': alarm_level_flag
    }


# def plot_periodogram(index, frequency, power, false_alarm_level, peaks, observation_period, title="Lomb-Scargle Periodogram", auto=False, show_best=True):
#     """
#     Plot the Lomb-Scargle periodogram.
    
#     Parameters:
#     -----------
#     frequency : array-like
#         Frequency values
#     power : array-like
#         Power values
#     title : str
#         Plot title
#     freq_range : tuple or None
#         (min_freq, max_freq) for x-axis limits
#     show_best : bool
#         Whether to mark the highest power peak
#     """
#     coords = pd.read_csv('merged_smc_lmc_coords.csv', comment='#', sep="\\s+", names=['RA', 'DEC'])
#     RA = coords['RA'].iloc[index]
#     dec = coords['DEC'].iloc[index]
#     print(f"Grid sampled at {len(frequency)} points")
#     plt.figure(figsize=(10, 6))
#     plt.plot(frequency, power, color='tab:blue', linestyle='-', linewidth=1)
#     plt.ylim(0, np.max(power) * 1.1)  # Set y-axis limit to 10% above max power
    
#     if show_best:
#         best_idx = np.argmax(power)
#         plt.axvline(x=frequency[best_idx], color='red', linestyle='--', alpha=0.7, 
#                    label=f'Best Period = {1/frequency[best_idx]:.2f} days')
#         plt.plot(frequency[best_idx], power[best_idx], 'ro', markersize=8)

#         for peak in peaks:
#             plt.axvline(x=peak, color='green', linestyle='--', alpha=0.5,
#                        label=f'Peak at {1/peak:.2f} days')


    
#     if auto is not True:
#         plt.xlim(0, 0.9)
    
#     if show_best:
#         plt.legend()
#     plt.axhline(y=false_alarm_level, color='grey', linestyle='--',
#                 label=f'False Alarm Level ({100*0.01:.1f}%): {false_alarm_level:.3e}')
#     plt.xlabel('Frequency (1/day)', fontsize=14)
#     plt.ylabel('Power', fontsize=14)
#     plt.tick_params(axis='both', which='major', direction ='in', labelsize=14)
#     plt.grid(True, alpha=0.3)
#     plt.title(f'{RA} {dec} periodogram', fontsize=14)
#     # plt.tight_layout()
#     # plt.savefig(f'lc_plots/{RA}{dec}_periodogramg.png', dpi=300, bbox_inches='tight')
#     # plt.show()