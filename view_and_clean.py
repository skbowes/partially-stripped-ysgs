### to use these functions, must import the following:
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

offsets_df = pd.read_csv('offsets.csv')
coords = pd.read_csv('merged_smc_lmc_coords.csv', comment='#', sep="\\s+", names=['RA', 'DEC'])


def get_telescope(image_str):
    """
    Extracts the telescope identifier from the image string.
    Parameters:
    image_str (str): The string containing the image identifier.
    Returns:
    str: The telescope identifier, or 'unknown' if not found.
    """
    # Extract the letter after 'b' in the image string
    try:
        b_index = image_str.index('b')
        if image_str[b_index + 1].isdigit():
            # If the next character is a digit, return 'unknown'
            return 'unknown'
        return image_str[b_index + 1]
    except:
        return 'unknown'
    
    
    
def find_header_line(file_path):
    """
    Finds the line number of the header in the file that starts with '###'.
    Parameters:
    file_path (str): The path to the file.
    Returns:
    int: The line number of the header line, or None if not found.
    """
    with open(file_path) as f:
        for i, line in enumerate(f):
            if line.startswith("###"):
                return i
            #for v2:
            if line.startswith("#") and not line.startswith("###"):
                return i+1
    return None



def modified_zscore(df, values, sigma=4.0, seeoutliers=False, report=False, v2 = False):
    #Returns the modified Z-score of all points in the array. This is defined as:
    #Zm = 0.6745 * (xi - median(xi))/MAD
    #where the MAD is the median avarage deviation defined as:
    #MAD = median(abs(xi - median(xi)))
    # xi is the array of values, like fluxes.
    """    
    Calculate the modified Z-score and median absolute deviation (MAD) for a given array of values.
    Args:
        values (array-like): Input array of values, usually fluxes.
        sigma (float): Threshold for identifying outliers. Default is 4.0.
        seeoutliers (bool): If True, do not remove outliers; if False, apply modified Z-score filtering.
        report (bool): If True, print the number of outliers and their fraction in the dataset.
    Returns:
        tuple: A tuple containing:
            - Zm (array): Modified Z-scores for each value.
            - MAD (float): Median absolute deviation of the values.
            - distances (array): Distances of each value from the median.
    """
    median = np.median(values)
    distances = values - median
    MAD = np.median(abs(distances))
    Zm = 0.6745 * distances / MAD
    outlier_bool=np.zeros(len(df))
    outlier_indices= []
    # print(df.columns)
    # print(df['flux_(mJy)'])
    # outliersindices = np.where(np.abs(Zm) >= sigma)[0]
    for j in range(len(values)):
    # for j in range(len(df['flux_(mJy)'])):
        # print(f'Index {j}, Zm = {Zm[j]}')
        # try:
        if np.abs(Zm[j]) >= sigma:
            idx = df.index[j]
            # print(idx)
            # print(f'Outlier detected at index {j} with Zm = {Zm[j]}')
            outlier_indices.append(idx)
            if seeoutliers==False:
                df.loc[idx, 'mag'] = np.nan
                df.loc[idx, 'mag_err'] = np.nan
                if v2 == False:
                    df.loc[idx, 'flux_(mJy)'] = np.nan
                    df.loc[idx, 'flux_err'] = np.nan
                if v2 == True:
                    df.loc[idx, 'flux'] = np.nan
                    df.loc[idx, 'flux_err'] = np.nan
                outlier_bool[j]=1.
        # except Exception as e:
        #     print(f"Error processing index {j}: {e}")
            # outlier_indices.append(j)
    if report==True and len(outlier_indices) > 0: 
        print(f'Number of outliers: {len(outlier_indices)}, Outlier fraction: {len(outlier_indices)/len(df)}')
    return df,Zm,MAD,distances,outlier_indices

def deg_to_hms_filenames(ra_deg):
    """Convert RA from decimal degrees to hours, minutes, seconds."""
    total_seconds = ra_deg / 15 * 3600
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = total_seconds % 60
    # print(h, m, s)
    s = int(s*100)
    if s < 1000:
    # s = int(s*10)
    # if s < 100:
        if s < 100:
            s1 = f"00{s}"
            s2 = f"00{s+1}"
        else:
            s1 = f"0{s}"
            s2 = f"0{s+1}"
    else:
        s1 = f"{s}"
        s2 = f"{s+1}"
    if m < 10:
        m = f"0{m}"
    if h < 10:
        h = f"0{h}"
    return f"{h}{m}{s1}", f"{h}{m}{s2}"

def deg_to_dms_filenames(dec_deg):
    """Convert Dec from decimal degrees to degrees, minutes, seconds."""
    sign = -1 if dec_deg < 0 else 1
    dec_deg = abs(dec_deg)
    d = int(dec_deg)
    m = int((dec_deg - d) * 60)
    s = (dec_deg - d - m / 60) * 3600
    # print(d, m, s)
    s = int(s*10)
    if s < 100:
        if s < 10:
            s1 = f"00{s}"
            s2 = f"00{s+1}"
        else:
            s1 = f"0{s}"
            s2 = f"0{s+1}"
    else:
        s1 = f"{s}"
        s2 = f"{s+1}"
    if m < 10:
        m = f"0{m}"
    if d < 10:
        if sign < 0:
            d = f"-0{d}"
        else:
            d = f"0{d}"
    if sign < 0:
        d = f"-{d}"
    return f"{d}{m}{s1}", f"{d}{m}{s2}"

def df_extract(index, g = True, seeoutliers=False, report=False):
    """
    Extracts data from a file based on the given index and whether to use 'g' or 'V' band data.
    Args:
        index (int): The index of the coordinates to use for file naming.
        g (bool): If True, use 'g' band data; if False, use 'V' band data.
        seeoutliers (bool): If True, do not remove outliers; if False, apply modified Z-score filtering.
        report (bool): If True, print the number of outliers and their fraction in the dataset.
    Returns:
        df (DataFrame): A pandas DataFrame containing the extracted data.
        telescopes (array): Unique telescopes found in the data.
    """

    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]
    # RA = coords[index]['RA']
    # dec = coords[index]['DEC']

    # new, works for directory sub2025, where new (2025) lcs are stored
    RA_hms = deg_to_hms_filenames(RA)
    dec_dms = deg_to_dms_filenames(dec)
    RA_hms1 = RA_hms[0]
    dec_dms1 = dec_dms[0]
    if g == True:
        file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms1}_g.dat')
        if not os.path.isfile(file_path):
            RA_hms2 = RA_hms[1]
            file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms1}_g.dat')
            if not os.path.isfile(file_path):
                dec_dms2 = dec_dms[1]
                file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms2}_g.dat')
                if not os.path.isfile(file_path):
                    file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms2}_g.dat')
                    if not os.path.isfile(file_path):
                        print(f"File not found for target {index}")
                        print(RA, dec)
    if g == False:
        file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms1}_V.dat')
        if not os.path.isfile(file_path):
            RA_hms2 = RA_hms[1]
            file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms1}_V.dat')
            if not os.path.isfile(file_path):
                dec_dms2 = dec_dms[1]
                file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms2}_V.dat')
                if not os.path.isfile(file_path):
                    file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms2}_V.dat')
                    if not os.path.isfile(file_path):
                        print(f"File not found for target {index}")
                        print(RA, dec)

    # Read the data
    header_line = find_header_line(file_path)
    if header_line is None:
        raise ValueError("Could not find header line starting with ###")

    # Read the header line to get column names
    with open(file_path) as f:
        for j, line in enumerate(f):
            if j == header_line:
                column_names = line.strip("# \n").replace("flux (mJy)", "flux_(mJy)").split()
                break
    # Read the data, skipping all lines before the header
    df = pd.read_csv(file_path, 
                    skiprows=header_line+1,  # Skip everything before and including the header
                    comment='#', 
                    sep="\\s+", 
                    names=column_names)
    
    # Modified Z-score filtering; If seeoutliers is True, we will not remove outliers
    if seeoutliers==False:
        df = modified_zscore(df, df['flux_(mJy)'], report=report)[0]

    # Drop rows with NaN values in any column
    df = df.dropna(axis=0, how='any')

    # Drop rows where flux_(mJy) is less than flux_err
    df = df[df['flux_(mJy)'] >= 3*df['flux_err']]

    # Drop rows where flux_err is zero
    df = df[df['flux_err'] > 0]

    # Where mag_err is < 0.01, we set it to 0.01, and accordingly set flux_err
    df.loc[df['mag_err'] < 0.01, 'mag_err'] = 0.01
    # for only the changed mag_err values, we need to recalculate flux_err
    df.loc[df['mag_err'] < 0.01, 'flux_err'] = df['mag_err'] * df['flux_(mJy)'] * np.log(10) / 2.5

    # Second filter: Drop rows where 'mag' originally had a '>' character
    df = df[~df['mag'].astype(str).str.startswith('>')]

    # Sometimes the 'mag' column is strings, so we need to convert it to numeric
    df = df[pd.to_numeric(df["mag"], errors="coerce").notna()]
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")

    # Reset the index of the DataFrame (most functions work without this, but it is good practice)
    df = df.reset_index(drop=True)

    df['telescope'] = df['IMAGE'].apply(get_telescope)
    telescopes = df['telescope'].unique()
    return df, telescopes

def df_extract_v2(index, filematcher_result = None, g = True, seeoutliers=False, report=False, v2 = False):
    """
    Extracts data from a file based on the given index and whether to use 'g' or 'V' band data.
    Args:
        index (int): The index of the coordinates to use for file naming.
        g (bool): If True, use 'g' band data; if False, use 'V' band data.
        seeoutliers (bool): If True, do not remove outliers; if False, apply modified Z-score filtering.
    Returns:
        df (DataFrame): A pandas DataFrame containing the extracted data.
        telescopes (array): Unique telescopes found in the data.
    """

    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]

    # new, works for directory sub2025, where new (2025) lcs are stored

    if v2 == True:
        file_path = os.path.join('sub2025_V2', f'{int(filematcher_result)}.csv')
        if not os.path.isfile(file_path):
            print(f"File not found for target {index}")
            print(RA, dec)
            raise FileNotFoundError(f"File not found: {file_path}")
        # else:
        #     print(f"File found for target {index}: {file_path}")
        #     print(RA, dec)
        #     raise FileNotFoundError(f"File not found: {file_path}")
        # return pd.read_csv(file_path), ['unknown']

    if v2 == False:
        RA_hms = deg_to_hms_filenames(RA)
        dec_dms = deg_to_dms_filenames(dec)
        RA_hms1 = RA_hms[0]
        dec_dms1 = dec_dms[0]
        if g == True:
            file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms1}_g.dat')
            if not os.path.isfile(file_path):
                RA_hms2 = RA_hms[1]
                file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms1}_g.dat')
                if not os.path.isfile(file_path):
                    dec_dms2 = dec_dms[1]
                    file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms2}_g.dat')
                    if not os.path.isfile(file_path):
                        file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms2}_g.dat')
                        if not os.path.isfile(file_path):
                            print(f"File not found for target {index}")
                            print(RA, dec)
        if g == False:
            file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms1}_V.dat')
            if not os.path.isfile(file_path):
                RA_hms2 = RA_hms[1]
                file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms1}_V.dat')
                if not os.path.isfile(file_path):
                    dec_dms2 = dec_dms[1]
                    file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms2}_V.dat')
                    if not os.path.isfile(file_path):
                        file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms2}_V.dat')
                        if not os.path.isfile(file_path):
                            print(f"File not found for target {index}")
                            print(RA, dec)

    # Read the data

    # header_line = find_header_line(file_path)
    if v2 == True:
        # V2 files are standard CSV format - just read them directly
        df = pd.read_csv(file_path, skiprows=1)  # Skip the comment line
        # No need for header processing or custom column names
    else:
        # Original logic for .dat files
        header_line = find_header_line(file_path)
        if header_line is None:
            raise ValueError("Could not find header line starting with ### or #")

        # Read the header line to get column names
        with open(file_path) as f:
            for j, line in enumerate(f):
                if j == header_line:
                    column_names = line.strip("# \n").replace("flux (mJy)", "flux").split()
                    break
        df = pd.read_csv(file_path, skiprows=header_line+1, comment='#', sep="\\s+", names=column_names)
    
    # Add flux_(mJy) column for V2 data compatibility
    if v2 == True and 'flux_(mJy)' not in df.columns:
        df['flux_(mJy)'] = df['flux']

    # Modified Z-score filtering; If seeoutliers is True, we will not remove outliers
    if seeoutliers==False:
            df = modified_zscore(df, df['flux'], report=report, v2=v2)[0]

    # Drop rows with NaN values in any column
    df = df.dropna(axis=0, how='any')

    # Drop rows where flux_(mJy) is less than flux_err
    df = df[df['flux'] >= 3*df['flux_err']]

    # Drop rows where flux_err is zero
    df = df[df['flux_err'] > 0]
    # Where mag_err is < 0.01, we set it to 0.01, and accordingly set flux_err to 

    # Second filter: Drop rows where 'mag' originally had a '>' character
    df = df[~df['mag'].astype(str).str.startswith('>')]

    # Sometimes the 'mag' column is strings, so we need to convert it to numeric
    df = df[pd.to_numeric(df["mag"], errors="coerce").notna()]
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")

    if v2==True and g==True:
        df = df[df['phot_filter'] == 'g']
        df = df[df['quality'] == 'G']
    if v2==True and g==False:
        df = df[df['phot_filter'] == 'V']
        df = df[df['quality'] == 'G']

    # Reset the index of the DataFrame (most functions work without this, but it is good practice)
    df = df.reset_index(drop=True)
    if v2 == False:
        df['telescope'] = df['IMAGE'].apply(get_telescope)
    if v2 == True:
        df['telescope'] = df['camera'].apply(get_telescope)
    telescopes = df['telescope'].unique()
    return df, telescopes





def individual_plotter(index, tail=1, g=True, seeoutliers=False, report=True):  
    """
    Plots the light curve for a given index, either in g-band or V-band.
    Args:
        index (int): Index of the object in the coordinates DataFrame.
        g (bool): If True, plot g-band light curve; if False, plot V-band light curve.
        seeoutliers (bool): If True, do not filter outliers; if False, apply modified Z-score filtering.
    """

    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']  # Different colors for different telescopes

    # df, telescopes = df_extract(index, g=g, seeoutliers=seeoutliers, report=report)
    df, telescopes = df_extract(index, g=g, seeoutliers=False, report=report)
    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]
    # RA = coords[index]['RA']
    # dec = coords[index]['DEC']

    # if seeoutliers==True:
    #     print('right before modified z-score')
    #     # df,Zm,MAD,distances,outlierindices = modified_zscore(df, df['flux_(mJy)'], seeoutliers=seeoutliers, report=report)  # Apply modified Z-score filter
    #     outlierindices = modified_zscore(df, df['flux_(mJy)'], seeoutliers=seeoutliers, report=report)[4]  # Get outlier indices
    #     print('right after modified z-score')
    if report:
        # Print summary of data points per telescope
        print("\nNumber of observations per telescope:")
        if g:
            print("g-band:")
        else:
            print("\nV-band:")
        print(df.groupby('telescope').size())

    # Plot data
    plt.figure(figsize=(12, 6))
    if g:
        for i, tel in enumerate(telescopes):
            mask = df['telescope'] == tel
            plt.errorbar(df[mask]['HJD'], df[mask]['mag'],  yerr=df[mask]['mag_err'],
                        fmt='o', markeredgecolor = 'black', capsize = 5, color=colors[i % len(colors)], zorder = 10,
                        label=f'g-band (telescope {tel})')
            
        if seeoutliers==True:
            # if len(outlierindices) > 0:
            #     for badindex in outlierindices:
            #             plt.plot(df.loc[badindex]['HJD'], df.loc[badindex]['mag'], 'x', color=colors[i % len(colors)], markersize=25, markeredgecolor='black', zorder=10)
            fulldata = df_extract(index, g=g, seeoutliers=True, report=report)[0]
            plt.errorbar(fulldata['HJD'], fulldata['mag'], yerr=fulldata['mag_err'],
                        fmt='.', color='black', markersize=10,capsize = 5, markeredgecolor='black',
                        label='Outliers')


    else:
        for i, tel in enumerate(telescopes):
            mask = df['telescope'] == tel
            plt.errorbar(df[mask]['HJD'], df[mask]['mag'],  yerr=df[mask]['mag_err'],
                        fmt='o', markeredgecolor = 'black', capsize = 5, color=colors[i % len(colors)],
                        label=f'V-band (telescope {tel})')

        if seeoutliers:
            # if len(outlierindices) > 0:
            #     for badindex in outlierindices:
            #             plt.plot(df.loc[badindex]['HJD'], df.loc[badindex]['mag'], 'x', color=colors[i % len(colors)], markersize=25, markeredgecolor='black', zorder=10)
            plt.errorbar(fulldata['HJD'], fulldata['mag'], yerr=fulldata['mag_err'],
                        fmt='.', color='black', markersize=10,capsize = 5, markeredgecolor='black')

    plt.xlabel('HJD')
    plt.ylabel('Magnitude')
    plt.title(f'{RA} {dec}')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invert y-axis since smaller magnitudes are brighter
    plt.tick_params(axis='both', direction='in')
    # plt.savefig(f'offsets_examples/{RA}{dec}_g.png')
    # plt.savefig(f'lc_plots/{RA}{dec}_g.png', bbox_inches='tight', dpi=300)
    plt.show()

def individual_plotter_v2(index, filematcher_result=None, tail=1, g=True, seeoutliers=False, report=True):  
    """
    Plots the light curve for a given index, either in g-band or V-band.
    Args:
        index (int): Index of the object in the coordinates DataFrame.
        g (bool): If True, plot g-band light curve; if False, plot V-band light curve.
        seeoutliers (bool): If True, do not filter outliers; if False, apply modified Z-score filtering.
    """

    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']  # Different colors for different telescopes

    # df, telescopes = df_extract(index, g=g, seeoutliers=seeoutliers, report=report)
    df, telescopes = df_extract_v2(index,filematcher_result = filematcher_result, g=g, seeoutliers=False, report=report, v2=True)
    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]
    # RA = coords[index]['RA']
    # dec = coords[index]['DEC']

    # if seeoutliers==True:
    #     print('right before modified z-score')
    #     # df,Zm,MAD,distances,outlierindices = modified_zscore(df, df['flux_(mJy)'], seeoutliers=seeoutliers, report=report)  # Apply modified Z-score filter
    #     outlierindices = modified_zscore(df, df['flux_(mJy)'], seeoutliers=seeoutliers, report=report)[4]  # Get outlier indices
    #     print('right after modified z-score')
    if report:
        # Print summary of data points per telescope
        print("\nNumber of observations per telescope:")
        if g:
            print("g-band:")
        else:
            print("\nV-band:")
        print(df.groupby('telescope').size())

    if filematcher_result is not None:
        days = 'jd'
    else:
        days = 'HJD'

    # Plot data
    plt.figure(figsize=(12, 6))
    if g:
        for i, tel in enumerate(telescopes):
            mask = df['telescope'] == tel
            plt.errorbar(df[mask][days], df[mask]['mag'],  yerr=df[mask]['mag_err'],
                        fmt='o', markeredgecolor = 'black', capsize = 5, color=colors[i % len(colors)], zorder = 10,
                        label=f'g-band (telescope {tel})')
            
        if seeoutliers==True:
            # if len(outlierindices) > 0:
            #     for badindex in outlierindices:
            #             plt.plot(df.loc[badindex]['HJD'], df.loc[badindex]['mag'], 'x', color=colors[i % len(colors)], markersize=25, markeredgecolor='black', zorder=10)
            fulldata = df_extract_v2(index, g=g, seeoutliers=True, report=report)[0]
            plt.errorbar(fulldata[days], fulldata['mag'], yerr=fulldata['mag_err'],
                        fmt='.', color='black', markersize=10,capsize = 5, markeredgecolor='black',
                        label='Outliers')


    else:
        for i, tel in enumerate(telescopes):
            mask = df['telescope'] == tel
            plt.errorbar(df[mask][days], df[mask]['mag'],  yerr=df[mask]['mag_err'],
                        fmt='o', markeredgecolor = 'black', capsize = 5, color=colors[i % len(colors)],
                        label=f'V-band (telescope {tel})')

        if seeoutliers:
            # if len(outlierindices) > 0:
            #     for badindex in outlierindices:
            #             plt.plot(df.loc[badindex]['HJD'], df.loc[badindex]['mag'], 'x', color=colors[i % len(colors)], markersize=25, markeredgecolor='black', zorder=10)
            plt.errorbar(fulldata[days], fulldata['mag'], yerr=fulldata['mag_err'],
                        fmt='.', color='black', markersize=10,capsize = 5, markeredgecolor='black')

    plt.xlabel('jd')
    plt.ylabel('Magnitude')
    plt.title(f'{RA} {dec}')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invert y-axis since smaller magnitudes are brighter
    plt.tick_params(axis='both', direction='in')
    # plt.savefig(f'offsets_examples/{RA}{dec}_g.png')
    # plt.savefig(f'lc_plots/{RA}{dec}_g.png', bbox_inches='tight', dpi=300)
    plt.show()



def individual_plotter_old(index, g=True, seeoutliers=False, report=True):  
    """
    Plots the light curve for a given index, either in g-band or V-band.
    Args:
        index (int): Index of the object in the coordinates DataFrame.
        g (bool): If True, plot g-band light curve; if False, plot V-band light curve.
        seeoutliers (bool): If True, do not filter outliers; if False, apply modified Z-score filtering.
    """

    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']  # Different colors for different telescopes

    df, telescopes = df_extract(index, g=g, seeoutliers=seeoutliers, report=report)
    # df, telescopes = df_extract(index, g=g, seeoutliers=False, report=report)
    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]

    if seeoutliers==True:
        print('right before modified z-score')
        # df,Zm,MAD,distances,outlierindices = modified_zscore(df, df['flux_(mJy)'], seeoutliers=seeoutliers, report=report)  # Apply modified Z-score filter
        outlierindices = modified_zscore(df, df['flux_(mJy)'], seeoutliers=seeoutliers, report=report)[4]  # Get outlier indices
        print('right after modified z-score')
    if report:
        # Print summary of data points per telescope
        print("\nNumber of observations per telescope:")
        if g:
            print("g-band:")
        else:
            print("\nV-band:")
        print(df.groupby('telescope').size())

    # Plot data
    plt.figure(figsize=(12, 8))
    if g:
        for i, tel in enumerate(telescopes):
            mask = df['telescope'] == tel
            plt.errorbar(df[mask]['HJD'], df[mask]['mag'],  yerr=df[mask]['mag_err'],
                        fmt='o', markeredgecolor = 'black', capsize = 5, color=colors[i % len(colors)], zorder = 10,
                        label=f'g-band (telescope {tel})')
        if seeoutliers==True:
            if len(outlierindices) > 0:
                for badindex in outlierindices:
                        plt.plot(df.loc[badindex]['HJD'], df.loc[badindex]['mag'], 'x', color=colors[i % len(colors)], markersize=25, markeredgecolor='black', zorder=10)
            # fulldata = df_extract(index, g=g, seeoutliers=True, report=report)[0]
            # plt.errorbar(fulldata['HJD'], fulldata['mag'], yerr=fulldata['mag_err'],
            #             fmt='x', color='black', markersize=10, markeredgecolor='black', zorder=1,
            #             label='Outliers')

    else:
        for i, tel in enumerate(telescopes):
            mask = df['telescope'] == tel
            plt.errorbar(df[mask]['HJD'], df[mask]['mag'],  yerr=df[mask]['mag_err'],
                        fmt='o', markeredgecolor = 'black', capsize = 5, color=colors[i % len(colors)],
                        label=f'V-band (telescope {tel})')
        if seeoutliers:
            if len(outlierindices) > 0:
                for badindex in outlierindices:
                        plt.plot(df.loc[badindex]['HJD'], df.loc[badindex]['mag'], 'x', color=colors[i % len(colors)], markersize=25, markeredgecolor='black', zorder=10)
    plt.xlabel('HJD')
    plt.ylabel('Magnitude')
    plt.title(f'{RA} {dec}')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invert y-axis since smaller magnitudes are brighter
    plt.tick_params(axis='both', direction='in')
    # plt.savefig(f'offsets_examples/{RA}{dec}_g.png')
    # plt.savefig(f'lc_plots/{RA}{dec}_g.png', bbox_inches='tight', dpi=300)
    plt.show()


def grid_plotter(starting_index, g = True):
    """
    Plots a 5x5 grid of light curves starting from the given index, either in g-band or V-band.
    Args:
        starting_index (int): The starting index of the object in the coordinates DataFrame.
        g (bool): If True, plot g-band light curves; if False, plot V-band light curves.
    """
    # Create a 5x5 grid of subplots
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    if g == True:
        fig.suptitle('g-band Light Curves for Multiple Objects', fontsize=16)
    if g == False:
        fig.suptitle('V-band Light Curves for Multiple Objects', fontsize=16)
    axes_flat = axes.flatten()
    # Colors for different telescopes
    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']

    plotnumber = 0
    for i in range(starting_index, starting_index + 25):
        ax = axes_flat[plotnumber]

        try:
            # Get coordinates for this object
            RA = coords['RA'][i]
            dec = coords['DEC'][i]

            df, telescopes = df_extract(i, g=g, seeoutliers=False, report=False)

            # Plot data for each telescope
            for j, tel in enumerate(telescopes):
                mask = df['telescope'] == tel
                ax.errorbar(df[mask]['HJD'], df[mask]['mag'], yerr=df[mask]['mag_err'], fmt='.', color=colors[j % len(colors)],
                        label=f'tel {tel}', capsize=1, markersize=2, elinewidth=0.3, capthick=0.3)
                # ax.set_ylim(ax.get_ylim()[::-1])  # Keep y-axis inverted, but you can set specific limits like:
                # ax.set_ylim(14.5, 13.0)  # Example: set y-axis from 16 (faint) to 12 (bright)
            
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            if RA < 10:
                ax.set_title(f'J0{RA}{dec}' + " Number " + str(i), fontsize=8)
            if RA >= 10:
                ax.set_title(f'J{RA}{dec}' + " Number " + str(i), fontsize=8)
            if len(telescopes) > 1:
                ax.legend(fontsize=6)
            ax.tick_params(axis='both', labelsize=6)
            
        except Exception as e:
            # If there's an error reading/plotting a file, show empty plot with error message
            ax.text(0.5, 0.5, f'Error: {str(e)}', 
                    ha='center', va='center', transform=ax.transAxes, 
                    fontsize=6, color='red')
            ax.set_title(f'J0{RA}{dec}'+ " Number " + str(i), fontsize=8)
        plotnumber += 1
    fig.text(0.5, 0.02, 'HJD', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Magnitude', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95) 
    plt.show()

def df_extract_pre_skypatrolv2(index, g = True, seeoutliers=False, report=False):
    """
    Extracts data from a file based on the given index and whether to use 'g' or 'V' band data.
    Args:
        index (int): The index of the coordinates to use for file naming.
        g (bool): If True, use 'g' band data; if False, use 'V' band data.
        seeoutliers (bool): If True, do not remove outliers; if False, apply modified Z-score filtering.
    Returns:
        df (DataFrame): A pandas DataFrame containing the extracted data.
        telescopes (array): Unique telescopes found in the data.
    """

    RA = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]

    # new, works for directory sub2025, where new (2025) lcs are stored
    RA_hms = deg_to_hms_filenames(RA)
    dec_dms = deg_to_dms_filenames(dec)
    RA_hms1 = RA_hms[0]
    dec_dms1 = dec_dms[0]
    if g == True:
        file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms1}_g.dat')
        if not os.path.isfile(file_path):
            RA_hms2 = RA_hms[1]
            file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms1}_g.dat')
            if not os.path.isfile(file_path):
                dec_dms2 = dec_dms[1]
                file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms2}_g.dat')
                if not os.path.isfile(file_path):
                    file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms2}_g.dat')
                    if not os.path.isfile(file_path):
                        print(f"File not found for target {index}")
                        print(RA, dec)
    if g == False:
        file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms1}_V.dat')
        if not os.path.isfile(file_path):
            RA_hms2 = RA_hms[1]
            file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms1}_V.dat')
            if not os.path.isfile(file_path):
                dec_dms2 = dec_dms[1]
                file_path = os.path.join('sub2025', f'{RA_hms1}{dec_dms2}_V.dat')
                if not os.path.isfile(file_path):
                    file_path = os.path.join('sub2025', f'{RA_hms2}{dec_dms2}_V.dat')
                    if not os.path.isfile(file_path):
                        print(f"File not found for target {index}")
                        print(RA, dec)

    # Read the data
    header_line = find_header_line(file_path)
    if header_line is None:
        raise ValueError("Could not find header line starting with ###")

    # Read the header line to get column names
    with open(file_path) as f:
        for j, line in enumerate(f):
            if j == header_line:
                column_names = line.strip("# \n").replace("flux (mJy)", "flux_(mJy)").split()
                break
    # Read the data, skipping all lines before the header
    df = pd.read_csv(file_path, 
                    skiprows=header_line+1,  # Skip everything before and including the header
                    comment='#', 
                    sep="\\s+", 
                    names=column_names)
    
    # Modified Z-score filtering; If seeoutliers is True, we will not remove outliers
    if seeoutliers==False:
        df = modified_zscore(df, df['flux_(mJy)'], report=report)[0]

    # Drop rows with NaN values in any column
    df = df.dropna(axis=0, how='any')

    # Drop rows where flux_(mJy) is less than flux_err
    df = df[df['flux_(mJy)'] >= 3*df['flux_err']]

    # Drop rows where flux_err is zero
    df = df[df['flux_err'] > 0]
    # Where mag_err is < 0.01, we set it to 0.01, and accordingly set flux_err to 

    # Second filter: Drop rows where 'mag' originally had a '>' character
    df = df[~df['mag'].astype(str).str.startswith('>')]

    # Sometimes the 'mag' column is strings, so we need to convert it to numeric
    df = df[pd.to_numeric(df["mag"], errors="coerce").notna()]
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")

    # Reset the index of the DataFrame (most functions work without this, but it is good practice)
    df = df.reset_index(drop=True)

    df['telescope'] = df['IMAGE'].apply(get_telescope)
    telescopes = df['telescope'].unique()
    return df, telescopes





########################## CORRECTING OFFSETS

def telescope_separator(target, column_name):
    """Separates string of telescopes from offsets.csv.
    Args:
        target (int): Index of the target in the coordinates DataFrame.
        column_name (str): Name of the column in offsets.csv to extract (e.g., 'discard_telescopes').
    Returns:
        list: List of individual telescopes to discard.
    """
    ra = coords['RA'].iloc[target]
    dec = coords['DEC'].iloc[target]
    # telescopes = offsets_df[column_name][(offsets_df['ra'] == ra) & (offsets_df['dec'] == dec)].iloc[0]
    matching_rows = offsets_df[(offsets_df['ra'] == ra) & (offsets_df['dec'] == dec)]
    if matching_rows.empty:
        return []  # Return empty list if target not found
    telescopes = matching_rows[column_name].iloc[0]
    # split each letter of the string
    if isinstance(telescopes, str):
        telescopes = list(telescopes) # need to add the other characters of the telescope
    else:
        telescopes = []
    # print(telescopes)
    return telescopes

def offset_corrector_window(index, g=True, additive=False, mag_space=False, show=False):
    """Corrects offsets in the light curve for a given target index.
    This only works for the g band as of now, due to the hard-coded time ranges.
    To change this, we need to add time ranges for V band. We would also need to change the df_extract function to extract V band data.
    We would also need to create a new offsets.csv file for V band, with the same format as the current one, to discard telescopes.
    Args:
        target (int): Index of the target in the coordinates DataFrame.
        additive (bool): If True, apply additive correction; if False, apply multiplicative correction.
        mag_space (bool): If True, work in magnitude space; if False, work in flux space.
        show (bool): If True, plot the light curve before and after correction.
    Returns:
        df_window (DataFrame): DataFrame with corrected light curve data.
    """
    ra = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]
    all_offsets = {}
    all_means = {}
    df_window, teles = df_extract(index, g=g)
    df_window['telescope_from_image'] = df_window['IMAGE'].apply(get_telescope)
    offset_teles = []
    discard_teles = telescope_separator(index, 'discard_telescopes')

    for t in teles:
        if t not in discard_teles:
            offset_teles.append(t)
    if not offset_teles:
        return None
    
    baseline_telescope = None
    for tele in df_window['telescope_from_image'].unique():
        if tele not in discard_teles and tele is not None:
            baseline_telescope = tele
            break
    # print(f"Baseline telescope: {baseline_telescope}")
    try:
        offsets_df_row = offsets_df[(offsets_df['ra'] == ra) & (offsets_df['dec'] == dec)].iloc[0]
        time_start = offsets_df_row['time_start']
        time_end = offsets_df_row['time_end']
        
        # Check if time_start or time_end are NaN/None and use fallback values
        if pd.isna(time_start) or pd.isna(time_end) or time_start is None or time_end is None:
            if index < 377:
                time_start = 2459325.0
                time_end = 2459650.0
            else:
                time_start = 2459400.0
                time_end = 2459725.0
    except:
        if index < 377:
            time_start = 2459325.0
            time_end = 2459650.0
        else:
            time_start = 2459400.0
            time_end = 2459725.0

    df_window = df_window[(df_window['HJD'] >= time_start) & (df_window['HJD'] <= time_end)]

    if show:
        fig, ax1 = plt.subplots(figsize=(18, 6))
    # going through each telescope and plotting
        for j, tele in enumerate(teles):
            colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
            mask = df_window['telescope_from_image'] == tele
            if mag_space ==False:
                mean = np.mean(df_window[mask]['flux_(mJy)'])
                ax1.errorbar(df_window[mask]['HJD'], df_window[mask]['flux_(mJy)'], yerr=df_window[mask]['flux_err'],
                            color=colors[j % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {tele}')
                ax1.axhline(y=mean, color=colors[j % len(colors)], linestyle='--', label=f'Mean {tele}')
            if mag_space == True:
                mean = np.mean(df_window[mask]['mag'])
                ax1.errorbar(df_window[mask]['HJD'], df_window[mask]['mag'], yerr=df_window[mask]['mag_err'],
                            color=colors[j % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {tele}')
                ax1.axhline(y=mean, color=colors[j % len(colors)], linestyle='--', label=f'Mean {tele}')
    # if show:
        ax1.set_xlabel('HJD', fontsize=14)
        if mag_space == False:
            ax1.set_ylabel('Flux (mJy)', fontsize=14)
        if mag_space == True:
            ax1.set_ylabel('Magnitude', fontsize=14)
            plt.gca().invert_yaxis() 
        ax1.legend(loc='upper left')
        ax1.grid(True)
        plt.title(f'{ra} {dec} Uncorrected window')
        plt.show()
        plt.close(fig)
    # going thru each offset telescope and calculating the mean offset
    for telescope in offset_teles:
        df_telescope = df_window[df_window['telescope_from_image'] == telescope]
        df_baseline = df_window[df_window['telescope_from_image'] == baseline_telescope]
        if mag_space == False:
            telescope_flux = df_telescope['flux_(mJy)'].values
            baseline_flux = df_baseline['flux_(mJy)'].values
            df_telescope_mean = np.mean(telescope_flux)
            df_baseline_mean = np.mean(baseline_flux)
        if mag_space == True:
            telescope_mag = df_telescope['mag'].values
            baseline_mag = df_baseline['mag'].values
            df_telescope_mean = np.mean(telescope_mag)
            df_baseline_mean = np.mean(baseline_mag) 

        if additive:
            mean_offset = df_baseline_mean - df_telescope_mean
            # print(f'Telescope: {telescope}, Mean Offset: {mean_offset}')
            # Apply the offset correction to the dataframe
            mask = df_window['telescope_from_image'] == telescope
            if mag_space == False:
                df_window.loc[mask, 'flux_(mJy)'] += mean_offset
            if mag_space == True:
                df_window.loc[mask, 'mag'] += mean_offset 
        else:
            mean_offset = df_baseline_mean / df_telescope_mean
            # print(f'Telescope: {telescope}, Mean Offset: {mean_offset}')
            # Apply the offset correction to the dataframe
            mask = df_window['telescope_from_image'] == telescope
            if mag_space == False:
                df_window.loc[mask, 'flux_(mJy)'] *= mean_offset
            if mag_space == True:
                df_window.loc[mask, 'mag'] *= mean_offset 
        all_means[telescope] = mean_offset
        all_offsets[telescope] = mean_offset

    if show:
        fig, ax1 = plt.subplots(figsize=(18, 6))
    # going through each telescope and plotting
        for i, t in enumerate(teles):
            # colors = cm.get_cmap('tab10').colors
            mask = df_window['telescope_from_image'] == t
            if mag_space == False:
                mean = np.mean(df_window[mask]['flux_(mJy)'])
                ax1.errorbar(df_window[mask]['HJD'], df_window[mask]['flux_(mJy)'], yerr=df_window[mask]['flux_err'],
                            color=colors[i % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {t}')
                ax1.axhline(y=mean, color=colors[i % len(colors)], linestyle='--', label=f'Mean {t}')
            if mag_space == True:
                mean = np.mean(df_window[mask]['mag'])
                ax1.errorbar(df_window[mask]['HJD'], df_window[mask]['mag'], yerr=df_window[mask]['mag_err'],
                            color=colors[i % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {t}')
                ax1.axhline(y=mean, color=colors[i % len(colors)], linestyle='--', label=f'Mean {t}') 
        ax1.set_xlabel('HJD', fontsize=14)
        if mag_space == False:
            ax1.set_ylabel('Flux (mJy)', fontsize=14)
        if mag_space == True:
            ax1.set_ylabel('Magnitude', fontsize=14)
            plt.gca().invert_yaxis()
        ax1.legend(loc='upper left')
        ax1.grid(True)
        plt.title(f'{ra} {dec} Corrected window, showing discarded telescopes')
        plt.show()
        plt.close(fig)
    return all_offsets


def offset_corrector(index, g=True, additive=False, mag_space=False, show=True, show_window=False):
    """Corrects offsets in the light curve for a given target index.
    This only works for the g band as of now, due to the hard-coded time ranges.
    To change this, we need to add time ranges for V band. We would also need to change the df_extract function to extract V band data.
    We would also need to create a new offsets.csv file for V band, with the same format as the current one, to discard telescopes.
    Args:
        target (int): Index of the target in the coordinates DataFrame.
        additive (bool): If True, apply additive correction; if False, apply multiplicative correction.
        mag_space (bool): If True, work in magnitude space; if False, work in flux space.
        show (bool): If True, plot the light curve before and after correction.
        show_window (bool): If True, plot the window used for offset calculation.
    Returns:
        df (DataFrame): DataFrame with corrected light curve data.
    """
    # from metrics import mean_med_flux  # Import here to avoid circular import
    
    ra = coords['RA'].iloc[index]
    dec = coords['DEC'].iloc[index]
    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
    df, teles = df_extract(index, g=g)
    df['telescope_from_image'] = df['IMAGE'].apply(get_telescope)
    # calculate the zeropoints for each measurement so that we can convert between flux and mag space later
    df['zeropoints'] = df['flux_(mJy)'] / (10**(-0.4 * df['mag']))

    offsets = offset_corrector_window(index, g=g, additive=additive, mag_space=mag_space, show=False)
    if show:
        fig, ax1 = plt.subplots(figsize=(18, 6))
        for i, t in enumerate(teles):
            mask = df['telescope_from_image'] == t
            if mag_space == False:
                mean = np.mean(df[mask]['flux_(mJy)'])
                ax1.errorbar(df[mask]['HJD'], df[mask]['flux_(mJy)'], yerr=df[mask]['flux_err'],
                            color=colors[i % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {t}')
            if mag_space == True:
                mean = np.mean(df[mask]['mag'])
                ax1.errorbar(df[mask]['HJD'], df[mask]['mag'], yerr=df[mask]['mag_err'],
                            color=colors[i % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {t}')
            ax1.axhline(y=mean, color=colors[i % len(colors)], linestyle='--', label=f'Mean {t}')
        # this is to show the window that the offsets will be calculated in!
        if offsets is not None:
            # DEPENDENT ON THE LENGTH OF SMC/LMC
            # if index <= 376:
            #     ax1.axvline(time_start, color='orange', linestyle='dashed')
            #     ax1.axvline(time_end, color='orange', linestyle='dashed')
            if index > 376:
                ax1.axvline(2459400.00, color='green', linestyle='dashed')
                ax1.axvline(2459725.00, color='green', linestyle='dashed') 
        ax1.set_xlabel('HJD', fontsize=14)
        if mag_space == False:
            ax1.set_ylabel('Flux (mJy)', fontsize=14)
        if mag_space == True:
            ax1.set_ylabel('Magnitude', fontsize=14)
            plt.gca().invert_yaxis()
        ax1.legend(loc='upper left')
        ax1.grid(True)
        plt.title(f'{ra} {dec} Uncorrected lightcurve')
        plt.show()
        plt.close(fig)

    # Calculate offsets again just so we can show plots in the right order
    offsets = offset_corrector_window(index, g=g, additive=additive, mag_space=mag_space, show=show_window)
    # correct offsets
    if offsets: # was previously 'if offsets is not None and offsets:'
        for telescope, offset in offsets.items():
            # use the offsets to correct the data for all times:
            if additive:
                if mag_space == False:
                    df.loc[df['telescope_from_image'] == telescope, 'flux_(mJy)'] += offset
                if mag_space == True:
                    df.loc[df['telescope_from_image'] == telescope, 'mag'] += offset
            else:
                if mag_space == False:
                    df.loc[df['telescope_from_image'] == telescope, 'flux_(mJy)'] *= offset
                if mag_space == True:
                    df.loc[df['telescope_from_image'] == telescope, 'mag'] *= offset

    ##### SHOULD WORK REGARDLESS OF FLUX OR MAG SPACE! SINCE DROPPING THE WHOLE ROW> LEAVING IN FLUX SPACE FOR NOW
    # discard telescopes (non-constant offsets)
    discard_teles = telescope_separator(index, 'discard_telescopes')
    for discard_tele in discard_teles:
        df.loc[df['telescope_from_image'] == discard_tele, 'flux_(mJy)'] = np.nan

    # discard lightcurve (poor-quality data in all telescopes)
    try:
        if offsets_df['discard_lc'][(offsets_df['ra'] == ra) & (offsets_df['dec'] == dec)].iloc[0] == 1:
            df['flux_(mJy)'] = np.nan
    except:
        pass

    if mag_space == True:
        df['flux_(mJy)'] = df['zeropoints'] * (10**(-0.4 * df['mag']))
    if mag_space == False:
        df['mag'] = -2.5 * np.log10(df['flux_(mJy)'] / df['zeropoints'])

    # REMOVE OUTLIERS
    df = modified_zscore(df, df['flux_(mJy)'], report=False)[0]
    # DROP NaNs
    df = df.dropna(axis=0, how='any')
    df = df.reset_index(drop=True)
    telescopes = df['telescope_from_image'].unique()

    # #for a test case, only use one of the remaining telescopes
    # df = df[df['telescope_from_image'] == telescopes[0]]

    if show:
        fig, ax1 = plt.subplots(figsize=(18, 6))
        for h, te in enumerate(teles):
            mask = df['telescope_from_image'] == te
            if mag_space == False:
                mean = np.mean(df[mask]['flux_(mJy)'])
                ax1.errorbar(df[mask]['HJD'], df[mask]['flux_(mJy)'], yerr=df[mask]['flux_err'],
                            color=colors[h % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {te}')
            if mag_space == True:
                mean = np.mean(df[mask]['mag'])
                ax1.errorbar(df[mask]['HJD'], df[mask]['mag'], yerr=df[mask]['mag_err'],
                            color=colors[h % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {te}')
            ax1.axhline(y=mean, color=colors[h % len(colors)], linestyle='--', label=f'Mean {te}')

        # mean_med = mean_med_flux(index, df=df, telescopes=telescopes, g=g)
        # only do if lightcurve is not discarded
        if not df['flux_(mJy)'].isnull().all():
            points = df['flux_(mJy)'].values
            tail = 1
            if mag_space == True:
                points = df['mag'].values
            lower_percentile = np.percentile(points, 0+tail)
            upper_percentile = np.percentile(points, 100-tail)
            ax1.axhline(lower_percentile, color='red', linestyle='--', label=f'{tail}% percentile')
            ax1.axhline(upper_percentile, color='red', linestyle='--')

        ax1.set_xlabel('HJD', fontsize=14)
        if mag_space == False:
            ax1.set_ylabel('Flux (mJy)', fontsize=14)
        if mag_space == True:
            ax1.set_ylabel('Magnitude', fontsize=14)
            plt.gca().invert_yaxis()
        ax1.legend(loc='upper left')
        ax1.grid(True)
        plt.title(f'{ra} {dec} Corrected lightcurve')
        plt.savefig(f'lc_plots_2025/{ra}{dec}_lcg.png', bbox_inches='tight', dpi=300)
        plt.show()
        plt.close(fig)

    return df, telescopes





def offset_corrector_window_old(target, additive=False, mag_space =False, show=False):
    """Correct offsets based on a specific time window. Telescopes are corrected only if they are listed in the offsets.csv file under 'offset_telescopes'."""
    ra = coords['RA'].iloc[target]
    dec = coords['DEC'].iloc[target]
    all_offsets = {}
    all_means = {}
    df_window, teles = df_extract(target, g=True)

    # Add telescope column to dataframe
    df_window['telescope_from_image'] = df_window['IMAGE'].apply(get_telescope)

    offset_teles = telescope_separator(target, 'offset_telescopes')
    discard_teles = telescope_separator(target, 'discard_telescopes')
    if show==True:
        print(f"Offset telescopes for target {target}: {offset_teles}")
    if not offset_teles:
        # raise ValueError(f"No offset data found for telescope: {target}")
        # raise Warning(f"No offset data found for telescope: {target}")
        return None
    
    baseline_telescope = None
    for tele in df_window['telescope_from_image'].unique():
        if tele not in offset_teles and tele not in discard_teles and tele is not None:
            baseline_telescope = tele
            break
    print(f"Baseline telescope: {baseline_telescope}")
    if target < 377:
        time_start = 2459325.0
        time_end = 2459650.0
    if target >= 377:
        time_start = 2459400.0
        time_end = 2459725.0
    # time_start = offsets_df['time_start'][(offsets_df['ra'] == ra) & (offsets_df['dec'] == dec)].iloc[0]
    # time_end = offsets_df['time_end'][(offsets_df['ra'] == ra) & (offsets_df['dec'] == dec)].iloc[0]
    df_window = df_window[(df_window['HJD'] >= time_start) & (df_window['HJD'] <= time_end)]

    if show:
        fig, ax1 = plt.subplots(figsize=(18, 6))
    # going through each telescope and plotting
        for j, tele in enumerate(teles):
            colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
            mask = df_window['telescope_from_image'] == tele
            if mag_space ==False:
                mean = np.mean(df_window[mask]['flux_(mJy)'])
                ax1.errorbar(df_window[mask]['HJD'], df_window[mask]['flux_(mJy)'], yerr=df_window[mask]['flux_err'],
                            color=colors[j % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {tele}')
                ax1.axhline(y=mean, color=colors[j % len(colors)], linestyle='--', label=f'Mean {tele}')
            if mag_space == True:
                mean = np.mean(df_window[mask]['mag'])
                ax1.errorbar(df_window[mask]['HJD'], df_window[mask]['mag'], yerr=df_window[mask]['mag_err'],
                            color=colors[j % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {tele}')
                ax1.axhline(y=mean, color=colors[j % len(colors)], linestyle='--', label=f'Mean {tele}')

    # if show:
        ax1.set_xlabel('HJD', fontsize=14)
        if mag_space == False:
            ax1.set_ylabel('Flux (mJy)', fontsize=14)
        if mag_space == True:
            ax1.set_ylabel('Magnitude', fontsize=14)
            plt.gca().invert_yaxis()
        ax1.legend(loc='upper left')
        ax1.grid(True)
        plt.title(f'{ra} {dec} Uncorrected window')
        plt.show()
        plt.close(fig)
    # going thru each offset telescope and calculating the mean offset
    for telescope in offset_teles:
        df_telescope = df_window[df_window['telescope_from_image'] == telescope]
        df_baseline = df_window[df_window['telescope_from_image'] == baseline_telescope]
        if mag_space == False:
            telescope_flux = df_telescope['flux_(mJy)'].values
            baseline_flux = df_baseline['flux_(mJy)'].values
            df_telescope_mean = np.mean(telescope_flux)
            df_baseline_mean = np.mean(baseline_flux)
        if mag_space == True:
            telescope_mag = df_telescope['mag'].values
            baseline_mag = df_baseline['mag'].values
            df_telescope_mean = np.mean(telescope_mag)
            df_baseline_mean = np.mean(baseline_mag) 

        if additive:
            mean_offset = df_baseline_mean - df_telescope_mean
            # print(f'Telescope: {telescope}, Mean Offset: {mean_offset}')
            # Apply the offset correction to the dataframe
            mask = df_window['telescope_from_image'] == telescope
            if mag_space == False:
                df_window.loc[mask, 'flux_(mJy)'] += mean_offset
            if mag_space == True:
                df_window.loc[mask, 'mag'] += mean_offset 
            # not going to change the errors for now
        else:
            mean_offset = df_baseline_mean / df_telescope_mean
            # print(f'Telescope: {telescope}, Mean Offset: {mean_offset}')
            # Apply the offset correction to the dataframe
            mask = df_window['telescope_from_image'] == telescope
            if mag_space == False:
                df_window.loc[mask, 'flux_(mJy)'] *= mean_offset
            if mag_space == True:
                df_window.loc[mask, 'mag'] *= mean_offset 
        all_means[telescope] = mean_offset
        all_offsets[telescope] = mean_offset
        # not going to change the errors for now

    if show:
        fig, ax1 = plt.subplots(figsize=(18, 6))
    # going through each telescope and plotting
        for i, t in enumerate(teles):
            # colors = cm.get_cmap('tab10').colors
            mask = df_window['telescope_from_image'] == t
            if mag_space == False:
                mean = np.mean(df_window[mask]['flux_(mJy)'])
                ax1.errorbar(df_window[mask]['HJD'], df_window[mask]['flux_(mJy)'], yerr=df_window[mask]['flux_err'],
                            color=colors[i % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {t}')
                ax1.axhline(y=mean, color=colors[i % len(colors)], linestyle='--', label=f'Mean {t}')
            if mag_space == True:
                mean = np.mean(df_window[mask]['mag'])
                ax1.errorbar(df_window[mask]['HJD'], df_window[mask]['mag'], yerr=df_window[mask]['mag_err'],
                            color=colors[i % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {t}')
                ax1.axhline(y=mean, color=colors[i % len(colors)], linestyle='--', label=f'Mean {t}') 
        ax1.set_xlabel('HJD', fontsize=14)
        if mag_space == False:
            ax1.set_ylabel('Flux (mJy)', fontsize=14)
        if mag_space == True:
            ax1.set_ylabel('Magnitude', fontsize=14)
            plt.gca().invert_yaxis()
        ax1.legend(loc='upper left')
        ax1.grid(True)
        plt.title(f'{ra} {dec} Corrected window, showing discarded telescopes')
        plt.show()
        plt.close(fig)
    return all_offsets


def offset_corrector_old(target, additive=False, mag_space=False, show=True, show_window=False):
    """Corrects offsets based on a specific time window. Telescopes are corrected only if they are listed in the offsets.csv file under 'offset_telescopes'.
    
    Could implement this by changing every function that starts with df, teles = df_extract 
    to df, teles = offset_corrector. """

    ra = coords['RA'].iloc[target]
    dec = coords['DEC'].iloc[target]
    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']
    df, teles = df_extract(target, g=True)
    # print(df)
    df['telescope_from_image'] = df['IMAGE'].apply(get_telescope)
    offsets = offset_corrector_window_old(target, additive=additive, mag_space=mag_space, show=False)
    if show:
        fig, ax1 = plt.subplots(figsize=(18, 6))
        for i, t in enumerate(teles):
            mask = df['telescope_from_image'] == t
            if mag_space == False:
                mean = np.mean(df[mask]['flux_(mJy)'])
                ax1.errorbar(df[mask]['HJD'], df[mask]['flux_(mJy)'], yerr=df[mask]['flux_err'],
                            color=colors[i % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {t}')
            if mag_space == True:
                mean = np.mean(df[mask]['mag'])
                ax1.errorbar(df[mask]['HJD'], df[mask]['mag'], yerr=df[mask]['mag_err'],
                            color=colors[i % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {t}') 
            ax1.axhline(y=mean, color=colors[i % len(colors)], linestyle='--', label=f'Mean {t}')
        # this is to show the window that the offsets will be calculated in!
        if offsets is not None:
            # DEPENDENT ON THE LENGTH OF SMC/LMC
            if target <= 376:
                ax1.axvline(2459325.00, color='orange', linestyle='dashed')
                ax1.axvline(2459650.00, color='orange', linestyle='dashed')
            if target > 376:
                ax1.axvline(2459400.00, color='green', linestyle='dashed')
                ax1.axvline(2459725.00, color='green', linestyle='dashed') 
        ax1.set_xlabel('HJD', fontsize=14)
        if mag_space == False:
            ax1.set_ylabel('Flux (mJy)', fontsize=14)
        if mag_space == True:
            ax1.set_ylabel('Magnitude', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True)
        plt.title(f'{ra} {dec} Uncorrected lightcurve')
        plt.show()
        plt.close(fig)
    # check:
    # df['telescope_from_image'] = df['IMAGE'].apply(extract_telescope_from_image)

    # Calculate offsets again just so we can show plots in the right order
    offsets = offset_corrector_window_old(target, additive=additive, mag_space=mag_space, show=show_window)
    # correct offsets
    if offsets is not None or offsets:
        for telescope, offset in offsets.items():
            # use the offsets to correct the data for all times:
            if additive:
                if mag_space == False:
                    df.loc[df['telescope_from_image'] == telescope, 'flux_(mJy)'] += offset
                if mag_space == True:
                    df.loc[df['telescope_from_image'] == telescope, 'mag'] += offset
            else:
                if mag_space == False:
                    df.loc[df['telescope_from_image'] == telescope, 'flux_(mJy)'] *= offset
                if mag_space == True:
                    df.loc[df['telescope_from_image'] == telescope, 'mag'] *= offset

    ##### SHOULD WORK  REGARDLESS OF FLUX OR MAG SPACE! SINCE DROPPING THE WHOLE ROW> LEAVING IN FLUX SPACE FOR NOW
    # discard telescopes (non-constant offsets)
    discard_teles = telescope_separator(target, 'discard_telescopes')
    for discard_tele in discard_teles:
        df.loc[df['telescope_from_image'] == discard_tele, 'flux_(mJy)'] = np.nan
    # discard lightcurve (poor-quality data in all telescopes)
    if offsets_df['discard_lc'][(offsets_df['ra'] == ra) & (offsets_df['dec'] == dec)].iloc[0] == 1:
        df['flux_(mJy)'] = np.nan
    # drop nans
    df = df.dropna(axis=0, how='any')

    if show:
        fig, ax1 = plt.subplots(figsize=(18, 6))
        for h, te in enumerate(teles):
            mask = df['telescope_from_image'] == te
            if mag_space == False:
                mean = np.mean(df[mask]['flux_(mJy)'])
                ax1.errorbar(df[mask]['HJD'], df[mask]['flux_(mJy)'], yerr=df[mask]['flux_err'],
                            color=colors[h % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {te}')
            if mag_space == True:
                mean = np.mean(df[mask]['mag'])
                ax1.errorbar(df[mask]['HJD'], df[mask]['mag'], yerr=df[mask]['mag_err'],
                            color=colors[h % len(colors)], capsize=5, markeredgecolor='black', fmt='o', label=f'Telescope {te}')
            ax1.axhline(y=mean, color=colors[h % len(colors)], linestyle='--', label=f'Mean {te}')
        ax1.set_xlabel('HJD', fontsize=14)
        if mag_space == False:
            ax1.set_ylabel('Flux (mJy)', fontsize=14)
        if mag_space == True:
            ax1.set_ylabel('Magnitude', fontsize=14) 
        ax1.legend(loc='upper left')
        ax1.grid(True)
        plt.title(f'{ra} {dec} Corrected lightcurve')
        plt.show()
        plt.close(fig)
    return df