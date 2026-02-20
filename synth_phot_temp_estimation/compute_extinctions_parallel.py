"""
Compute extinction values for YSG candidates using parallel processing.

This script calculates E(B-V) and A_V values from multiple dust maps:
- Edenhofer 2023 3D dust map
- Schlegel, Finkbeiner & Davis (SFD) map with Schlafly & Finkbeiner 2011 recalibration
- Zaritsky et al. extinction map for LMC/SMC

Uses multiprocessing to speed up computation for large catalogs.
"""

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from zh_scraper import get_extinction
from dustmaps.sfd import SFDQuery
from dustmaps.edenhofer2023 import Edenhofer2023Query
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Initialize dust map queries
sfd = SFDQuery()
eden = Edenhofer2023Query(integrated=True)


def get_ebv(ra, dec):
    """
    Collects E(B-V) values for a star location. Produces a 'minimum' E(B-V)
    from the Edenhofer 23 3D map, and a 'mean' E(B-V) from the SFD map,
    modified with the recalibration from SF 2011. 

    Parameters:
        ra - Right Ascension in decimal degrees
        dec - Declination in decimal degrees

    Returns:
        ebv_eden - Minimum E(B-V) from Edenhofer 23
        ebv_sf - 'Average' E(B-V) from SFD with the SF alteration
    """ 
    coords_eden = SkyCoord(ra*u.degree, dec*u.degree,
                          distance=1.24*u.kpc, frame='icrs')  # Distance is max distance for this map
    ebv_eden = eden(coords_eden)

    coords_sf = SkyCoord(ra*u.degree, dec*u.degree, frame='icrs')
    ebv_sfd = sfd(coords_sf)
    ebv_sf = 0.86 * ebv_sfd  # To fix from the older SFD value to the newer SF value

    return ebv_eden, ebv_sf


def process_single_star(args):
    """
    Worker function to process a single star's extinction values.
    Designed to be called in parallel.
    
    Parameters:
        args - tuple of (index, ra, dec, ra_hours, dec_hours)
    
    Returns:
        dict with index and all computed values
    """
    i, ra, dec, ra_hours, dec_hours = args
    
    # Rv values
    rv_gal = 3.17
    rv_smc = 3.02
    rv_lmc = 3.41
    set_radius = 10  # radius in arcmin for Zaritsky query
    
    # Get E(B-V) values
    try:
        ebv_eden_t, ebv_sf_t = get_ebv(ra, dec)
        ebv_eden_val = round(float(ebv_eden_t), 5)
        ebv_sf_val = round(float(ebv_sf_t), 5)
        av_eden_val = round(float(ebv_eden_t) * rv_gal, 5)
    except Exception as e:
        print(f"Error getting E(B-V) for star {i} at RA={ra}, DEC={dec}: {e}")
        return {
            'index': i, 'ebv_eden': np.nan, 'ebv_sf': np.nan, 
            'av_eden': np.nan, 'av_sf': np.nan, 'av_zh': np.nan, 'av_zh_std': np.nan
        }
    
    # Get Zaritsky extinction based on galaxy (LMC vs SMC)
    if ra > 40.0:  # LMC
        av_sf_val = round(float(ebv_sf_t) * rv_lmc, 5)
        try:
            ext_result = get_extinction(galaxy="LMC", ra=float(ra_hours), 
                                       dec=float(dec_hours), radius=set_radius)
            av_zh_val = round(ext_result[0], 5)
            av_zh_std_val = round(ext_result[1], 5)
        except Exception as e:
            av_zh_val = np.nan
            av_zh_std_val = np.nan
            print(f"Could not get Zaritsky extinction for LMC star {i} at RA: {ra_hours}, DEC: {dec_hours}")
    else:  # SMC
        av_sf_val = round(float(ebv_sf_t) * rv_smc, 5)
        try:
            ext_result = get_extinction(galaxy="SMC", ra=float(ra_hours), 
                                       dec=float(dec_hours), radius=set_radius)
            av_zh_val = round(ext_result[0], 5)
            av_zh_std_val = round(ext_result[1], 5)
        except Exception as e:
            av_zh_val = np.nan
            av_zh_std_val = np.nan
            print(f"Could not get Zaritsky extinction for SMC star {i} at RA: {ra_hours}, DEC: {dec_hours}")
    
    return {
        'index': i,
        'ebv_eden': ebv_eden_val,
        'ebv_sf': ebv_sf_val,
        'av_eden': av_eden_val,
        'av_sf': av_sf_val,
        'av_zh': av_zh_val,
        'av_zh_std': av_zh_std_val
    }


def find_min_av(coords, n_cores=8, output_file='ysg_candidate_extinctions_all.csv'):
    """
    Compute extinction values for all stars using parallel processing.
    
    Parameters:
        coords - DataFrame with 'RA' and 'DEC' columns
        n_cores - Number of CPU cores to use (default: 8)
        output_file - Name of output CSV file
    
    Returns:
        Astropy Table with extinction values
    """
    # Set up arrays to hold results
    n_stars = len(coords)
    ebv_eden = np.zeros(n_stars)
    ebv_sf = np.zeros(n_stars)
    av_eden = np.zeros(n_stars)
    av_sf = np.zeros(n_stars)
    av_zh = np.zeros(n_stars)
    av_zh_std = np.zeros(n_stars)

    # Convert coords to hours for Zaritsky queries
    coords_hours = coords.copy()
    coords_hours['RA'] = coords['RA'] / 15.0  # Convert degrees to hours
    coords_hours['DEC'] = coords['DEC']  # DEC remains in degrees
    
    # Prepare arguments for parallel processing
    args_list = [
        (i, coords.iloc[i]['RA'], coords.iloc[i]['DEC'], 
         coords_hours.iloc[i]['RA'], coords_hours.iloc[i]['DEC'])
        for i in range(n_stars)
    ]
    
    print(f"Processing {n_stars} stars using {n_cores} cores...")
    print(f"Output will be saved to: {output_file}")
    
    # Process in parallel with progress bar
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_star, arg): arg for arg in args_list}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=n_stars, desc="Processing stars"):
            try:
                result = future.result()
                idx = result['index']
                ebv_eden[idx] = result['ebv_eden']
                ebv_sf[idx] = result['ebv_sf']
                av_eden[idx] = result['av_eden']
                av_sf[idx] = result['av_sf']
                av_zh[idx] = result['av_zh']
                av_zh_std[idx] = result['av_zh_std']
            except Exception as e:
                print(f"Error processing star: {e}")
    
    # Create output table
    ext_tab = Table([coords['RA'], coords['DEC'], ebv_eden, ebv_sf, 
                     av_eden, av_sf, av_zh, av_zh_std],
                    names=['RA', 'DEC', 'ebv_eden', 'ebv_sf', 
                           'av_eden', 'av_sf', 'av_zh', 'av_zh_std'])
    
    # Write to file
    ext_tab.write(output_file, format='csv', overwrite=True)
    print(f"\n✓ Results written to {output_file}")
    print(f"  Total stars processed: {n_stars}")
    print(f"  Stars with valid extinctions: {np.sum(~np.isnan(av_zh))}")
    print(f"  Stars with NaN extinctions: {np.sum(np.isnan(av_zh))}")
    
    return ext_tab


def main():
    """Main function to run extinction calculations from command line."""
    parser = argparse.ArgumentParser(
        description='Calculate extinction values for YSG candidates using parallel processing.'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='../merged_smc_lmc_coords_all.csv',
        help='Input CSV file with RA and DEC columns (default: ../merged_smc_lmc_coords_all.csv)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='ysg_candidate_extinctions_all.csv',
        help='Output CSV file name (default: ysg_candidate_extinctions_all.csv)'
    )
    parser.add_argument(
        '--cores', 
        type=int, 
        default=8,
        help='Number of CPU cores to use (default: 8)'
    )
    
    args = parser.parse_args()
    
    # Read input coordinates
    print(f"Reading coordinates from {args.input}...")
    coords = pd.read_csv(args.input, comment='#', sep=r"\s+", names=['RA', 'DEC'])
    print(f"Loaded {len(coords)} stars")
    
    # Compute extinctions
    ext_tab = find_min_av(coords, n_cores=args.cores, output_file=args.output)
    
    return ext_tab


if __name__ == '__main__':
    main()
