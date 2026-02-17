#!/usr/bin/env python3
"""
Filter BOSZ bulk download script to only include models with specific parameters:
- r2000 (already in the filename)
- m-0.75 (metallicity, already filtered by script name)
- a+0.00 (alpha enhancement = 0)
- c+0.00 (carbon enhancement = 0) 
- v2 (microturbulence = 2 km/s)
- Temperature between 4000-14000K
"""

import re

def filter_bosz_script(input_file, output_file, metallicity):
    """Filter the BOSZ bulk download script for YSG parameters"""
    
    filtered_lines = []
    total_lines = 0
    kept_lines = 0
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Keep the shebang line
    if lines[0].startswith('#!'):
        filtered_lines.append(lines[0])
    
    # Create regex pattern for the specific metallicity
    pattern = rf'bosz2024_(\w+)_t(\d+)_g([+-]\d+\.\d+)_m{metallicity}_a([+-]\d+\.\d+)_c([+-]\d+\.\d+)_v(\d+)_r2000_resam\.txt\.gz'
    
    for line in lines:
        total_lines += 1
        
        # Skip shebang and empty lines
        if line.startswith('#!') or line.strip() == '':
            continue
            
        # Parse the filename to extract parameters
        match = re.search(pattern, line)
        
        if match:
            atmos, teff, logg, alpha, carbon, vmicro = match.groups()
            teff = int(teff)
            logg = float(logg)
            alpha = float(alpha)
            carbon = float(carbon)
            vmicro = int(vmicro)
            
            # Apply filters for YSG parameters, split up for logg values that I downloaded (after discussion with Anna)
            if (4000 < teff <= 10000 and    
                logg <= 2.5 and   
                    alpha == 0.0 and           
                    carbon == 0.0 and  
                    vmicro == 2):                 
                    
                    filtered_lines.append(line)
                    kept_lines += 1
                    print(f"Keeping: T={teff}K, log(g)={logg}, α={alpha}, C={carbon}, v={vmicro}")

            if (10000 < teff <= 14000 and     # Temperature range, had formerly done 4000-10000K
                logg <= 3.0 and    # previously had been <= 2.5
                alpha == 0.0 and           
                carbon == 0.0 and  
                vmicro == 2):                 
                
                filtered_lines.append(line)
                kept_lines += 1
                print(f"Keeping: T={teff}K, log(g)={logg}, α={alpha}, C={carbon}, v={vmicro}")
    
    # Write filtered script
    with open(output_file, 'w') as f:
        f.writelines(filtered_lines)
    
    print(f"\nFiltering complete:")
    print(f"Total lines processed: {total_lines}")
    print(f"Lines kept: {kept_lines}")
    print(f"Filtered script saved as: {output_file}")
    
    return kept_lines

if __name__ == "__main__":
    # Filter both metallicity scripts
    print("Filtering m-0.75 script...")
    kept_075 = filter_bosz_script(
        "hlsp_bosz_bosz2024_sim_r2000_m-0.75_v1_bulkdl.sh", # the input script filename that is from MAST and lists all the model files available
        "bosz_ysg_m-0.75_filtered.sh",
        "-0\\.75"
    )
    
    print("\nFiltering m-0.25 script...")
    kept_025 = filter_bosz_script(
        "hlsp_bosz_bosz2024_sim_r2000_m-0.25_v1_bulkdl.sh", # the input script filename that is from MAST and lists all the model files available
        "bosz_ysg_m-0.25_filtered.sh",
        "-0\\.25"
    )
    
    print(f"\nTotal models for YSG analysis: {kept_075 + kept_025}")