#!/usr/bin/env python3
"""
Merge coordinates from merged_smc_lmc_coords.csv with additional coordinates
from SMC and LMC candidate files.
"""

import csv

def main():
    # Read the existing merged file
    existing_coords = []
    with open('merged_smc_lmc_coords.csv', 'r') as f:
        header = f.readline().strip()  # Save the header line
        for line in f:
            line = line.strip()
            if line:
                existing_coords.append(line)
    
    print(f"Read {len(existing_coords)} coordinates from merged_smc_lmc_coords.csv")
    
    # Read SMC candidates and extract RA, DEC
    smc_coords = []
    with open('ysg_candidates/prefinal_smc_ysgcands_allphot.csv', 'r') as f:
        # Read all lines
        lines = f.readlines()
        
        # Find the header line (first non-comment line)
        header_idx = 0
        for i, line in enumerate(lines):
            if not line.startswith('#'):
                header_idx = i
                break
        
        # Parse the CSV starting from header
        reader = csv.DictReader(lines[header_idx:])
        for row in reader:
            ra = row['ra']
            dec = row['dec']
            smc_coords.append(f"{ra} {dec}")
    
    print(f"Read {len(smc_coords)} coordinates from prefinal_smc_ysgcands_allphot.csv")
    
    # Read LMC candidates and extract RA, DEC
    lmc_coords = []
    with open('ysg_candidates/prefinal_lmc_ysgcands_allphot.csv', 'r') as f:
        # Read all lines
        lines = f.readlines()
        
        # Find the header line (first non-comment line)
        header_idx = 0
        for i, line in enumerate(lines):
            if not line.startswith('#'):
                header_idx = i
                break
        
        # Parse the CSV starting from header
        reader = csv.DictReader(lines[header_idx:])
        for row in reader:
            ra = row['ra']
            dec = row['dec']
            lmc_coords.append(f"{ra} {dec}")
    
    print(f"Read {len(lmc_coords)} coordinates from prefinal_lmc_ysgcands_allphot.csv")
    
    # Write the merged file
    with open('merged_smc_lmc_coords_all.csv', 'w') as f:
        # Write header
        f.write(header + '\n')
        
        # Write existing coordinates
        for coord in existing_coords:
            f.write(coord + '\n')
        
        # Write SMC coordinates
        for coord in smc_coords:
            f.write(coord + '\n')
        
        # Write LMC coordinates
        for coord in lmc_coords:
            f.write(coord + '\n')
    
    total = len(existing_coords) + len(smc_coords) + len(lmc_coords)
    print(f"\nWrote {total} coordinates to merged_smc_lmc_coords_all.csv")
    print(f"  - {len(existing_coords)} existing")
    print(f"  - {len(smc_coords)} SMC")
    print(f"  - {len(lmc_coords)} LMC")

if __name__ == '__main__':
    main()
