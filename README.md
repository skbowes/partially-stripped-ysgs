Running YSG Temperature modelling

Download BOSZ Models:
To download the BOSZ models, go to pysynphot_data/grid/bosz. From there, run the script files: 
./bosz_ysg_m-0.25_filtered.sh
./bosz_ysg_m-0.75_filtered.sh
This will grab the right models in the parameter range we have been targeting and put them in two directories, labelled by metallicity (m-0.25 and m-0.75). This should be all you need to do here!

Extra info on downloading BOSZ models:
There are two other script files, hlsp_bosz…_v1_bulkdl.sh. These are important to keep in the same directory since they contain all the possible model names, but definitely don’t run unless you want to download every model possible!
The pysynphot_data/grid/bosz directory also contains important files, such as the filter transmissions, the wavelengths that the model spectra are calculated at (stored separately in file bosz2024_wave_r2000.txt to save filespace).
The synthetic photometry was pre-computed and stored in the main directory in synth_phot_all_models.csv. 
If for some reason you wanted to re-compute the synthetic photometry (don’t think you’ll need to but just in case) you can use temp_estimation.ipynb (cmd+f to search for the synth_phot_all_models() function, then python ysg_temp_estimation.py --stars 848 --cores 8 to actually do the SED fitting comparison to the BOSZ models.

Downloading YSG lightcurves:
Not necessary for you to see the spectra, but if you want to do this in order to see the lightcurves of the stars, you can download the tar file that *********** sent via email and put the contents in a subdirectory off the main directory you should call /sub2025.

Downloading the temp fitting results for each star:
If you want to just play with the SED fitting or with overall statistics, you won’t need to do this. If you want to be able to see the histograms for a specific star, you’ll need to take the zipped file contained in this google drive folder (temp_fitting.zip) and open it in the main directory. It will create a subdirectory called /temp_fitting that contains parquet files for the results of fitting to cut bands and full bands for each star. It’s a sizable directory (1600 files). Only necessary for looking at the histogram distributions of parameters for a specific star.



Playing around!

Use the notebook temp_lum_results.ipynb for playing around. It contains the functions for making all the plots I’ve been showing you the past few weeks.

Probably the most important thing to note is that if you are using a plotting function that is showing the properties of one star (rather than all on the HR diagram for example), then I call the star by the index number in coords (the variable defined in all my files/for all my functions for the file merged_smc_lmc_coords.csv). It simply contains the coordinates for all the YSGs, both SMC and LMC. The first set of coordinates is index 0, the second index 1, and goes all the way to star 848. So for most of my functions, you just pass the index of the star you want to look at (say, fit_models_to_star_flux(4)) and it’ll show the SED and model spectra for the fourth star on that list.

If you want to pick out a specific star from a certain summary diagram (say, the new HR diagram, or the 1-1 fitting comparison plot) and figure out what index it is to check it out, I think it is easiest to put ysg_temp_fitting_summary_v2 (new results from SED fitting) into TOPCAT, make the plot in there and select the star. Then in the table view you should see the table column for ‘star_idx’.

Other important files that are called on in this file:
summary_results15.csv: This is the file that contains the results of our variability analysis so far! It contains Anna's old T and L estimations and my estimations of the dominant period.

synth_phot_all_models.csv: The pre-computed synthetic photometry.
Created using temp_estimation.ipynb

ysg_temp_fitting_summary_v2: This is the results of fitting for the temperatures (comparing observed to synthetic photometry). Stores the best fit (means, medians) information from the comparison.
Created using python ysg_temp_estimation.py --stars 848 --cores 8 (or however many cores you want to use)

Other details:
If you DO care about seeing the histograms for parameters for a certain star and you downloaded the /temp_fitting directory noted above, you can comment in a call to the histo() function to see this.

Most of the past work I did is stored as functions in metrics.py. This is already imported at the top of the notebook, and you can simply run the info() function to see most of the plots for any given star (lightcurve before and after correction, periodogram, SED in Jy, etc). The functions that perform the actual cleaning of data are contained in view_and_clean.py.
