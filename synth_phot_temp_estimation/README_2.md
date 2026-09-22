# synth_phot_temp_estimation directory
this directory contains all the work for estimating temperatures for the stars.

### Notebooks:
 in temp_estimation_prep.ipynb: before estimating temperatures, most pre-work for that calculation is done here.

check_surveys_verification.ipynb: making sure our offset test for when to include APASS or MCPS or both is working.

temp_lum_results_excludeworst_allphot.ipynb: the main notebook for visualizing results from the calculations done in ysg_temp_estimation_excludeworst_allphot.py.

check_temp_fitting*.ipynb: these look at the results of ysg_temp_estimation_excludeworst_allphot.py in more nitty gritty detail, with different lists of potential problems from the fitting that were determined by eye. these lists/divisions were done in a google spreadsheet.

### Scripts:
ysg_temp_estimation_excludeworst_allphot.py. the main file. can be run with python ysg_temp_estimation_excludeworst_allphot.py --stars 1270 --cores 8
compute_extinctions_parallel.py: run to query the extinctions for edenhofer and sf maps for all stars. writes to file ysg_extinctions_all.csv. 
zh_scraper.py: file from anna that queries the extinctions from zh. not used here.

### Files:
synth_phot_all_models_*.csv: these contain different versions of pre-computing the synthetic photometry. the latest version which should be used is synth_phot_all_models_allphot_gordon.csv. if it doesn't say gordon, it is an older file that was calculated with a ccm89 extinction law.

ysg_extinctions_all.csv: this contains the extinctions for all targets according to both edenhofer and sf maps. these are used to limit the Av on BOSZ models tested against the observed SED in ysg_temp_estimation_excludeworst_allphot.py. was generated using compute_extinctions_parallel.py.

no_optical_*_coords.csv: the coordinates of stars that have no optical (neither APASS or MCPS) data.

choose_surveys_v3.csv: this contains the choice of MCPS or APASS for all stars based on plot_lin_interp function in temp_estimation_prep.ipynb. this is used by ysg_temp_estimation_excludeworst_allphot.py to flag on and off which photometry to include for each star.

ysg_temp_fitting_summary_v10_prefinal.csv: the summarized final results that contain THE final estimates of temperatures. the key csv file.


### Directories:
temp_fitting/: contains the individual files for each fitting technique, for each star. The iterations are saved here so we can look more deeply at how the temperature was estimated for any given star.





