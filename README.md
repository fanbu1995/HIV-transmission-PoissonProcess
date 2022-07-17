# HIV-transmission-PoissonProcess
Code for ``Inferring HIV Transmission Patterns from Viral Deep-Sequence Data via Latent Spatial Poisson Processes''

List of scripts:

- `main.py`: the main file to run for real data analysis (Case Study section in paper)
- `utilsHupdate.py`: all utility functions (imported by main.py)
- `server_main_simulations.py`: simulation code (run on a Slurm-system server)
- `process_simulations.py`: code for post-processing simulation experiment results
- `real-data-plots.R`: code for making various real data plots (used for Data and Case Study sections in paper)
- `process-simulations.R`: R code for post-processing simulation results and making plots

An anonymized real data table is included in `Rakai_data_processed.csv`.
