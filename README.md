# HIV-transmission-PoissonProcess
Code for _Inferring HIV Transmission Patterns from Viral Deep-Sequence Data via Latent Spatial Poisson Processes_

List of scripts:

- `main.py`: the main file to run for real data analysis (Case Study section in paper)
- `utilsHupdate.py`: all utility functions (imported by main.py)
- `utilsHupdatePrev.py`: utility functions used for simulation code (in `server_main_simulations.py`)
- `server_main_simulations.py`: simulation code (run on a Slurm-system server)
- `process_simulations.py`: code for post-processing simulation experiment results
- `real-data-plots.R`: code for making various real data plots (used for Data and Case Study sections in paper)
- `process-simulations.R`: R code for post-processing simulation results and making plots

An anonymized real data table is included in `Rakai_data_processed.csv`.

To run the code, you may first install [`Anaconda`](https://docs.anaconda.com/anaconda/install/index.html) and then install all the dependencies by running

```
conda env create --file dependencies.yml
```
