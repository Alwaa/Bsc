# Bsc
This is the code repository for the Project in Constrained Bayesian Optimization by Alexander Walbækken at the Technichal University of Denmark

## Structure
There are several files for running experiments and testing the algoritms. The `\Experiments` forlder is where local results are stored in numpy format, and also where cached calculations for cizualizations are stored. The `\Utils` folder contains functions for plotting and storing/loading experiments.

The main ADMMBO algorithm is accesible as an import from `\ADMMBO_scaffold`. 

The Test-problems are contained in `\opt_problems`, with the problems defined as a dictionary of:

```
"Bound Type" : <string> (not used exepct in early phase)
"Bounds" : <Tuple of length d*2 with lower, upper bounds for all variables>
"Cost Function (x)":  <amda funcions to evaluate objective>
"Constraint Functions (z)": <List containing lamda funcions to evaluate contraints>
"Best Value": <float> (optional for graphs)

```

## For PESC
When running PESC there are several steps to the setup. After the database is setup, make an anaconda enviroment with the name "pesc" using the file `envPESC.yml`. The automatic scripts for running PESC as a subproscess should then run. 

When making a new problem, a manual addition to the problem files will be needed, then input "yes" when evrything is correctly inserted as per the file outline.

### Database
Assuming you are running windows, following this outline could help in the installation.

Add mongoDB ver 2.6.12 from archive in website (https://www.mongodb.com/download-center/community/releases/archive)
I installed it on Parent of Bsc directory for simplicity. Also see: https://stackoverflow.com/questions/2404742/how-to-install-mongodb-on-windows

The commands for installation must be run in <MONGODB_PATH>\bin, after being added to path and with administrator priviliges.

If an error occours, I would suffest starting over with (for my installation) `D:\mongodb\bin>mongod --remove`

Make the folder: `\comparisson_algs\PESC_folders\dbfolder`
Then run:
`mongod --dbpath=<PATH TO REP>\Bsc\comparisson_algs\PESC_folders\dbfolder --logpath=<PATH TO REP>\Bsc\comparisson_algs\PESC_folders\logfolder\log.txt --install`
followed by :
`net start MongoDB`

### Other changes
PESC env works with c code commeted out in the kernel_utils file of the PESC Spearmint files

- Changed kernel_utils.py so it used the slower python version insted of weave
- Changed print function in tasks/input_space.py line 272 function
    - All 4 in the input_params functiontion into normal print statements


### Simnibs
With simnibs as a sub-folder in the project (here named `\simhack`) This roughly follows the download instruction from https://github.com/simnibs/simnibs,
<!-- - download the simnibs_charm branch instead, then conda install the env `conda env create -f environment_win.yml -n simnibs` -->
# Eviroment
For the full enviroment have Anaconda in your path and run create an eviroemtn from the `envFull.yml` file.
- Run the `python setup.py develop` in administrator terminal for windows from the `\simhack` directory.
- Intall `pyvista` pip install
