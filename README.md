# Bsc
Project in Constrained Bayesian Optimization

# TODO:
- PESC gets Nan error sometimes, simply entering command again resumes job?
    - Make a try-catch loop in running script?
- How to decouple PESC??
- Check output of PESC for decoupled problems
- Lock versions of env file when starting results

# DONE? (Check)

conda update -n base -c defaults conda

## To look into
- lbfgs failed to converge (status=2):
ABNORMAL_TERMINATION_IN_LNSRCH

- C:\ ... \sklearn\gaussian_process\kernels.py:420: ConvergenceWarning: The optimal value found for dimension 0 of parameter k2__length_scale is close to the specified lower bound 0.01. Decreasing the bound and calling fit again may find a better value.

-    Windows 64-bit packages of scikit-learn can be accelerated using scikit-learn-intelex.
    More details are available here: https://intel.github.io/scikit-learn-intelex

    For example:

        $ conda install scikit-learn-intelex
        $ python -m sklearnex my_application.py


### First one specifies location, other does not
conda env update --prefix ./env --file env.yml  --prune
vs.
conda env update --file env.yml  --prune


## For PESC
Add mongoDB ver 2.6.12 from archive in website (https://www.mongodb.com/download-center/community/releases/archive)
I installed it on Parent of Bsc directory for simplicity

https://stackoverflow.com/questions/2404742/how-to-install-mongodb-on-windows

IN <MONGODB_PATH>\bin
AFTER ADDET TO PATH
AS ADMINISTRATOR

OWN COMPUTER: D:\mongodb\bin>mongod --remove

OR: .\ if not in path...
`mongod --dbpath=<PATH TO REP>\Bsc\comparisson_algs\PESC_folders\dbfolder --logpath=<PATH TO REP>\Bsc\comparisson_algs\PESC_folders\logfolder\log.txt --install`
then:
`net start MongoDB`

MAKE: \comparisson_algs\PESC_folders\dbfolder

(Could not work on lapto, will try to remove everything in dbfolder and try the isntall again)

PESC env works with c code commeted out in the kernel_utils file of the PESC Spearmint files


- Changed kernel_utils.py so it used the slower python version insted of weave
- Changed print function in tasks/input_space.py line 272 function
    - All 4 in the input_params functiontion into normal print statements 


### Simnibs
- Follow download instruction from ...
- download the simnibs_charm branch instead, then conda install the env `conda env create -f environment_win.yml -n simnibs`
- Run the `python setup.py develop` in administrator terminal for windows
- Intall `pyvista` pip install
#### Potential
Might try again to fix plotting behaviour and maybe add a modern python c compiler
FUTURIZE with:
futurize -0 -w Spearmint
