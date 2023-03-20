# Bsc
Project in Constrained Bayesian Optimization

# TODO:
- costf and constraintf not same behaviour with single input
- Lock versions of env file when starting results

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

Add mongoDB ver 2.smth from archive in website
I Installed it on ????

IN <MONGODB_PATH>\bin
AFTER ADDET TO PATH
AS ADMINISTRATOR

D:\mongodb\bin>mongod --remove

mongod --dbpath=<PATH TO REP>\Bsc\comparisson_algs\PESC_folders\dbfolder --logpath=<PATH TO REP>\Bsc\comparisson_algs\PESC_folders\logfolder\log.txt --install

PESC env works with c code commeted out in the kernel_utils file of the PESC Spearmint files


Might try again to add a modern c compiler
FUTURIZE with:
futurize -0 -w Spearmint

Changed kernel_utils.py so it used the slower python version insted of weave