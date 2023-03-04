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