from comparisson_algs.cma_pilot import cma_es
from comparisson_algs.cobyla_pilot import COBYLA
from comparisson_algs.pesc_script import pesc_main, pesc_run_experiment, pesc_create_problem
from plotting import vizualize_toy
from opt_problems.ADMMBO_paper_problems import gardner1, gardner2
problem = gardner1

# For testting
xs = None


#xs, objs, all_objective_values = cma_es(problem)


# NOT CORRECT OUTPUT YET !!! #xs, objs, all_objective_values = COBYLA(problem)
COBYLA(problem)



#pesc_create_problem(gardner1, "test3")
# pesc_run_experiment("test3")
# xs, objs, all_objective_values = pesc_main(name = "test3")


if xs != None:
    vizualize_toy(xs,objs,problem,decoupled=False)



