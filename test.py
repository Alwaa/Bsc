from comparisson_algs.cma_pilot import cma_es
from comparisson_algs.COBYLA_pilot import COBYLA
from comparisson_algs.PESC_script import PESC_main, PESC_run_experiment, PESC_create_problem
from plotting import vizualize_toy
from opt_problems.ADMMBO_paper_problems import gardner1, gardner2
problem = gardner1
import cma
c_fun = cma_es(problem)


#COBYLA(problem)
#PESC_create_problem(gardner1, "test3")

# PESC_run_experiment("test3")
#xs, objs, all_objective_values = PESC_main(name = "test3")

#vizualize_toy(xs,objs,problem,decoupled=False)



