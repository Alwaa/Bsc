from comparisson_algs.cma_pilot import cma_es
from comparisson_algs.COBYLA_pilot import COBYLA


from opt_problems.ADMMBO_paper_problems import gardner1, gardner2
problem = gardner1

cma_es(problem)
#COBYLA(problem)