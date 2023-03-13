from opt_problems.ADMMBO_paper_problems import gardner1, gardner2
problem = gardner1
from comparisson_algs.cma_pilot import cma_es

cma_es(problem)