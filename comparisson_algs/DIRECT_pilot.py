from scipy import optimize as opt

def scipy_bounds(problem_bounds):
    lbs = problem_bounds[0::2]
    ubs = problem_bounds[1::2]
    assert len(lbs) == len(ubs), "Bounds not defined correctly in scipy bounds object"
    keep_feas = [True]*len(ubs)
    bounds = opt.Bounds(lbs,ubs,keep_feasible=keep_feas)

    return bounds


def direct_run(problem):
    bounds_obj = scipy_bounds(problem["Bounds"])
    
    obj_func = problem["Cost Function (x)"]
    
    

if __name__ == "__main__":
    from opt_problems.ADMMBO_paper_problems import gardner1
    
    direct_run(gardner1)
    
    
    