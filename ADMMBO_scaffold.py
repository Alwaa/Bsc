# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from sklearn import gaussian_process
# import GPyOpt
# from GPyOpt import BayesianOptimization
from scipy.special import ndtr
from scipy.stats import norm
import copy
import warnings #For the non converging lnsrch

DEFAULT_OPTIONS = {
    "K"         : 15,
    "rho"       : 1, #orig. 0.1
    "epsilon"   : 1e-6, 
    "alpha"     : 5,
    "beta"      : 5,
    "alpha0"    : 10, #orig. 20
    "beta0"     : 10, #orig. 20
    "adjust_rho" : True,
}
AVAILABLE_OPTIONS = list(DEFAULT_OPTIONS.keys())

# bounds = search-space B ; n and m_i missing, insted the initial points are passed in
# M = penalty of infeasability
# K = max number of it; rho = penalty parm
# epsilon = tolerance parameter for stopping rule
# delta missing, 1 - delta is acceptance for prob that final solution is infeasable, for parameter to use if it does not converge
def admmbo(cost, constraints, M, bounds, grid, x0, f0=None, c0=None,
           format_return = False, options = {}):
    
    _D = DEFAULT_OPTIONS
    ## Overriding default options ##
    for k,v in options.items():
        if not k in AVAILABLE_OPTIONS:
            print(f"{k} not applicabple option in ADMMBO")
        _D[k] = v #New, unused value is assigned as of now if unvalid is inputted
    K, rho, epsilon = _D["K"], _D["rho"], _D["epsilon"]
    alpha, alpha0, beta, beta0 = _D["alpha"], _D["alpha0"],  _D["beta"], _D["beta0"]
    adjust_rho = _D["adjust_rho"]
    ## -------------------------- ##

    ## For debugging
    gp_logger = []
    rho_list = []
    ## -------------
    
    ## For outputting in unified
    xs_out = []
    objs = []
    ind_evals = []
    ## -------------------------------
    
    ### Defining Kernels ###
    K1 = gaussian_process.kernels.ConstantKernel(constant_value_bounds=(1e-10,1e10)) * \
             gaussian_process.kernels.Matern(nu=1.5,length_scale_bounds=(1e-2, 1e2)) #+\
                # gaussian_process.kernels.WhiteKernel()
    gpr = gaussian_process.GaussianProcessRegressor(kernel=K1)
    
    K2 = gaussian_process.kernels.ConstantKernel(constant_value_bounds=(1e-10,1e10)) * \
        gaussian_process.kernels.Matern(nu=1.5,length_scale_bounds=(1e-2, 1e2))
    ### ---------------- ###
    minimize_opts = {'eps': 0.001, 'maxls': 100} #Can't find documentation for maxls
        
    S = False
    k = 0 # "k=1 (0 as first index)"
    N = len(constraints) # As in paper

    #Initialicing GP classifiers
    #? Should have individual kernels based on bounds dimenstion?
    #TODO: Follow up on Kernel investigation
    #    TODO: OPT; Method where only one class can be present
    gpcs = [gaussian_process.GaussianProcessClassifier(kernel=K2) for i in range(N)]
    
    ## rho adjusting parameters ## # TODO Investigate further into rho setting
    #rho = 1 # IT WAS RHO #BUT still wierd that it hugs a bad corner
    tau = 2
    mup = 10
    rho_list.append(rho)
    ## ------------------------ ##

    ## Initializing z,y ## 
    # in the top corner, per meeting notes
    # TODO: Check initialization of zs and ys
    bounds=bounds.astype('float')
    zs = np.array([bounds[:,1].copy() for i in range(N)])
    ys = np.array([bounds[:,1].copy() for i in range(N)])
    zolds = [zs[i].copy() for i in range(N)]
    ## ---------------- ##

    if f0 is None:
        f0 = cost(x0)
    if c0 is None:
        c0s = [[]]*N
        for i in range(N):
            constraint = lambda inp: 1 - ( constraints[i](inp) <= 0 )#Boolean constraint
            c0s[i] = constraint(x0)

    #print(x0,f0)
    gpr.fit(x0,f0) #Initializing the GP regression
    for i in range(N): #Initializing the GP classifiers
        #print(x0,c0s[i])
        gpcs[i].fit(x0,c0s[i]) 
    
    #Logging first variables
    xs_out.append(x0)
    objs_arr =np.concatenate((f0.reshape(-1,1),np.array(c0s).T),axis=1)
    objs.append(objs_arr)
    ind_evals.append(np.full(objs_arr.shape,True))


    def u_post_pluss(x_in,zs_in,ys_in,rho_in): ## Eq. (10) after the f(x) + .....
        sq_norm = np.sum((x_in[:,None,:]-zs_in+ys_in/rho_in)**2,axis = 2) # Squared 2 norm for each xs/ys dimension
        u_post = 0.5*rho_in*np.mean(sq_norm,axis=1) #?DONE?: Check the sum funciton works propperly with new zs and ys
        return u_post
    
    def gpr_ei(x_in,zs,ys,rho,gpr,ubest):
        x=np.atleast_2d(x_in)
        x = np.nan_to_num(x, nan= bounds[0][1]) #TODO: Is there a problem with example0?
        mu,std = gpr.predict(x, return_std=True) #TODO:std returns 0? Causes divide by zero
        muu = mu + u_post_pluss(x,zs,ys,rho)
        #xx = (ubest-muu)/std #Original formulation
        xx = -(muu-ubest)/(std+1e-10) #?Why negative here?? #Added small value 
        ei = std * (xx * ndtr(xx) + norm.pdf(xx))
        return ei
    
    def h_post_minus(x_in,z_in,y_in,rho_in,M_in):
        x,z,y = np.atleast_2d(x_in),np.atleast_2d(z_in),np.atleast_2d(y_in)
        sq_norm = np.sum((x-z+y/rho_in)**2,axis = 1)
        h_post = 0.5*rho_in/M_in * sq_norm
        return h_post

    def gpc_ei(x,z,y,rho,M,gpc,hbest):
        z=np.atleast_2d(z)
        hh2 = hbest - h_post_minus(x,z,y,rho,M)
        hh = hbest - 0.5*rho/M * np.sum((x[None]-z+y[None]/rho)**2,axis=1)
        theta = gpc.predict_proba(z)[:,1]
        ei = np.zeros(z.shape[0])
        idx1 = (hh>0) & (hh<1)
        ei[idx1] = hh[idx1] * (1.0 - theta[idx1])
        idx2 = hh > 1.0
        ei[idx2] = hh[idx2] - theta[idx2]
        return ei
    
    # z=np.mean(gpc.base_estimator_.X_train_[gpc.base_estimator_.y_train_==0],0)
    ## First iteration has higher bidget as per discussion
    alphac=alpha0
    betac=beta0
    while k < K and not S:
        ### OPT ###
        for t in range(alphac):
            fx = gpr.y_train_ # f(x) for each value untill now
            X = gpr.X_train_ # xs until now

            ubest = np.min(fx + u_post_pluss(X,zs,ys,rho)) # Eq. (10)

            ei = gpr_ei(grid,zs,ys,rho,gpr,ubest)
            x = grid[np.argmax(ei)]
            eif = lambda x_in: -gpr_ei(x_in,zs,ys,rho,gpr,ubest)
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                opt = minimize(eif, x, bounds=bounds, options = minimize_opts) # Minize negative expected improvement
                if not caught_warnings is None:
                    for warn in caught_warnings:
                        print(f"++++\nwarn: {warn.message}")
                        print(warn.category)
                        print(str(warn))
            old_x = copy.copy(x)
            x = opt.x
            # print(f'eix:{ei.max()}')
            gp_logger.append([grid,0,0,x,-eif(x),old_x,-eif(old_x),
                              copy.deepcopy(ei),copy.deepcopy(gpr.X_train_),copy.deepcopy(gpr.y_train_)])


            x = np.nan_to_num(x, nan= bounds[0][1]) #TODO: Is there also problem with lamwillcox3?
            x_eval,cost_eval = x[None],cost(x[None])
            gpr.fit(np.concatenate((gpr.X_train_,x_eval),axis=0),
                    np.concatenate((gpr.y_train_,cost_eval),axis=0))
            
            print(np.round(cost_eval,decimals=1),end = "|",flush=True)
            ## Logging for unified outuput
            xs_out.append(x_eval)
            arr_objs = np.full(N+1,0.0)
            arr_objs[0] = cost_eval[0]
            objs.append(arr_objs[None])
            arr_evls = np.full(N+1,False)
            arr_evls[0] = True
            ind_evals.append(arr_evls[None])
            
        ### --- ###

        #original_u = gpr.y_train_ + 0.5*rho*np.sum((gpr.X_train_-zs+ys/rho)**2,axis=1)
        # TODO: check and mark part of algorithm 
        u = gpr.y_train_ + u_post_pluss(gpr.X_train_,zs,ys,rho)
        x = gpr.X_train_[np.argmin(u)]

        ### FEAS ###
        rs = [[]]*N
        ss = [[]]*N
        for i in range(N):
            for t in range(betac):
                constraint = lambda inp: 1 - ( constraints[i](inp) <= 0 )#Boolean constraint
                gpc = gpcs[i]

                #Line 4 in Alg. 3.3
                h = gpc.base_estimator_.y_train_ + \
                    0.5*rho/M*np.sum((x[None]-gpc.base_estimator_.X_train_
                                    +ys[i][None]/rho)**2,axis=1) #TODO: Chech correctness with ys
                hbest = np.min(h)
                
                eif = lambda z: -gpc_ei(x,z,ys[i],rho,M,gpc,hbest) #TODO: Chech correctness with ys and zs
                qi = lambda zi: 0.5*rho/M * np.sum((x[None]-zi+ys[i][None]/rho)**2,axis=1) #TODO: Chech correctness with ys
                hh = hbest - qi(grid)
                theta = gpc.predict_proba(grid)[:,1]

                ei = np.zeros(grid.shape[0])
                # idx = hh>0
                # ei[idx]=(1-theta)*hh[idx]
                idx1 = (hh>0)&(hh<1)
                ei[idx1] = hh[idx1] * (1.0 - theta[idx1])
                idx2 = hh>1.0
                ei[idx2] = hh[idx2] - theta[idx2]
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    opt=minimize(eif,grid[np.argmax(ei)],bounds=bounds, options = minimize_opts) #,options={ "maxiter" : 15000000}) #Sklearn minimize
                    if not caught_warnings is None:
                        for warn in caught_warnings:
                            print(f"----\nwarn: {warn.message}")
                            print(warn.category)
                            print(str(warn))
                z = opt.x
                # ei[idx2][ei[idx2]<0.0] = 0.0
                # temp = (hh[idx2]-1)*theta[idx2]
                # temp[temp<0] = 0.0
                # ei[idx2] += temp
                # print(ei.max())
                z = grid[np.argmax(ei)]
                
                z_eval, const_eval = z[None], constraint(z[None])
                gpc.fit(np.concatenate((gpc.base_estimator_.X_train_,z_eval), axis=0),
                        np.concatenate((gpc.base_estimator_.y_train_,const_eval),axis=0))
                
                print(const_eval,end = "|",flush=True)            
                ## Logging for unified outuput
                xs_out.append(z_eval)
                arr_objs = np.full(N+1,0.0)
                arr_objs[i+1] = const_eval[0]
                objs.append(arr_objs[None])
                arr_evls = np.full(N+1,False)
                arr_evls[i+1] = True
                ind_evals.append(arr_evls[None])
            
            h = gpc.base_estimator_.y_train_ + \
                    0.5*rho/M*np.sum((x[None]-gpc.base_estimator_.X_train_
                                    +ys[i][None]/rho)**2,axis=1) #TODO: check correctness with ys
            z = gpc.base_estimator_.X_train_[np.argmin(h)]

            zs[i] = z #TODO: refactor when stable

            ys[i] += rho * (x - z)
            # print(f"{x}\n{zs[i]}\n{rho}")
            # print(f'x: {x} \nz: {z} \ny: {ys[i]}')
            r = (x - zs[i])**2
            rl=np.sqrt(np.sum(r**2))
            s = - rho * (z - zolds[i])
            sl = np.sqrt(np.sum(s**2))
            # print(r)
            # print(s)

            zolds[i] = z.copy()

        if rl < epsilon and sl < epsilon:
            S = True
        k += 1
        ## rho adjusting step ##
        if rl > (mup*sl) and adjust_rho:
            rho *= tau
        elif sl > (mup*rl) and adjust_rho:
            rho /= tau
        
        #print(f' rho:{rho} \n r:{rl} \n s:{sl}')
        rho_list.append(rho)
        ## ----------------- ##
        if S:
            break
        alphac=alpha
        betac=beta
        
    if format_return:
        xs_out = np.concatenate(xs_out,axis = 0)
        objs = np.concatenate(objs,axis = 0)
        ind_evals = np.concatenate(ind_evals,axis = 0)
        return x,z,gpr,gpcs, gp_logger, rho_list, xs_out, objs, ind_evals

    return x,z,gpr,gpcs, gp_logger, rho_list

def admmbo_run(problem, x0, max_iter = 100, admmbo_pars = {}, debugging = False, start_all = True):
    print("ADMMBO:")
    #################################
    # For setting the type of grid to use for solving the problem (discreticing space, and then 
    # selectin a less fine grid for less GP calculations)
    num_samples = 400 # in each dimension
    grid_step = 10
    M = admmbo_pars.pop("M", None)
    options_in = admmbo_pars
    #################################

# Fetching problem info #
    bounds = problem["Bounds"]
    dim_num = len(bounds)//2
    mem_saving = 1
    if not debugging:
        mem_saving = grid_step
    if dim_num > 3:
        num_samples = 140
        mem_saving = grid_step
        if M is None:
            M = 10
    xins = (np.linspace(bounds[i*2], bounds[1+i*2], num_samples//mem_saving) for i in range(dim_num))
    xy = np.array(np.meshgrid(*xins, indexing='ij')).reshape(dim_num, -1).T

    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Functions (z)"]
    num_constraints = len(constraintf)
# --------------------- #

    extent_tuple = bounds #bounds are defined as (lb1,ub1,...,lb#,ub#)
    bounds_array = np.array(bounds).reshape(-1,2)

    # For drawing bounds + true cost function
    # Bound area is then 0 in heatmap
    if M is None:
        func = costf(xy)
        con = np.all(np.array([constraintf[i](xy) <= 0  for i in range(num_constraints)]),
                axis = 0) #Boolean constraints
        ff = con*func
        M = np.max(ff) - np.min(ff) # They set it as the unconstrained range of f, while this is the constrained range
                            # ADMMBO is (claimed) not sensitive to M in a wide range of values ref. sec. 5.7 (only to smaller values)
    
    if not start_all: #Go through all points untill you have one of each constraint both upheld and violated...
        found = False
        cons_start = np.array([constraintf[c](x0[:1]) <= 0  for c in range(num_constraints)])
        for i in range(1,(len(x0)-1)):
            print(i,end = "|", flush = True)
            #OLDcons_start = np.array([constraintf[c](x0[:i]) <= 0  for c in range(num_constraints)])
            cons_add = np.array([constraintf[c](x0[i]) <= 0  for c in range(num_constraints)]) #Saving on evaluations...
            cons_start = np.concatenate((cons_start,cons_add), axis = 1)
            #print(cons_start)
            cons_per = np.sum(cons_start, axis = 1)
            more_than_none = np.all((0 < cons_per))
            less_than_all = np.all((i > cons_per))
            one_fully = np.any((num_constraints == np.sum(cons_start, axis=0))) #TODO:Check that this ensures one is fully upheld
            if less_than_all and more_than_none and one_fully:
                x0 = x0[:(i+1)]
                found = True
                break
        print("\n")
        if not found and not debugging: #Return the initialization points if failed
            print("ADMMBO initialization did not find feasable strating point set")
            fs_failed = costf(x0).reshape((-1,1))
            cons_failed = np.array([constraintf[c](x0)  for c in range(num_constraints)]).reshape((-1,num_constraints))
            obj_failed = np.concatenate((fs_failed,cons_failed),axis=1)
            return x0, obj_failed, np.full((x0.shape[0],num_constraints+1), True)
    
    K_in_old = (max_iter-len(x0))//4 #50 #example0 K = 30
    ## Calculating ADMMBO budget based on alpha and beta values pluss max_iter
    a0, a = admmbo_pars.get("alpha0", DEFAULT_OPTIONS["alpha0"]), admmbo_pars.get("alpha", DEFAULT_OPTIONS["alpha"])
    b0, b = admmbo_pars.get("beta0", DEFAULT_OPTIONS["beta0"]), admmbo_pars.get("beta", DEFAULT_OPTIONS["beta"])
    K_in = int(1+(max_iter-a0-b0*num_constraints-(len(x0)))/(a+b*num_constraints)) #Prev. forgot to subtract the number of x0s tested
    print(K_in_old, K_in)
    K_in = max(K_in, 2) #At least 2 iterationis (Would be very unlucky for it to fire)
    options_in["K"] = K_in
    
    # Grid with evry grid_step-th point of space
    # Should now behave well with non square inputs
    xins = (np.linspace(bounds[i*2], bounds[1+i*2], num_samples)[::grid_step] for i in range(dim_num))
    #TODO: Check if it is same as //grid_step
    grid = np.array(np.meshgrid(*xins,indexing='ij')).reshape(dim_num,-1).T

    #Running ADMMBO 
    xo,zo,gpr,gpc, gp_logger, rho_list, xs_out, obj_out, eval_type = admmbo(costf, constraintf, M, bounds_array, grid, x0, 
                                                                            options=options_in, format_return=True)

    ## Formatting output ## #DONE: Format so order of queries is correct
    # xsr = gpr.X_train_
    # obj = gpr.y_train_
    # xsc = gpc.base_estimator_.X_train_
    # cc = gpc.base_estimator_.y_train_

    # new_obj = np.concatenate((obj,np.zeros(len(cc)))).reshape(-1,1)
    # new_cc = np.concatenate((np.ones(len(obj)),cc)).reshape(-1,1)
    # obj_out = np.concatenate((new_obj,new_cc),axis = 1) ## SImple combined obj and constraied after eachother

    # objmaks = np.concatenate((np.full(len(obj),True),np.full(len(cc),False))).reshape(-1,1)
    # constmaks = np.concatenate((np.full(len(obj),False),np.full(len(cc),True))).reshape(-1,1)
    # eval_type = np.concatenate((objmaks,constmaks),axis = 1)

    # xs_out = np.concatenate((xsr,xsc))
    ## ------------------ ##

    if not debugging:
        #print(xs_out, obj_out, eval_type)
        return xs_out, obj_out, eval_type
    
    ### Debugging ###
    #TODO: FIx debugging now that it isnt xin but xins eh xin = xins[0] for sq bounds or smth
    import matplotlib.pyplot as plt
    
    #print(xsc.shape, cc.shape)
    #print(xsr,obj)
    
    # if xsr.shape == xsc.shape:
    #     print("Diff of xs")
    #     print(xsr-xsc)
    
    K1 = gaussian_process.kernels.ConstantKernel(constant_value_bounds=(1e-10,1e10)) * \
            gaussian_process.kernels.Matern(nu=1.5,length_scale_bounds=(1e-2, 1e2)) #+\
    fig_list = []
    for i, round in enumerate(gp_logger):
        sample_n = len(round[-1])
        if i%10 == 0:
            fig = plt.figure(figsize=(24,8));fig_list.append(fig)
            gs = fig.add_gridspec(2, 5)
            axs = gs.subplots();flat_axs = axs.flat
            fig.suptitle(f"EI for [{i}-{i+10}]")

        # print(sample_n)
        # print(f"x sampled: {round[3]}")
        # print(f"{round[5]}: {round[6]}")
        # print(f"{round[3]}: {round[4]}")

        # gpr_cop = gaussian_process.GaussianProcessRegressor(kernel=K1)
        # gpr_cop.fit(round[-2],round[-1])
        # plt.figure();plt.imshow(gpr_cop.predict(xy).reshape(len(xin),-1),
        #                         extent=(extent_tuple),origin='lower')
        # ;plt.title(f'Objective Function Estimate [{i}] {round[3]}')
        # plt.show(block = True)
        ei = round[-3]
        mp = flat_axs[i%10].imshow(ei.reshape(int(np.sqrt(len(ei))),int(np.sqrt(len(ei)))).T,
                                extent=(extent_tuple),origin='lower')
        flat_axs[i%10].plot(round[3][0],round[3][1],"rx")
        plt.colorbar(mp, ax = flat_axs[i%10],location='bottom')#;flat_axs[i%10].label_outer()
    for f in fig_list:
        f.tight_layout()

    plt.figure();plt.plot(rho_list);plt.title('Rho')
    print(f"M = {M}")
    plt.show(block = True)
        
    ### Plots ###
    ## Cost/Objective Function Estimate ##
    plt.figure();plt.imshow(gpr.predict(xy).reshape(len(xin),-1).T,
                            extent=(extent_tuple),origin='lower')
    plt.colorbar();plt.title('Cost/Objective Function Estimate')
    ## Constraint Probability Estimate ##
    plt.figure();plt.imshow(gpc.predict_proba(xy)[:,1].reshape(len(xin),-1).T,
                            extent=(extent_tuple),origin='lower')
    plt.colorbar();plt.title('Constraint Probability Estimate')
    ## True Cost/Objective Function ## 
    plt.figure()
    plt.imshow(func.reshape(len(xin),-1).T,extent=(extent_tuple),origin='lower')
    plt.colorbar();plt.title('True Cost/Objective Function')
    ### ---- ###
    
    # Needed for VScode
    plt.show()
    
    return xs_out, obj_out, eval_type

if __name__ == "__main__":
    from opt_problems.example_problems import example0
    from opt_problems.paper_problems import gardner1, gardner2
    from utils.sampling import grid_sampling

    problem = gardner2
    x0 = grid_sampling(problem, num_per_dim = 5) # 5 is original, 3 works
    x0 = np.array([[1.5,4.5],[2.5,3.5]])
    admmbo_run(problem, x0)