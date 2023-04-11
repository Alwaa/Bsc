# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from sklearn import gaussian_process
# import GPyOpt
# from GPyOpt import BayesianOptimization
from scipy.special import ndtr
from scipy.stats import norm
import copy

# bounds = search-space B ; n and m_i missing, insted the initial points are passed in
# M = penalty of infeasability
# K = max number of it; rho = penalty parm
# epsilon = tolerance parameter for stopping rule
# delta missing, 1 - delta is acceptance for prob that final solution is infeasable, for parameter to use if it does not converge
def admmbo(cost, constraints, M, bounds, grid, x0, f0=None, c0=None,
           K=15, rho=0.1, alpha=5, beta=5, alpha0=20, beta0=20,
           epsilon=1e-6):

    ## For debugging
    gp_logger = []
    rho_list = []
    
    ### Defining Kernels ###
    K1 = gaussian_process.kernels.ConstantKernel(constant_value_bounds=(1e-10,1e10)) * \
             gaussian_process.kernels.Matern(nu=1.5,length_scale_bounds=(1e-2, 1e2)) #+\
                # gaussian_process.kernels.WhiteKernel()
    gpr = gaussian_process.GaussianProcessRegressor(kernel=K1)
    
    K2 = gaussian_process.kernels.ConstantKernel(constant_value_bounds=(1e-10,1e10)) * \
        gaussian_process.kernels.Matern(nu=1.5,length_scale_bounds=(1e-2, 1e2)) #TODO: sjekk bound for corner behaviour
    ### ---------------- ###
    
        
    S = False
    k = 0 # "k=1 (0 as first index)"
    N = len(constraints) # As in paper

    #Initialicing GP classifiers
    #? Should have individual kernels based on bounds dimenstion?
    #TODO: Follow up on Kernel investigation
    #    TODO: OPT; Method where only one class can be present
    gpcs = [gaussian_process.GaussianProcessClassifier(kernel=K2) for i in range(N)]
    
    ## rho adjusting parameters ## # TODO Investigate further into rho setting
    adjust_rho = True
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
    print(zs)
    # z = np.array((1,1))
    # z = x0[c0==0][np.argmin(f0[c0==0])]
    # y = np.ones(x0.shape[1])
    # y = np.zeros(x0.shape[1])
    zolds = [zs[i].copy() for i in range(N)]
    ## ---------------- ##

    if f0 is None:
        f0 = cost(x0)
    if c0 is None:
        c0s = [[]]*N
        for i in range(N):
            constraint = lambda inp: 1 - ( constraints[i](inp) <= 0 )#Boolean constraint
            c0s[i] = constraint(x0)

    print(x0,f0)
    gpr.fit(x0,f0)
    for i in range(N):
        print(x0,c0s[i])
        gpcs[i].fit(x0,c0s[i]) 


    def u_post_pluss(x_in,zs_in,ys_in,rho_in): ## Eq. (10) after the f(x) + .....
        sq_norm = np.sum((x_in[:,None,:]-zs_in+ys_in/rho_in)**2,axis = 2) # Squared 2 norm for each xs/ys dimension
        u_post = 0.5*rho_in*np.mean(sq_norm,axis=1) #?DONE?: Check the sum funciton works propperly with new zs and ys
        return u_post
    
    def gpr_ei(x_in,zs,ys,rho,gpr,ubest):
        x=np.atleast_2d(x_in)
        mu,std = gpr.predict(x, return_std=True)
        muu = mu + u_post_pluss(x,zs,ys,rho)
        #xx = (ubest-muu)/std #Original formulation
        xx = -(muu-ubest)/std #?Why negative here??
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
            opt = minimize(eif, x, bounds=bounds) # Minize negative expected improvement
            old_x = copy.copy(x)
            x = opt.x
            # print(f'eix:{ei.max()}')
            gp_logger.append([grid,0,0,x,-eif(x),old_x,-eif(old_x),
                              copy.deepcopy(ei),copy.deepcopy(gpr.X_train_),copy.deepcopy(gpr.y_train_)])

            #print(x[None],"\n-----\n",gpr.y_train_)
            #print(cost(x[None]),"\n-----\n",gpr.y_train_)
            gpr.fit(np.concatenate((gpr.X_train_,x[None]),axis=0),
                    np.concatenate((gpr.y_train_,cost(x[None])),axis=0))
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
                opt=minimize(eif,grid[np.argmax(ei)],bounds=bounds) #Sklearn minimize
                z = opt.x
                # ei[idx2][ei[idx2]<0.0] = 0.0
                # temp = (hh[idx2]-1)*theta[idx2]
                # temp[temp<0] = 0.0
                # ei[idx2] += temp
                # print(ei.max())
                z = grid[np.argmax(ei)]

                gpc.fit(np.concatenate((gpc.base_estimator_.X_train_,z[None]), axis=0),
                        np.concatenate((gpc.base_estimator_.y_train_,constraint(z[None])),axis=0))
            
            h = gpc.base_estimator_.y_train_ + \
                    0.5*rho/M*np.sum((x[None]-gpc.base_estimator_.X_train_
                                    +ys[i][None]/rho)**2,axis=1) #TODO: check correctness with ys
            z = gpc.base_estimator_.X_train_[np.argmin(h)]

            zs[i] = z #TODO: refactor when stable

            print(f"{x}\n{zs[i]}\n{rho}")
            ys[i] += rho * (x - z)
            print(f'x: {x} \nz: {z} \ny: {ys[i]}')
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
        print(f' rho:{rho} \n r:{rl} \n s:{sl}')
        rho_list.append(rho)
        ## ----------------- ##
        if S:
            break
        alphac=alpha
        betac=beta
        
    return x,z,gpr,gpc, gp_logger, rho_list

def admmbo_run(problem, x0, max_iter = 100, admmbo_pars = {}, debugging = False): #TODO: implement default max_iter to budjet
    #################################
    K_in = 60 #example0 K = 30


    # For setting the type of grid to use for solving the problem (discreticing space, and then 
    # selectin a less fine grid for less GP calculations)
    num_samples = 301 # in each dimension
    grid_step = 10
    #################################

# Fetching problem info #
    bounds = problem["Bounds"]
    if problem["Bound Type"] == "square":
        xin = np.linspace(bounds[0], bounds[1], num_samples)
        xy = np.array(np.meshgrid(xin, xin, indexing='ij')).reshape(2, -1).T
    else:
        raise Exception("Not yet Implemented non-square bounds")

    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Functions (z)"] #TODO: in code refactor for multiple problems
# --------------------- #

    extent_tuple = bounds #bounds are defined as (lb1,ub1,...,lb#,ub#)
    bounds_array = np.array(bounds).reshape(-1,2)

    # For drawing bounds + true cost function
    # Bound area is then 0 in heatmap
    func = costf(xy)
    if len(constraintf) != 1:
        print("NO PLOTTING SUPPORT FOR MULTIPLE CONSTRAINTS ... YET")
    con = ( constraintf[0](xy) <= 0 )#Boolean constraint
    ff = con*func

    M = np.max(ff) - np.min(ff) # They set it as the unconstrained range of f, while this is the constrained range
                         # ADMMBO is (claimed) not sensitive to M in a wide range of values ref. sec. 5.7 (only to smaller values)
    
    # Grid with evry grid_step-th point of space
    grid = np.array(np.meshgrid(xin[::grid_step],xin[::grid_step],indexing='ij')).reshape(2,-1).T

    #Running ADMMBO
    xo,zo,gpr,gpc, gp_logger, rho_list = admmbo(costf, constraintf, M, bounds_array, grid, x0, 
                                                alpha=2,beta=2, K=K_in, alpha0 = 2, beta0 = 2, rho = 1)

    #Formatting output #TODO: Format so order of queries is correct
    xsr = gpr.X_train_
    obj = gpr.y_train_
    xsc = gpc.base_estimator_.X_train_
    cc = gpc.base_estimator_.y_train_

    if xsr.shape == xsc.shape:
        print("Diff of xs")
        print(xsr-xsc)
    new_obj = np.concatenate((obj,np.zeros(len(cc)))).reshape(-1,1)
    new_cc = np.concatenate((np.ones(len(obj)),cc)).reshape(-1,1)
    obj_out = np.concatenate((new_obj,new_cc),axis = 1) ## SImple combined obj and constraied after eachother

    objmaks = np.concatenate((np.full(len(obj),True),np.full(len(cc),False))).reshape(-1,1)
    constmaks = np.concatenate((np.full(len(obj),False),np.full(len(cc),True))).reshape(-1,1)
    eval_type = np.concatenate((objmaks,constmaks),axis = 1)

    xs_out = np.concatenate((xsr,xsc))
    
    ## TODO:! Move into better flow
    
    # from plotting import vizualize_toy
    # vizualize_toy(
    #     xs_out,
    #     obj_out,
    #     eval_type,
    #     problem,
    #     decoupled = True
    # )
    

    if not debugging:
        return xs_out, obj_out, eval_type
    
    ### Debugging ###
    
    import matplotlib.pyplot as plt
    
    #print(xsc.shape, cc.shape)
    #print(xsr,obj)
    
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
    from opt_problems.ADMMBO_paper_problems import gardner1, gardner2
    from utils.sampling import grid_sampling

    problem = gardner2
    x0 = grid_sampling(problem, num_per_dim = 5) # 5 is original, 3 works
    x0 = np.array([[1.5,4.5],[2.5,3.5]])
    admmbo_run(problem, x0)