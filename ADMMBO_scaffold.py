# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:10:42 2022

@author: khm
"""

import numpy as np
from scipy.optimize import minimize
from sklearn import gaussian_process
# import GPyOpt
# from GPyOpt import BayesianOptimization
from scipy.special import ndtr
from scipy.stats import norm



rnd = np.random.RandomState(42)

from opt_problems.example_problems import example0
from opt_problems.ADMMBO_paper_problems import gardner1, gardner2

#################################
# Problem to solve
problem = gardner1
K_in = 10 #example0 K = 30

start_sample_type = "grid"
point_per_axis_start = 7 #7 is original, 5 works for small feas area problem

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
constraintf = problem["Constraint Function (z)"] #TODO: rewrite to multiple (s)
# --------------------- #
class BO: #Unused for now
    def __init__(self,func,x0=None,f0=None,discrete=False):
        self.kernel=gaussian_process.kernels.ConstantKernel() * \
            gaussian_process.kernels.Matern(nu=1.5)
        
        self.gp = gaussian_process.GaussianProcessRegressor(kernel=self.kernel)
        self.func=func
        self.x = []
        self.x.extend(x0)
        if self.f0 is None:
            self.f=[self.func(x) for x in x0]
        else:
            self.f.extend(f0)
        self.gp.fit(self.x,self.f)
    
    def update(self,x,f=None):
        self.x.append(x)
        if f is None:
            self.f.append(self.func(x))
        self.gp.fit(self.x,self.f)
    
    def EI(self,x):
        mu,std = self.gp.predict(x)
        xx = (mu - self.cmax)/std
        ei = std * (xx * ndtr(xx) + norm.pdf(xx))
        return ei
    
    def eimax(self,x):
        return x[np.argmax(self.EI(x))]


def gpr_ei(x,ys,zs,rho,gpr,ubest):
    x=np.atleast_2d(x)
    mu,std = gpr.predict(np.atleast_2d(x), return_std=True)
    muu = mu + 0.5*rho*np.sum((x-zs+ys/rho)**2,axis=1)
    xx = (ubest-muu)/std
    ei = std * (xx * ndtr(xx) + norm.pdf(xx))
    return ei

def gpc_ei(x,y,z,rho,M,gpc,hbest):
    z=np.atleast_2d(z)
    hh = hbest - 0.5*rho/M * np.sum((x[None]-z+y[None]/rho)**2,axis=1)
    theta = gpc.predict_proba(z)[:,1]
    ei = np.zeros(z.shape[0])
    idx1 = (hh>0) & (hh<1)
    ei[idx1] = hh[idx1] * (1.0 - theta[idx1])
    idx2 = hh > 1.0
    ei[idx2] = hh[idx2] - theta[idx2]
    return ei


# bounds = search-space B ; n and m_i missing, insted the initial points are passed in
# M = penalty of infeasability
# K = max number of it; rho = penalty parm
# epsilon = tolerance parameter for stopping rule
# delta missing, 1 - delta is acceptance for prob that final solution is infeasable, for parameter to use if it does not converge
def admmbo(cost, constraints, M, bounds, grid, x0, f0=None, c0=None,
           K=15, rho=0.1, alpha=5, beta=5, alpha0=20, beta0=20,
           epsilon=1e-6):
    K1 = gaussian_process.kernels.ConstantKernel(constant_value_bounds=(1e-10,1e10)) * \
            gaussian_process.kernels.Matern(nu=1.5,length_scale_bounds=(1e-2, 1e2)) #+\
                # gaussian_process.kernels.WhiteKernel()
    gpr = gaussian_process.GaussianProcessRegressor(kernel=K1)
    
    K2 = gaussian_process.kernels.ConstantKernel(constant_value_bounds=(1e-10,1e10)) * \
        gaussian_process.kernels.Matern(nu=1.5,length_scale_bounds=(1e-2, 1e2)) 
        
    S = False
    k = 0 # "k=1 (0 as first index)"
    N = len(constraints) # As in paper

    #Initialicing GP classifiers
    #? Should have individual kernels based on bounds dimenstion?
    #TODO: Follow up on Kernel investigation
    #    TODO: OPT; Method where only one class can be present
    gpcs = [gaussian_process.GaussianProcessClassifier(kernel=K2) for i in range(N)]
    
    ## rho adjusting parameters ## # TODO Investigate further into rho setting
    tau = 2
    mup = 10
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


    # Simply removing limitation to one of each class present did not do anything 
        
    
    # z=np.mean(gpc.base_estimator_.X_train_[gpc.base_estimator_.y_train_==0],0)
    alphac=alpha0
    betac=beta0
    while k < K and not S:
        ### OPT ###
        for t in range(alphac):
            ## Eq. (10) ##
            u = gpr.y_train_ + \
                0.5*rho*np.sum((gpr.X_train_-zs+ys/rho)**2,axis=1) #TODO: Check the sum funciton works propperly with new zs and ys
            ubest = np.min(u)
            ## -------- ##
            eif = lambda x: -gpr_ei(x,ys,zs,rho,gpr,ubest)
            mu,std = gpr.predict(grid, return_std=True)
            muu = mu + 0.5*rho*np.sum((grid-zs+ys/rho)**2,axis=1)  #TODO: Check the sum funciton works propperly
            xx = (ubest-muu)/std
            ei = std * (xx * ndtr(xx) + norm.pdf(xx))
            x = grid[np.argmax(ei)]
            opt = minimize(eif, x, bounds=bounds)
            x = opt.x
            # print(f'eix:{ei.max()}')

            #print(x[None],"\n-----\n",gpr.y_train_)
            #print(cost(x[None]),"\n-----\n",gpr.y_train_)
            gpr.fit(np.concatenate((gpr.X_train_,x[None]),axis=0),
                    np.concatenate((gpr.y_train_,cost(x[None])),axis=0))
        ### --- ###

        # TODO: check nececisty of this part
        u = gpr.y_train_ + \
                0.5*rho*np.sum((gpr.X_train_-zs+ys/rho)**2,axis=1)
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
                
                eif = lambda z: -gpc_ei(x,ys[i],zs[i],rho,M,gpc,hbest) #TODO: Chech correctness with ys and zs
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
        if rl > (mup*sl):
            rho *= tau
        elif sl > (mup*rl):
            rho /= tau
        print(f' rho:{rho} \n r:{rl} \n s:{sl}')
        ## ----------------- ##
        if S:
            break
        alphac=alpha
        betac=beta
        
    return x,z,gpr,gpc

if __name__=='__main__':
    extent_tuple = (bounds[0],bounds[1],bounds[0],bounds[1])
    bounds_array = np.array((extent_tuple[:2],extent_tuple[2:])) #For now, fix later

    # For drawing bounds + true cost function
    # Bound area is then 0 in heatmap
    func = costf(xy)
    if len(constraintf) != 1:
        print("NO PLOTTING SUPPORT FOR MULTIPLE CONSTRAINTS")
    con = ( constraintf[0](xy) <= 0 )#Boolean constraint
    ff = con*func

    ## Starting Points ##
    if start_sample_type == "grid":
        xx0=np.linspace(bounds[0],bounds[1],point_per_axis_start)[1:-1]
        x0 = np.array(np.meshgrid(xx0,xx0)).reshape(2,-1).T
    else: 
        # x0 = (rnd.rand(5,2)-.5)*2
        raise Exception("Non-grid start samples not yet implemted")
    ## --------------- ##

    M = np.max(np.abs(ff)) # They set it as the unconstrained range of f, while this is the constrained range
    #                           ADMMBO is (claimed) not sensitive to M in a wide range of values ref. sec. 5.7
    
    # Grid with evry grid_step-th point of space
    grid = np.array(np.meshgrid(xin[::grid_step],xin[::grid_step],indexing='ij')).reshape(2,-1).T

    xo,zo,gpr,gpc=admmbo(costf, constraintf, M, bounds_array, grid, x0,alpha=2,beta=2,K=K_in)

    xsr = gpr.X_train_
    obj = gpr.y_train_
    xsc = gpc.base_estimator_.X_train_
    cc = gpc.base_estimator_.y_train_


    new_obj = np.concatenate((obj,np.zeros(len(cc)))).reshape(-1,1)
    new_cc = np.concatenate((np.ones(len(obj)),cc)).reshape(-1,1)
    obj_out = np.concatenate((new_obj,new_cc),axis = 1) ## SImple combined obj and constraied after eachother

    objmaks = np.concatenate((np.full(len(obj),True),np.full(len(cc),False))).reshape(-1,1)
    constmaks = np.concatenate((np.full(len(obj),False),np.full(len(cc),True))).reshape(-1,1)
    eval_type = np.concatenate((objmaks,constmaks),axis = 1)

    xs_out = np.concatenate((xsr,xsc))
    ## TODO:! Move into better flow
    from plotting import vizualize_toy
    vizualize_toy(
        xs_out,
        obj_out,
        problem,
        eval_type = eval_type,
        decoupled = True
    )

    #print(xsc.shape, cc.shape)
    #print(xsr,obj)
    
    import matplotlib.pyplot as plt
    ### Main Plot ###
    plt.figure()
    plt.imshow(ff.reshape(len(xin),-1).T,extent=(extent_tuple),origin='lower')
    plt.colorbar()
    plt.plot(xsr[:,0],xsr[:,1],'kx')
    for i in range(xsr.shape[0]):
        plt.text(xsr[i,0],xsr[i,1],f'{i}')

    plt.plot(xsc[:,0][cc==1],xsc[:,1][cc==1],'r+') #Where classification was attempted but is incorrect
    plt.plot(xsc[:,0][cc==0],xsc[:,1][cc==0],'ro') #Where it is correct
    ### --------- ###

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
    
    # plt.figure();plt.imshow(ei.reshape(301,-1).T,origin='lower')

    # Needed for VScode
    plt.show()
    