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

nun_samples = 301
point_per_axis_start = 7 #7 is original, 5 works for small feas area problem
#################################

# Fetching problem info #
bounds = problem["Bounds"]
if problem["Bound Type"] == "square":
    xin = np.linspace(bounds[0], bounds[1], 301)
    xy = np.array(np.meshgrid(xin, xin, indexing='ij')).reshape(2, -1).T
else:
    raise Exception("Not yet Implemented non-square bounds")

costf = problem["Cost Function (x)"]
constraintf = problem["Constraint Function (z)"]
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


def gpr_ei(x,y,z,rho,gpr,ubest):
    x=np.atleast_2d(x)
    mu,std = gpr.predict(np.atleast_2d(x), return_std=True)
    muu = mu + 0.5*rho*np.sum((x-z[None]+y[None]/rho)**2,axis=1)
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
def admmbo(cost, constraint, M, bounds, grid, x0, f0=None, c0=None,
           K=15, rho=0.1, alpha=5, beta=5, alpha0=20, beta0=20,
           epsilon=1e-6):
    K1 = gaussian_process.kernels.ConstantKernel(constant_value_bounds=(1e-10,1e10)) * \
            gaussian_process.kernels.Matern(nu=1.5,length_scale_bounds=(1e-2, 1e2)) #+\
                # gaussian_process.kernels.WhiteKernel()
    gpr = gaussian_process.GaussianProcessRegressor(kernel=K1)
    
    K2 = gaussian_process.kernels.ConstantKernel(constant_value_bounds=(1e-10,1e10)) * \
        gaussian_process.kernels.Matern(nu=1.5,length_scale_bounds=(1e-2, 1e2)) 
    gpc = gaussian_process.GaussianProcessClassifier(kernel=K2)
        
    S = False
    k = 0 # "k=1 (0 as first index)"
    
    ## rho adjusting parameters ##
    tau = 2
    mup = 10
    ## ------------------------ ##

    ## Initializing z,y ## 
    # in the top corner, per meeting notes
    bounds=bounds.astype('float')
    z = bounds[:,1].copy()
    y = bounds[:,1].copy()
    print(z)
    # z = np.array((1,1))
    # z = x0[c0==0][np.argmin(f0[c0==0])]
    # y = np.ones(x0.shape[1])
    # y = np.zeros(x0.shape[1])
    zold = z.copy()
    ## ---------------- ##

    if f0 is None:
        f0 = cost(x0)
    if c0 is None:
        c0 = constraint(x0)

    print(x0,f0)
    print(x0,c0)


    gpr.fit(x0,f0)
    #TODO: Mthod where only one class can be present
    # Simply removing limitation to one of each class present did not do anything 
    gpc.fit(x0,c0)     
    
    # z=np.mean(gpc.base_estimator_.X_train_[gpc.base_estimator_.y_train_==0],0)
    alphac=alpha0
    betac=beta0
    while k < K and not S:
        ### OPT ###
        for t in range(alphac):
            ## Eq. (10) ##
            u = gpr.y_train_ + \
                0.5*rho*np.sum((gpr.X_train_-z[None]+y[None]/rho)**2,axis=1)
            ubest = np.min(u)
            ## -------- ##
            eif = lambda x: -gpr_ei(x,y,z,rho,gpr,ubest)
            mu,std = gpr.predict(grid, return_std=True)
            muu = mu + 0.5*rho*np.sum((grid-z[None]+y[None]/rho)**2,axis=1)
            xx = (ubest-muu)/std
            ei = std * (xx * ndtr(xx) + norm.pdf(xx))
            x = grid[np.argmax(ei)]
            opt = minimize(eif, x, bounds=bounds)
            x = opt.x
            # print(f'eix:{ei.max()}')

            # TODO: Output of costf should be same shape as constraintf always?
            # print(x[None],"\n-----\n",[cost(x)])
            print(x[None],"\n-----\n",gpr.y_train_)
            gpr.fit(np.concatenate((gpr.X_train_,x[None]),axis=0),
                    np.concatenate((gpr.y_train_,cost(x[None])),axis=0))
        ### --- ###
        ### FEAS ###
        u = gpr.y_train_ + \
                0.5*rho*np.sum((gpr.X_train_-z[None]+y[None]/rho)**2,axis=1)
        x = gpr.X_train_[np.argmin(u)]
        for t in range(betac):
            #Line 4 in Alg. 3.3
            h = gpc.base_estimator_.y_train_ + \
                0.5*rho/M*np.sum((x[None]-gpc.base_estimator_.X_train_
                                  +y[None]/rho)**2,axis=1)
            hbest = np.min(h)
            
            eif = lambda z: -gpc_ei(x,y,z,rho,M,gpc,hbest)
            qi = lambda zi: 0.5*rho/M * np.sum((x[None]-zi+y[None]/rho)**2,axis=1)
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
                                  +y[None]/rho)**2,axis=1)
        z=gpc.base_estimator_.X_train_[np.argmin(h)]
        print(f"{x}\n{z}\n{rho}")
        y += rho * (x - z)
        print(f'x: {x} \nz: {z} \ny: {y}')
        r = (x - z)**2
        rl=np.sqrt(np.sum(r**2))
        s = - rho * (z - zold)
        sl = np.sqrt(np.sum(s**2))
        # print(r)
        # print(s)

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
        zold = z.copy()
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
    con = 1 - constraintf(xy)
    ff = con*func

    ## Starting Points##
    # x0 = (rnd.rand(5,2)-.5)*2
    xx0=np.linspace(bounds[0],bounds[1],point_per_axis_start)[1:-1]

    x0 = np.array(np.meshgrid(xx0,xx0)).reshape(2,-1).T

    M = np.max(np.abs(ff)) # They set it as the unconstrained range of f, while this is the constrained range
    #                           ADMMBO is not sensitive to M in a wide range of values ref. sec. 5.7
    
    # Grid with evry 10th point of space
    grid = np.array(np.meshgrid(xin[::10],xin[::10],indexing='ij')).reshape(2,-1).T

    xo,zo,gpr,gpc=admmbo(costf, constraintf, M, bounds_array, grid, x0,alpha=2,beta=2,K=30)
    
    import matplotlib.pyplot as plt
    ### Main Plot ###
    plt.figure()
    plt.imshow(ff.reshape(len(xin),-1).T,extent=(extent_tuple),origin='lower')
    plt.colorbar()
    plt.plot(gpr.X_train_[:,0],gpr.X_train_[:,1],'kx')
    for i in range(gpr.X_train_.shape[0]):
        plt.text(gpr.X_train_[i,0],gpr.X_train_[i,1],f'{i}')
    xc=gpc.base_estimator_.X_train_[:,0]
    yc=gpc.base_estimator_.X_train_[:,1]
    cc=gpc.base_estimator_.y_train_
    plt.plot(xc[cc==1],yc[cc==1],'r+')
    plt.plot(xc[cc==0],yc[cc==0],'ro')
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
    plt.figure();plt.imshow(func.reshape(len(xin),-1).T,extent=(extent_tuple),origin='lower')
    plt.colorbar();plt.title('True Cost/Objective Function')
    ### ---- ###
    
    # plt.figure();plt.imshow(ei.reshape(301,-1).T,origin='lower')

    # Needed for VScode
    plt.show()
    