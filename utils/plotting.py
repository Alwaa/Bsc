import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy.typing import NDArray
import copy
import os
import pandas as pd

NUM_OF_OBJECTIVE_FUNCTIONS = 1 #Multi-objective optimization is not covered. One should conbine and weigh the multiple objectives into one in thoscases

def vizualize_toy(xs: NDArray[np.float64],
                  objs: NDArray[np.float64], #objective values
                  eval_type: NDArray[np.bool8],
                  problem,
                  decoupled: bool = False,
                  title = ""):
    num_samples = 301
    ## Fetching problem info ##
    bounds = problem["Bounds"]
    if problem["Bound Type"] == "square":
        xin = np.linspace(bounds[0], bounds[1], num_samples)
        xy = np.array(np.meshgrid(xin, xin, indexing='ij')).reshape(2, -1).T
    else:
        raise Exception("Not yet Implemented non-square bounds")
    
    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Functions (z)"] #TODO: rewrite to multiple (s)
    ## ---------------------- ##
    cc = np.all(
        objs[:,1:],
        axis=1) #axis 1 since it is per point not per constraint as is it below 

    extent_tuple = (bounds[0],bounds[1],bounds[0],bounds[1])

    
    func = costf(xy)
    con = np.all(
            np.array([constraintf[i](xy) <= 0  for i in range(len(constraintf))]),
            axis=0) #Boolean constraints
    ff = con*func

    #Maybe merge both decoupled and not into same flow by creating a full eval_type mask?
    decoupled = not len(eval_type) == 0

    print("DECOUPLED = ", decoupled)
    
    if decoupled:
        assert eval_type.shape == objs.shape, "Eval type mask should batch one to one the obj"
        assert NUM_OF_OBJECTIVE_FUNCTIONS == 1, "Not implemented"

        o1mask = eval_type[:,0]
        class_mask = np.any(eval_type[:,1:],axis=1)

        xs_obj = xs[o1mask]
        xs_class = xs[class_mask]
        cc = cc[class_mask]
        o1_vals = objs[:,0][o1mask]
    else:
        xs_obj = xs
        xs_class = xs
        o1_vals = objs[:,0]
        
        
    def check_validity(xx):
        cons = np.array([constraintf[i](xx) <= 0 for i in range(len(constraintf))])#Boolean constraint #Upheld if over 0 ??
        return np.all(cons,axis=0)
    
    cc = np.logical_not(check_validity(xs_class)) #TODO: check output of obj_matrix

    
    ### Main Plot ###
    plt.figure()
    plt.title(title)
    plt.imshow(ff.reshape(len(xin),-1).T,extent=(extent_tuple),origin='lower')
    plt.colorbar()
    plt.plot(xs_obj[:,0],xs_obj[:,1],'kx') #WWhere objective function was evaluated
    for i in range(xs_obj.shape[0]):
        plt.text(xs_obj[i,0],xs_obj[i,1],f'{i}')

    plt.plot(xs_class[:,0][cc>0],xs_class[:,1][cc>0],'r+') #Where classification was attempted but is incorrect
    plt.plot(xs_class[:,0][cc<=0],xs_class[:,1][cc<=0],'ro') #Where it is correct
    ### --------- ###

    #Objfunction plot
    plt.figure()
    plt.title(title)
    sampl_it = np.arange(len(o1_vals))
    valid_it, o1_valid = sampl_it[check_validity(xs_obj)], o1_vals[check_validity(xs_obj)]
    plt.plot(sampl_it, o1_vals,"kx")
    plt.plot(valid_it, o1_valid,"mo")
    if len(o1_valid) > 0:
        curr_min = o1_valid[0]
        rolling_min = np.ones(len(o1_valid))
        for i in range(len(o1_valid)):
            if curr_min > o1_valid[i]:
                curr_min = o1_valid[i]
            rolling_min[i] = curr_min
        plt.plot(valid_it,rolling_min,"r--") #TODO: Fix indexing of valid line to drop off abruplty (per it.)

    # Needed for VScode
    plt.show()

def vizualize_toy_problem(problem, points_per_dim = 400, name = " "):
    costf = problem["Cost Function (x)"]
    constraintf = problem["Constraint Functions (z)"]
    bounds = problem["Bounds"]

    num_samples, dim_num = points_per_dim, len(bounds)//2
    xins = (np.linspace(bounds[i*2], bounds[1+i*2], num_samples) for i in range(dim_num))
    xy = np.array(np.meshgrid(*xins, indexing='ij')).reshape(dim_num, -1).T
    #could make into per 2 axis plots

    func = costf(xy)
    con = np.all(np.array([constraintf[i](xy) <= 0  for i in range(len(constraintf))]),
                axis = 0) #Boolean constraints
    ff = con*func
    

    constr_i = np.argmin(func[con])
    glbl_i = np.argmin(func)
    constr_x, constr_obj = xy[con][constr_i], func[con][constr_i]
    glbl_obj = func[glbl_i]
    glbl_x = xy[glbl_i]
    glbl_xs = [] # xy[np.argwhere(func<= glbl_obj + 0.001)]
    
    prct_feasible = 100*(np.sum(con)/(points_per_dim**dim_num))
    
    if dim_num == 2:
        if problem["Bound Type"] == "square":
            xin = np.linspace(bounds[0], bounds[1], num_samples)
        else:
            raise Exception("Not yet Implemented non-square bounds")
        plt.figure()
        plt.imshow(ff.reshape(len(xin),-1).T,extent=(bounds),origin='lower')
        plt.colorbar();plt.title('Constrained Cost Function')
        ## Aera lower than constrained optima
        mask_better = np.logical_and(np.logical_not(con), func <= constr_obj) 
        better_xs = xy[mask_better]
        plt.plot(better_xs[:,0],better_xs[:,1], 'mx', markersize = 50/points_per_dim)
        ## Optima
        plt.plot(glbl_x[0], glbl_x[1], 'b*')
        plt.text(glbl_x[0], glbl_x[1], f"{glbl_obj:.1f}")
        for glbl_x in glbl_xs:
            plt.plot(glbl_x[0][0], glbl_x[0][1], 'b*')
            plt.text(glbl_x[0][0], glbl_x[0][1], f"{glbl_obj:.1f}")
        
        
        plt.plot(constr_x[0], constr_x[1], 'r*')
        plt.text(constr_x[0], constr_x[1], f"{constr_obj:.1f}")
    

    print(f"Best Value: {constr_obj}")
    print(f"Prct Feasible: {prct_feasible}%")
    plt.title(name)
    # Needed for VScode
    plt.show()
    #Coould (maybe) add local (maybe) optima to plot. Both constrained and unconstrained
    pass


def expretiment_plot(   exps, problem, e_folder,
                        title = "Comparisson plot",
                        override = False,
                        name_from_to = {},
                        quantile = 0.1,
                        just_mean = False, 
                        xlim = 124, 
                        split_starting = False):
    ## Fetching problem info ##
    constraintf = problem["Constraint Functions (z)"]
    best_value = problem.get("Best Value", None)
    ## ---------------------- ##
    def check_validity(xx):
        cons_list = [constraintf[i](xx) <= 0 for i in range(len(constraintf))]
        cons = np.array(cons_list)#Boolean constraint #Upheld if over 0 ??
        return np.all(cons,axis=0)
    
    # Making cache folder
    p_fol = e_folder + "\plot_cache"
    if not "plot_cache" in os.listdir(e_folder):
        os.makedirs(p_fol)
        
    point_comps = {};alg_run_lengs_min = 1e10
    comps = exps #"legacy" name
    
    ##-- Hacked together plotting adjustment --##
    # Reserving colors for main comparisson algs CMA-ES, COBYLA (Multiple), PESC
    colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.XKCD_COLORS)
    reserved_num = 0
    for alg_name in exps.keys():
        if alg_name in ['cma','cobyla', 'pesc']:
            reserved_num +=1
    reserved = ["blue", "orange", "green"][:reserved_num];res_counter = 0#copy(colors[:3])
    colors = colors[reserved_num:]
    admmbo_plotted = False #For only plotting one ADMMBO in feasible
    
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.xlim(left=-2)
    plt.xlim(right=xlim)
    plot_num = 0
    ##-----------------------------------------##
    
    for alg_name, exp_list in comps.items():
        num_exp = len(exp_list)
        if num_exp < 3: 
            continue #Not worth plotting if too few
        
        file = p_fol + f"/{alg_name}.npz" #For saving and loading
        if override or not os.path.exists(file):
            decoupled = not len(exp_list[0][2]) == 0
            roll_min_list = []
            max_roll_min_len = 0
            plot_iters = []
            candidate_points, candidate_objs = [], []
            for xs, objs, eval_type in exp_list:
                if decoupled: 
                    #assert eval_type.shape == objs.shape, "Eval type mask should batch one to one the obj"
                    if split_starting: #Correctly displaying the initial samples being fully sampled on value and constraints
                        starting_mask = np.sum(eval_type,axis=1) > 1
                        try:
                            sampl_start = np.where(np.logical_not(starting_mask))[0][0] #Last +1 or starting bunch
                            starting_mask[sampl_start:] = False
                            start_xs = xs[starting_mask,:]
                            start_objs = objs[starting_mask,:]
                            start_et = eval_type[starting_mask,:]
                            
                            xs_empty = np.empty((sampl_start*2,xs.shape[1]), dtype=xs.dtype)
                            objs_empty = np.empty((sampl_start*2,objs.shape[1]), dtype=objs.dtype)
                            et_empty = np.empty((sampl_start*2,eval_type.shape[1]), dtype=eval_type.dtype)
                            
                            xs_empty[0::2] = start_xs
                            xs_empty[1::2] = start_xs
                            objs_empty[0::2] = start_objs
                            objs_empty[1::2] = start_objs
                            et_empty[0::2] = start_et
                            et_empty[1::2] = start_et                        
                            
                            objs = np.concatenate((objs_empty, objs[sampl_start:]), axis = 0)
                            xs = np.concatenate((xs_empty ,xs[sampl_start:]), axis = 0)
                            eval_type = np.concatenate((et_empty, eval_type[sampl_start:]), axis = 0)
                        except:
                            print("hmm")
                        
                    o1mask = eval_type[:,0]
                    class_mask = np.any(eval_type[:,1:],axis=1)
                    xs_class = xs[class_mask]
                else:
                    o1mask = np.full(len(xs),True)
                    xs_class = xs

                xs_obj = xs[o1mask]
                o1_vals = objs[:,0][o1mask]

                #Objfunction rolling min
                sampl_it = np.arange(len(o1_vals))
                total_it = np.arange(len(xs))
                valid_objxs_mask = check_validity(xs_obj)
                o1_valid =  o1_vals[valid_objxs_mask]
                xs_obj_valid = xs_obj[valid_objxs_mask]
                #valid_it = sampl_it[check_validity(xs_obj)]
                valid_totxs_mask = check_validity(xs)
                tot_valid_it = total_it[o1mask][valid_objxs_mask] #total_it[np.logical_and(check_validity(xs), o1mask)]
                tot_roll_min = np.full(len(total_it), np.nan)
                #smpl_roll_min = np.full(len(sampl_it), np.nan)
                # print(tot_valid_it,exp_list[0][2])

                for it in range(len(o1_valid)):
                    tot_roll_min[tot_valid_it[it]] = o1_valid[it]

                #Rolling min
                roll_min_len = len(tot_roll_min)
                if len(o1_valid) > 0:
                    a_m = np.argmax(o1_valid)
                    curr_min, cand_x = o1_valid[a_m], xs_obj_valid[a_m]
                    assert np.all(check_validity(xs_obj_valid)), "something wrong"
                    for i in range(1,roll_min_len):
                        if tot_roll_min[i-1] < tot_roll_min[i] or np.isnan(tot_roll_min[i]):
                            tot_roll_min[i] = tot_roll_min[i-1]
                        if tot_roll_min[i-1] < curr_min and not np.isnan(tot_roll_min[i-1]):
                            cand_x, curr_min = xs[i-1], tot_roll_min[i-1]
                            if not cand_x in xs_obj_valid:
                                print("|", end = "-", flush = True)

                    candidate_points.append(cand_x);candidate_objs.append(curr_min)
                roll_min_list.append(tot_roll_min)
                if max_roll_min_len < roll_min_len:
                    max_roll_min_len = roll_min_len
                    plot_iters = total_it
            
            #Prob overkill and not needed
            ## resizing to be same size on all runs ##
            for i in range(len(roll_min_list)):
                if len(roll_min_list[i]) < max_roll_min_len:
                    roll_min_list[i] = np.concatenate(
                        (roll_min_list[i],np.full(max_roll_min_len - len(roll_min_list[i]) ,roll_min_list[i][-1])) #pad with last element instead of nan?
                    )
                elif len(roll_min_list[i] > max_roll_min_len): 
                    roll_min_list[i] = copy.deepcopy(roll_min_list[i][:max_roll_min_len])
            ## ------------------------------------  ##
            
            if not decoupled:
                plot_iters *= (1 + len(constraintf))
            
            arr_roll_mins = np.array(roll_min_list)
            stds = np.nanstd(arr_roll_mins, axis=0)
            means = np.nanmean(arr_roll_mins, axis=0)
            
            if quantile <= 0:
                uq,lq = None,None
            else:
                uq = np.nanquantile(arr_roll_mins, max(quantile,1-quantile), axis=0)
                lq = np.nanquantile(arr_roll_mins, min(quantile,1-quantile), axis=0)


            ## How many found feasbale ##
            non_feasible = np.isnan(arr_roll_mins[:,-1])
            tot_runs = len(non_feasible)
            feasible = tot_runs - np.sum(non_feasible)
            feas_prct = 100*feasible/tot_runs        
            arr_feas = copy.deepcopy(arr_roll_mins)
            arr_feas[np.logical_not(np.isnan(arr_roll_mins))] = 1
            arr_feas[np.isnan(arr_roll_mins)] = 0        
            feas_per_it = np.sum(arr_feas, axis = 0)/tot_runs
            
            print(f"{alg_name} found {feas_prct:.1f}% feasible ({feasible}/{tot_runs})")
            ## ----------------------- ##
            x_b, x_objs =  np.array(candidate_points),np.array(candidate_objs)
            np.savez(file, pi = plot_iters, m = means, s = stds, uq = uq, lq = lq,
                     f = feas_per_it, xi = x_b, x_o = x_objs)
        else:
            data = np.load(file)
            plot_iters, means, stds, feas_per_it = data["pi"], data["m"], data["s"], data["f"]
            x_b, x_objs = data["xi"], data["x_o"]
            uq, lq, = data.get("uq", None), data.get("lq", None)
            if quantile < 0:
                uq,lq = None,None
        
        ## Prettyfying names ##
        for nfrom, nto in name_from_to.items():
            alg_name = alg_name.replace(nfrom,nto)
        
        
        #print(feas_per_it)
        plt.figure(1) #Not best practice!
        #plt.errorbar(plot_iters, means,yerr=stds)
        if alg_name in ['CMA-ES','COBYLA (Multiple)', 'PESC']:
            curr_color = reserved[res_counter];res_counter += 1
        else:
            curr_color  = colors[plot_num]
        plt.plot(plot_iters,means, color = curr_color, label = alg_name)
        if just_mean:
            no_quantile = False
        else:
            if lq is None:
                no_quantile = True
                #plt.fill_between(plot_iters, means - stds, means + stds, color= curr_color, alpha=0.2)
            else:
                no_quantile = False
                plt.fill_between(plot_iters, lq, uq, color= curr_color, alpha=0.2)
        plt.figure(2) #Not best practice!
        admmbo_in = alg_name.lower().find("admmbo") != -1
        if not admmbo_plotted or not admmbo_in:
            label = "All ADMMBO" if admmbo_in else alg_name
            plt.plot(plot_iters,feas_per_it, color = curr_color, label = label)
        admmbo_plotted = admmbo_in
        ## ----------------------- ##
        
        plot_num += 1
        top10 = np.argsort(x_objs)[:10]
        top10obj = x_objs[top10]
        print("\n", alg_name, ":")
        
        for o_i, arr in enumerate(np.round(x_b[top10[:5]],decimals=2)):
            print(list(arr), "\t"*2, top10obj[o_i], "\t"*2, check_validity(arr))

        means_out = np.empty(plot_iters[-1])
        prev = 0
        for en, plot_it in enumerate(plot_iters):
            means_out[prev:plot_it] = means[en]
            prev = plot_it
        point_comps[alg_name] = means_out
        alg_run_lengs_min = min(len(means_out),alg_run_lengs_min)
    
    if True:
        print("\n\n", "-"*40)
        dfd = {}
        l = alg_run_lengs_min
        for name,mean in point_comps.items():
            dfd[name] = [mean[int(l*0.4)],mean[l-1]] #Could also add feasability...
        col_names = [f"40% ({int(l*0.4)})", f"100% ({int(l-1)})"]
        df = pd.DataFrame(data=dfd).T
        df.columns = col_names
        print("40% Sort\n",df.sort_values(col_names[0]), "\n\n")
        print("100% Sort\n",df.sort_values(col_names[1]))
    
        if no_quantile:
            title += " (st.d)"
    
    plt.figure(1) #Not best practice!
    plt.title(title)
    plt.xlabel("Function Evaluations")
    plt.ylabel("Objective Value")
    plt.legend()
    if not best_value is None:
        plt.axhline(y = best_value, color = 'k', linestyle = '--')
    
    plt.figure(2) #Not best practice!
    plt.title("Feasability Rercentage")
    plt.legend()
    plt.rcParams['figure.figsize'] = [6, 6]  #Not best practice! #Not best practice!

def exploration_xs_plot(xx):
    col_number = xx.shape[0]
    
    for col in range(col_number):
        plt.figure()
        plt.plot(xx[:,col], label = f"x{col}")

def exploration_hist(exp_list, name = "Algorithm"):
        all_xs = []
        for xs, _,_  in exp_list:
            all_xs.append(xs)
        xplored = np.concatenate(all_xs, axis = 0)
        col_number = xplored.shape[1]
        lables = [f"x{col}" for col in range(col_number)]
        if col_number > 5:
            start_end_list = [(i,min(i+3,col_number)) for i in range(0,col_number,3)]
        for st, en in start_end_list:
            plt.figure()
            plt.hist(xplored[:,st:en], bins = 40, label=lables[st:en])
            plt.title(f"Distribution of sampling of -{name}-")
            plt.legend()
           