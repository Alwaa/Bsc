import json
import numpy as np
import os

PERSONAL_PATH = "C:/Users/Alexander/wgit/Bsc/Experiments"
def create_exp_folder(name, path_parent_folder = PERSONAL_PATH):
    p_exists = True
    
    v_num = 0
    while p_exists:
        e_folder = path_parent_folder + f"/{name}{v_num:02d}"
        p_exists = os.path.exists(e_folder)
        v_num += 1
    
    os.makedirs(e_folder)
    return e_folder

def save_exps(list_of_res, alg_name, e_folder, info = "Experiment ran"):
    num_exp = len(list_of_res)
    
    alg_folder = e_folder + f"/{alg_name}"
    if not os.path.exists(alg_folder):
        os.makedirs(alg_folder)
    
    json_dict = {"number": num_exp,
                 "algorithm": alg_name,
                 "info": info}
    
    ## Serializing json ##
    json_dump = json.dumps(json_dict, indent=4)
    print(alg_folder)
    with open(alg_folder + "/exp.json", "w") as outfile:
        outfile.write(json_dump)
    ## ------------------ ##
    
    for exp_num in range(num_exp):
        outfile = alg_folder + f"/{exp_num:03d}"
        xs,objs,indivs = list_of_res[exp_num]
        np.savez(outfile, xs, objs, indivs)

    
def fol(name, v_num, path_parent_folder = PERSONAL_PATH):
    return path_parent_folder + f"\\{name}{v_num:02d}"