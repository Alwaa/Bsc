import subprocess
import os
import json
import numpy as np
from time import time



def pesc_main(name):
    folder_rel="comparisson_algs/PESC_folders/" + name
    ## Reading from config file to parse ##
    f = open(folder_rel + "/config.json")
    config = json.load(f)
    f.close()

    num_vars = len(config["variables"])
    var_names = list(config["variables"].keys())
    num_tasks = len(config["tasks"])
    task_names = list(config["tasks"].keys())
    ## --------------------------------- ##

    out_pros_result = subprocess.run(["conda", "run", "-n", "PESC", "python", 
                               "Spearmint/spearmint/print_all_results.py", folder_rel],shell = True, capture_output=True)
    results = out_pros_result.stdout.split(sep=b"Job ") #Splitting byte output by windows newline #NB! different separator for other OS

    jobs_num = len(results)-1
    xs_out = np.zeros((jobs_num,num_vars))
    objs_out = np.zeros((jobs_num,num_tasks))
    indiv_eval = np.full((jobs_num,num_tasks),False)

    for i, job in enumerate(results):
        fields = job.split(sep=b"\r\r\n")
        if len(fields) == 1:
            continue
        job_count = int(fields[0])
        assert job_count == i, f"Result from PESC not parsed correctly!\n Incorrect itetarion number --{job_count}-- is not --{i}--" #Me being paranoid after the trublesome setup
        #print(job_count)
        fields = fields[3:] #Making indexing easier by removing headers
        var_res = [s.split() for s in fields[:num_vars]]


        for j in range(len(var_res)):
            if var_res[j][0].decode("utf-8") == var_names[-j-1]:
                xs_out[i-1,j] = float(var_res[-j-1][-1].decode("utf-8"))
            else:
                xs_out[i-1,j] = float(var_res[j][-1].decode("utf-8"))
                #f"Variable names outputted in different order than reverse of config\n\n{var_res[j][0].decode('utf-8')}\n{var_names[-j-1]}"
            #print(var_res[-j-1][-1].decode("utf-8"))

        
        if "group" in config["tasks"]["f"].keys():
            tasks_res = fields[num_vars].split()
            task_ran = tasks_res[0].decode("utf-8")[:-1]
            ran_res = float(tasks_res[1][1:-1].decode("utf-8"))
            colmnn = config["tasks"][task_ran]["group"] - 1
            
            objs_out[i-1][colmnn] = ran_res
            indiv_eval[i-1][colmnn] = True
            
            #print(task_ran,ran_res, config["tasks"][task_ran]["group"] - 1)
        else:
            indiv_eval = []
            tasks_res = [s.split() for s in fields[num_vars:num_vars+num_tasks]]
            for j in range(len(tasks_res)):
                if tasks_res[j][0].decode("utf-8")[:-1] == task_names[-j-1]:
                    objs_out[i-1,j] = float(tasks_res[-j-1][-1][1:-1].decode("utf-8"))
                else:
                    objs_out[i-1,j] = float(tasks_res[-j-1][-1][1:-1].decode("utf-8"))
                    #print(f"Task names outputted in different order than reverse of config\n\n{tasks_res[j][0].decode('utf-8')}\n{task_names[-j-1]}")

    objs_out[:,1:] *= -1 #Flipping constraint for now!!!! 
    
    #print(xs_out, "\n", objs_out)
    return xs_out, objs_out, indiv_eval

def pesc_run_experiment(name = "test" ):
    folder_rel = "comparisson_algs/PESC_folders/" + name
    out_pros_result = subprocess.run(["conda", "run", "-n", "PESC", "python", 
                               "Spearmint/spearmint/cleanup.py", folder_rel],shell = True, capture_output=True)
    out_pros_result = subprocess.run(["conda", "run", "-n", "PESC", "python", 
                               "Spearmint/spearmint/main.py", folder_rel],shell = True, capture_output=True)
    print(out_pros_result)

def pesc_create_problem(problem_in, name, max_iter = 80, decoupled = False):
    max_finished_jobs = max_iter #budjet
    p_folder = "comparisson_algs/PESC_folders/" + name
    p_exists = os.path.exists(p_folder)
    if p_exists:
        print("Problem already exsists")
        print("Override not implemented")
        return
    os.makedirs(p_folder)
    
    bounds = problem_in["Bounds"]
    n_var = len(bounds)//2
    con_num = len(problem_in["Constraint Functions (z)"])
    ## Constructing JSON ##
    var_dict = {f"x{i}" : {"type" : "FLOAT",
                            "size" : 1,
                            "min"  : bounds[i*2],
                            "max"  : bounds[i*2 +1]} for i in range(n_var)}
    
    
    if decoupled:
        task_dict = {f"c{i}":{"type": "constraint", "group" : int(i+2),"main_file": f"c{i}"} for i in range(con_num)}
        task_dict["f"] = {"type":"objective", "group" : 1, "main_file": "f"}
    else:
        task_dict = {f"c{i}":{"type": "constraint"} for i in range(con_num)}
        task_dict["f"] = {"type":"objective"}

    json_info = {
        "language"          : "PYTHON",
        "main_file"         : "problem",
        "experiment-name"   : "PESC-" + name,
        "acquisition"       : "PES",
        "max_finished_jobs" : max_finished_jobs,
        "variables" : var_dict,
        "tasks" : task_dict
    }
    # Serializing json
    json_dump = json.dumps(json_info, indent=4)
    with open(p_folder + "/config.json", "w") as outfile:
        outfile.write(json_dump)
    ## ------------------ ##
    
    imp_s = "import numpy as np\n\n\n"
    params_s = [f"params['x{i}']" for i in range(n_var)]
    array_s = "xx = np.array([[" + ",".join(params_s) + "]])"
    f_line = "\n    \n    f = ???"
    c_ln = [f"c{i} = ???" for i in range(con_num)]
    c_lines = "\n    ".join(c_ln)
    c_rs = [f"'c{i}' : np.array([float(c{i}[0][0] <= 0)])" for i in range(con_num)]
    
    if not decoupled:
        r_s = "\n    \n    return " + "{'f':f[0], " + ",".join(c_rs) + "}"
        file = imp_s + "def main(job_id, params):\n    " + array_s + f_line +"\n    " + c_lines + r_s
        
        with open(p_folder + "/problem.py", 'w') as f:
            f.write(file)
        inp = ""
        while inp != "yes":
            inp = input("\nNB!\n"+ p_folder +"/problem.py \nHave you changed the problem.py file with the function and constraints?: ")

    else:
        names = [f"c{i}" for i in range(con_num)]
        for num, file_name in enumerate(names):
            r_s = "\n    \n    return " + "{" + c_rs[num] + "}"
            file = imp_s + "def main(job_id, params):\n    " + array_s + "\n    \n    " + c_ln[num] + r_s

            with open(p_folder + "/"+ file_name +".py", 'w') as f:
                f.write(file)
        r_s = "\n    \n    return " + "{'f':f[0]}"
        file = imp_s + "def main(job_id, params):\n    " + array_s + f_line + r_s
        with open(p_folder + "/f.py", 'w') as f:
            f.write(file)
        
        clicky =[p_folder + f"/c{i}.py" for i in range(con_num)]+ [p_folder +"/f.py"]
        
        names.append("f")
        
        r_d = [f"'{nam}':f.main(*args)['{nam}']" for nam in names]
        r_dict = ",\n    ".join(r_d)
        file = "import " + "\nimport ".join(names) + "\n\ndef main(*args):\n    \n    return {" + r_dict +  "}"
        
        with open(p_folder + "/problem.py", 'w') as f:
            f.write(file)
        
        
        inp = ""
        while inp != "yes":
            inp = input("\nNB!\n"+ "\n".join(clicky) +"\nHave you changed the problem files with the function and constraints?: ")
            
    
        