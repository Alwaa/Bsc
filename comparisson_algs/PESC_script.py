import subprocess
import os
import json
import numpy as np
from time import time



def PESC_main(name):
    folder_rel="comparisson_algs/PESC_folders/" + name
    start_t = time()
    ## For testing
    # conda run -n PESC python --version
    out_pros = subprocess.run(["conda", "run", "-n", "PESC", "python", "--version"],shell = True, capture_output=True)
    print(out_pros);out1_t = time()

    ## Reading from config file to parse ##
    f = open(folder_rel + "/config.json")
    config = json.load(f)
    f.close()
    print(config)

    num_vars = len(config["variables"])
    var_names = list(config["variables"].keys())
    print(num_vars)
    num_tasks = len(config["tasks"])
    task_names = list(config["tasks"].keys())
    print(num_tasks)
    ## --------------------------------- ##

    out_pros_result = subprocess.run(["conda", "run", "-n", "PESC", "python", 
                               "Spearmint/spearmint/print_all_results.py", folder_rel],shell = True, capture_output=True)
    results = out_pros_result.stdout.split(sep=b"Job ") #Splitting byte output by windows newline #NB! different separator for other OS
    out2_t = time()

    jobs_num = len(results)-1
    xs_out = np.zeros((jobs_num,num_vars))
    objs_out = np.zeros((jobs_num,num_tasks))

    for i, job in enumerate(results):
        fields = job.split(sep=b"\r\r\n")
        if len(fields) == 1:
            continue
        job_count = int(fields[0])
        assert job_count == i, f"Result from PESC not parsed correctly!\n Incorrect itetarion number --{job_count}-- is not --{i}--" #Me being paranoid after the trublesome setup
        print(job_count)
        fields = fields[3:] #Making indexing easier by removing headers
        var_res = [s.split() for s in fields[:num_vars]]

        tasks_res = [s.split() for s in fields[num_vars:num_vars+num_tasks]]

        for j in range(len(var_res)):
            #assert var_res[j][0].decode("utf-8") == var_names[-j-1], f"Variable names outputted in different order than reverse of config\n\n{var_res[j][0].decode('utf-8')}\n{var_names[-j-1]}"
            print(var_res[-j-1][-1].decode("utf-8"))
            xs_out[i-1,j] = float(var_res[-j-1][-1].decode("utf-8"))
        for j in range(len(tasks_res)):
            #assert tasks_res[j][0].decode("utf-8")[:-1] == task_names[-j-1], f"Task names outputted in different order than reverse of config\n\n{tasks_res[j][0].decode('utf-8')}\n{task_names[-j-1]}"
            objs_out[i-1,j] = float(tasks_res[-j-1][-1][1:-1].decode("utf-8"))


    print(xs_out, "\n", objs_out)
    print(f"\nTime for:\n 1st {-start_t + out1_t:.2f}\n 2nd {-out1_t + out2_t:.2f} \n 3nd {-out2_t + time():.2f}")
    all_objs = True
    return xs_out, objs_out, all_objs

def PESC_run_experiment(name = "test" ):
    folder_rel = "comparisson_algs/PESC_folders/" + name
    out_pros_result = subprocess.run(["conda", "run", "-n", "PESC", "python", 
                               "Spearmint/spearmint/cleanup.py", folder_rel],shell = True, capture_output=True)
    out_pros_result = subprocess.run(["conda", "run", "-n", "PESC", "python", 
                               "Spearmint/spearmint/main.py", folder_rel],shell = True, capture_output=True)
    print(out_pros_result)

def PESC_create_problem(problem_in, name):
    max_finished_jobs = 40 #default budjet
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
    c_lines = "\n    ".join([f"c{i} = ???" for i in range(con_num)])
    r_s = "\n    \n    return " + "{'f':f[0], " + ",".join([f"'c{i}' : np.array([float(c{i}[0][0] <= 0)]))" for i in range(con_num)]) + "}"
    file = imp_s + "def main(job_id, params):\n    " + array_s + f_line +"\n    " + c_lines + r_s
    
    with open(p_folder + "/problem.py", 'w') as f:
        f.write(file)
    
        
# DEBUGGING
if __name__ == "__main__":
    from opt_problems.ADMMBO_paper_problems import gardner1
    PESC_create_problem(gardner1, "test")