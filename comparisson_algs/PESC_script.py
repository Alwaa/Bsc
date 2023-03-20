import subprocess
import json
from time import time



def PESC_main(problem_folder_rel = "Spearmint/examples/toy"):
    start_t = time()
    ## For testing
    # conda run -n PESC python --version
    out_pros = subprocess.run(["conda", "run", "-n", "PESC", "python", "--version"],shell = True, capture_output=True)
    print(out_pros);out1_t = time()

    ## Reading from config file to parse ##
    f = open(problem_folder_rel + "/config.json")
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
                               "Spearmint/spearmint/print_all_results.py", problem_folder_rel],shell = True, capture_output=True)
    results = out_pros_result.stdout.split(sep=b"Job ") #Splitting byte output by windows newline #NB! different separator for other OS
    out2_t = time()

    for i, job in enumerate(results):
        fields = job.split(sep=b"\r\r\n")
        if len(fields) == 1:
            continue
        job_num = int(fields[0])
        assert job_num == i, f"Result from PESC not parsed correctly!\n Incorrect itetarion number --{job_num}-- is not --{i}--" #Me being paranoid after the trublesome setup
        print(job_num)
        fields = fields[3:] #Making indexing easier by removing headers
        var_res = [s.split() for s in fields[:num_vars]]

        tasks_res = [s.split() for s in fields[num_vars:num_vars+num_tasks]]
        print(var_names)
        for j in range(len(var_res)):
            assert var_res[j][0].decode("utf-8") == var_names[-j-1], f"Variable names outputted in different order than reverse of config\n\n{var_res[j][0].decode('utf-8')}\n{var_names[-j-1]}"
            print(var_res[j][0], var_names[-j-1])
        for j in range(len(tasks_res)):
            assert str(tasks_res[j][0]) == task_names[-j], f"Task names outputted in different order than reverse of config\n\n{tasks_res}"



    print(f"\nTime for:\n 1st {-start_t + out1_t:.2f}\n 2nd {-out1_t + out2_t:.2f}")