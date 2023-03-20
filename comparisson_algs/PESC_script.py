import subprocess
from time import time



def PESC_main():
    start_t = time()
    ## For testing
    # conda run -n PESC python --version
    out_pros = subprocess.run(["conda", "run", "-n", "PESC", "python", "--version"],shell = True, capture_output=True)
    print(out_pros);out1_t = time()
    out_pros_result = subprocess.run(["conda", "run", "-n", "PESC", "python", 
                               "Spearmint/spearmint/print_all_results.py", "Spearmint/examples/toy"],shell = True, capture_output=True)
    results = out_pros_result.stdout.split(sep=b"Job ") #Splitting byte output by windows newline #NB! different separator for other OS
    out2_t = time()

    for job in results:
        print(job.split(sep=b"\r\r\n"))

    print(f"\nTime for:\n 1st {-start_t + out1_t:.2f}\n 2nd {-out1_t + out2_t:.2f}")