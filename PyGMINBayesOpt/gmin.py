import math
import os
from typing import TextIO
import numpy as np
import random
import subprocess
from smt.sampling_methods import LHS
from .utils import util
import random

def main():
    params = {
    "n_gmin": 50,
    "temp_min": 0.1,
    "temp_max": 10,
    "step_min": 0.01,
    "step_max": 1,
    "steps_min": 100,
    "steps_max": 1000,
    "gmin_exec": "/home/sma86/softwarewales/GMIN/builds/compiler/GMIN", # String. Path to GMIN executable.
    }
    gmin(params)
    count_lgbfs(n_gmin)

def gmin(params: dict):
    gen_gmin_data(params["n_gmin"], params["temp_min"], params["temp_max"], params["step_min"], params["step_max"],
                  params["steps_min"], params["steps_max"])
    prep_gmin(params["n_gmin"])
    #run_gmin(params["n_gmin"], params["gmin_exec"])   ! Must run GMIN using a bash script.

def count_lgbfs(n_gmin):
    check_convergence(n_gmin)
    n_steps = np.zeros(n_gmin)
    for i in range(n_gmin):
        logfile_path = os.path.join("gmin", "gmin" + str(i+1), "logfile")
        with open(logfile_path) as file:
            data = file.readlines()
        for line in data:
            if "steps=" in line:
                steps_flag = False
                for word in line.split():
                    if steps_flag == True:
                        n_steps[i] += round(float(word))
                        break
                    if word == "steps=":
                        steps_flag = True
    print(n_steps)
    calls_per_step = 3 # At each LFBGS step, there is 3 potential calls.
    n_calls = n_steps * calls_per_step
    mean_calls = np.mean(n_calls)
    std_calls = np.std(n_calls)
    print("Mean calls: " + str(mean_calls))
    print("Standard deviation: " + str(std_calls))

def check_convergence(n_gmin):
    for i in range(n_gmin):
        lowest_path = os.path.join("gmin", "gmin" + str(i+1), "lowest")
        with open(lowest_path) as file:
            data = file.readlines()
        lowest_energy = data[1].split()[4]
        print(lowest_energy)

def gen_gmin_data(n_gmin: int, temp_min: float, temp_max: float, step_min: float,
                   step_max: float, steps_min: int, steps_max: int):
    for i in range(n_gmin):
        subprocess.Popen(["cp", "-r", "data_template/data", "data/data" + str(i+1)]).wait()
        data_path = os.path.join("data", "data" + str(i+1))
        update_keyword(data_path, "TEMPERATURE", 1, random.uniform(temp_min, temp_max))
        update_keyword(data_path, "STEP", 1, random.uniform(step_min, step_max))
        update_keyword(data_path, "STEPS", 1, random.randint(steps_min, steps_max))

def prep_gmin(n_gmin: int):
    for i in range(n_gmin):
        if os.path.exists(os.path.join("gmin", "gmin" + str(i+1))) == False:
            subprocess.Popen(["mkdir", "gmin/gmin" + str(i+1)]).wait()
        subprocess.Popen(["cp", "-r", "data/data" + str(i+1), "gmin/gmin" + str(i+1) + r"/data"]).wait()
        subprocess.Popen(["cp", "-r", "coords/coords" + str(i+1), "gmin/gmin" + str(i+1) + r"/coords"]).wait()

def run_gmin(n_gmin: int, gmin_exec: str):
    for i in range(n_gmin):
        gmin_directory = os.path.join("gmin/gmin" + str(i+1))
        result = subprocess.run([gmin_exec], cwd = gmin_directory, stdout=subprocess.PIPE)

def update_keyword(file_path: TextIO, keyword: str, position: int, value: float) -> list:
    '''Read (o)data file and replace entry in specified position for specified keyword with "value".
    Position counts from 1.'''
    with open(file_path) as file:
        data = file.readlines()
    line_index = 0
    for line in data:
        if keyword in line:
            changed_line = line.split()
            changed_line[position] = str(value)
            changed_line = ' '.join(changed_line) + "\n"
            data[line_index] = changed_line
        line_index += 1
    with open(file_path, 'w') as file:
        file.writelines(data)


if __name__ == "__main__":
    main()