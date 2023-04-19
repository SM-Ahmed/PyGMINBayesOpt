import os
import numpy as np
import subprocess

def main():
    num_runs = 10
    analyse_runs(num_runs)

def analyse_runs(num_runs: int):
    report_best_run(num_runs)
    count_empty_lowestGP(num_runs)
    energy_vs_calls(num_runs)

def report_best_run(num_runs: int):
    response_path = "run" + str(num_runs) + r"/response"
    training_path = "run" + str(num_runs) + r"/training"
    index, energy = find_lowest(response_path)
    coords = find_coords(training_path, index)
    print(str(index+1))
    print(energy)
    print(coords)

def find_lowest(response: str) -> tuple:
    '''Takes the path to a response file and returns the row index and energy of the lowest energy present.'''
    lowest_energy = 99999
    lowest_index = None
    with open(response) as file:
        line_index = 0
        for line in file:
                energy = float(line.split()[0])
                if energy < lowest_energy:
                    lowest_energy = energy
                    lowest_index = line_index
                line_index += 1
    return (lowest_index, lowest_energy)

def find_coords(training: str, index: int) -> np.array:
    '''Takes the path to a training file and a row index, and returns the training coordinates of the specified row.'''
    with open(training) as file:
        line_index = 0
        for line in file:
                if line_index == index:
                    return line.split()
                line_index += 1

def count_empty_lowestGP(num_runs: int) -> int:
    '''Reads a specified number of runs and counts how many runs have empty lowestGP.1 files.'''
    count = 0
    for i in range(num_runs-1):
        lowest_GP_path = os.path.join("run" + str(i+1), "lowestGP.1")
        if os.stat(lowest_GP_path).st_size == 0:
            count += 1
    print("Out of " + str(num_runs-1) + " runs.")
    print(str(count) + " runs have empty lowestGP.1 files.")
    return count

def energy_vs_calls(num_runs: int):
    '''Reads a specified number of runs and scrapes lowest energy as a function of potential calls.
    Outputs number of calls and lowest potential call as separate files.'''
    energies = filter_response(num_runs)
    print(np.shape(energies))
    num_rows = np.shape(energies)[0]
    num_calls_vals = np.zeros((num_rows))
    lowest_call_vals = np.zeros((num_rows))
    lowest_call = 99999
    for row in range(num_rows):
        num_calls_vals[row] = row
        energy = energies[row]
        if energy < lowest_call:
            lowest_call = energy
        lowest_call_vals[row] = lowest_call
    if os.path.isdir(os.path.join("analysis", "energy_vs_calls")) == False:
        subprocess.Popen(["mkdir", "analysis/energy_vs_calls"]).wait() 
    np.savetxt(os.path.join("analysis", "energy_vs_calls", "num_calls"), num_calls_vals)
    np.savetxt(os.path.join("analysis", "energy_vs_calls", "lowest_call"), lowest_call_vals)
    return num_calls_vals, lowest_call_vals

def filter_response(num_runs: int) -> tuple:
    '''Reads final response file from a specified number of runs and removes all rows corresponding to
    "bad coords" scenario where potential energy wasn't actually evaluated (e.g. coords was in bounds or gave
    non-physical cell volume).
    Returns response with all bad rows removed'''
    response = np.loadtxt(os.path.join("run" + str(num_runs), "response"))
    print(np.shape(response))
    response_index = 0
    bad_coords_rows = []
    for run in range(num_runs-1):
        if run == 0:
            training_response = np.loadtxt(os.path.join("run" + str(run + 1), "response"))
            training_size = np.shape(training_response)[0]
            response_index += training_size
        for i in range(3):
            coords_path =  os.path.join("run" + str(run + 1), "coords", "coords" + str(i + 1))
            coords_bad_path =  os.path.join("run" + str(run + 1), "coords", "coords" + str(i + 1) + r"_bad")
            if (os.path.exists(coords_path)):
                response_index += 1
            elif (os.path.exists(coords_bad_path)):
                bad_coords_rows.append(response_index)
                response_index += 1
    good_response = np.delete(response, bad_coords_rows)
    training_rows = [i for i in range(training_size)]
    good_bayesopt_response = np.delete(good_response, training_rows)
    print("Total response size: " + str(np.shape(response)[0]))
    print("Total rows checked: " + str(response_index))
    print("Total training rows: " + str(len(training_rows)))
    print("Total bad rows: " + str(len(bad_coords_rows)))
    print("Final good rows: " + str(np.shape(good_bayesopt_response)[0]))
    return good_response


if __name__ == "__main__":
    main()