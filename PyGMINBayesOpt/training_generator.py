import math
import os
from typing import NoReturn
import numpy as np
import subprocess
from smt.sampling_methods import LHS
from .utils import util

def main():
    params = {
    "n_atoms": 2, # Int. No. atoms per unit cell.
    "n_configs": 20, # Int. No. cell configurations to be generated in central region of config space.
    "n_configs_bounds": 20, # Int. No. cell configurations to be generated in bounds region of config space.
    "a_max": 5, # Float. Maximum cell length.
    "a_min": 0, # Float. Minimum cell length.
    "a_bound_width": 0.5, # Float. Cell length width that is considered "bounds" region.
    "v_min": 0, # Float. Minimum cell volume.
    "angle_min": 0.1, # Float. Minimum cell angle in radians.
    "angle_max": 3, # Float. Maximum cell angle in radians.
    "angle_bound_width": 0.2, # Float. Cell angle width that is considered "bounds" region.
    "min_bond_length": 1, # Float. Minimum acceptable distance between any pair of atoms
    "origin": 0.1, # Float. Fixed fractional coordinate of first atom in all three dimensions.
    "ortho_flag": False, # Bool. If true, min_angle and max_angle parameters are ignored and all cell angles are set to 90 deg.
    "cluster_flag": False, # Bool. If true, program does not generate cell length and cell angle features.
    "E_max": 3, # Float. Values in "response" file above E_max are truncated to E_max
    "autoEmax_flag": True, # Bool. If true, E_max parameter is ignored. Instead, program takes E_max as the modulus of the lowest negative value in "response".
    "cleanup_flag": True, # Bool. If true, program begins by deleting contents of "coords", "singlepoint" and "BayesOpt" folders.
    "dftbp_flag": True, # Bool. If true, program prepares dftbp_in.hsd file instead of coords.
    "gmin_exec": "/home/sma86/softwarewales/GMIN/builds/compiler/GMIN", # String. Path to GMIN executable.
    "dftbpgmin_exec": "/home/sma86/softwarewales/GMIN/builds/ifort_dftbp/DFTBPGMIN" # String. Path to DFTBPGMIN executable.
    }
    training_generator(params)

def training_generator(params: dict):
    if (params["cleanup_flag"]):
        cleanup()
    n_training = gen_training(params)
    E_max = gen_response(params, n_training)
    gen_bounds(params, E_max)
    join_bounds()

def gen_training(params: dict):
    '''Generate a set of acceptable configurations in central region of config space via latin hypercube sampling.
    Outputs training file and coords directories.
    Returns the number of acceptable configurations.'''
    if (params["cluster_flag"]):
        training = generate_cluster(params["n_atoms"], params["n_configs"], params["a_min"], params["a_max"])
    else:
        training = generate_cell(params["n_atoms"], params["n_configs"],
                                 params["a_min"], params["a_max"], params["a_bound_width"],
                                params["angle_min"], params["angle_max"], params["angle_bound_width"],
                                params["ortho_flag"])
        training = remove_bad_rows(training, params["n_atoms"], params["v_min"], params["min_bond_length"],
                                    params["origin"], params["cluster_flag"], params["ortho_flag"])
    np.savetxt(os.path.join("BayesOpt", "training"), training)
    output_coords(training, params["n_atoms"], params["origin"], params["cluster_flag"], params["ortho_flag"])
    n_training = count_good_rows(training, params["n_configs"])
    return n_training

def gen_response(params: dict, n_training: int):
    '''Run singlepoint calculations for each training configuration and collects energies.
    Truncates energies above Emax. Outputs raw_response, Emax and response files.
    Return E_max.'''
    util.singlepoints(n_training, 9999, params["ortho_flag"], params["dftbp_flag"], params["gmin_exec"], params["dftbpgmin_exec"])
    raw_response = util.scrape_response(n_training)
    if (params["autoEmax_flag"]):
        E_max = calc_E_max(raw_response)
    response = np.clip(raw_response, None, E_max)
    np.savetxt(os.path.join("BayesOpt", "raw_response"), raw_response)
    with open(os.path.join("BayesOpt", "Emax"), "w") as file:
        file.write(str(E_max))
    np.savetxt(os.path.join("BayesOpt", "response"), response)
    remove_large_energies("BayesOpt/response", "BayesOpt/training", E_max)
    return E_max

def gen_bounds(params: dict, E_max: float):
    '''Generate configurations in bounds of configuration space via LHS.
    Outputs training_bounds (contains configs) and response_bounds (all entries = E_max) files.'''
    training = gen_cell_bounds(params["n_configs_bounds"], params["a_min"], params["a_max"], params["a_bound_width"],
                    params["angle_min"], params["angle_max"], params["angle_bound_width"])
    np.savetxt(os.path.join("BayesOpt", "training_bounds"), training)
    response = np.array([[E_max] for i in range(np.shape(training)[0])])
    np.savetxt(os.path.join("BayesOpt", "response_bounds"), response)

def join_bounds():
    '''Concatenates training/response data with bounds data, then
    outputs "training_final" and "response_final" files.'''
    training_central = np.loadtxt(os.path.join("BayesOpt", "training"))
    training_bounds = np.loadtxt(os.path.join("BayesOpt", "training_bounds"))
    response_central = np.loadtxt(os.path.join("BayesOpt", "response"))
    response_bounds = np.loadtxt(os.path.join("BayesOpt", "response_bounds"))
    training = np.concatenate((training_central, training_bounds), axis=0)
    response = np.concatenate((response_central, response_bounds), axis=0)
    np.savetxt(os.path.join("BayesOpt", "training_final"), training)
    np.savetxt(os.path.join("BayesOpt", "response_final"), response)

def cleanup():
    print("Cleanup")
    subprocess.Popen("rm -rf coords/*", shell=True).wait()
    subprocess.Popen("rm -rf singlepoint/*", shell=True).wait()
    subprocess.Popen("rm -rf BayesOpt/*", shell=True).wait()

def generate_cluster(n_atoms: int, n_configs: int, a_min: float, a_max: float) -> np.array:
    "Generate position parameters via LHS"
    coord_limits = [[a_min, a_max] for i in range(3 * (n_atoms - 1))]
    limits = np.array(coord_limits)
    sampling = LHS(xlimits = limits)
    training_data = sampling(n_configs)
    return training_data

def generate_cell(n_atoms: int, n_configs: int, a_min: float, a_max: float, a_bound_width: float,
                   angle_min: float, angle_max: float, angle_bound_width: float, ortho_flag: bool) -> np.array:
    "Generate cell parameters via LHS"
    fract_coord_limits = [[0, 1] for i in range(3 * (n_atoms - 1))]
    if (ortho_flag):
        angle_limits = [[math.pi / 2, math.pi / 2] for i in range(3)]
    else:
        angle_limits = [[angle_min + angle_bound_width, angle_max - angle_bound_width] for i in range(3)]
    length_limits = [[a_min + a_bound_width, a_max - a_bound_width] for i in range(3)]
    limits = np.array(fract_coord_limits + angle_limits + length_limits)
    sampling = LHS(xlimits = limits)
    training_data = sampling(n_configs)
    return training_data

def remove_bad_rows(data: np.array, n_atoms: int, v_min: float, min_bond_length: float,
                     origin: float, cluster_flag: bool, ortho_flag: bool) -> np.array:
    "Remove rows corresponding to data points with unphysical volumes."
    row_number = 0
    bad_rows = []
    for data_point in data:
        vol = util.calc_cell_vol(data_point[-6:])
        if not ((np.isreal(vol) == True) and (vol > v_min)): # Vol of a good row is real and greater than v_min.
            bad_rows.append(row_number)
            row_number += 1
            continue
        pair_list = util.calc_pair_list(data_point, origin, n_atoms, cluster_flag)
        for distance in pair_list:
            if distance <= min_bond_length:
                bad_rows.append(row_number)
                break
        row_number += 1
    data = np.delete(data, bad_rows, axis = 0)
    if ortho_flag == True:
        data = np.delete(data, (-4, -5, -6), axis = 1) # Remove cell angles
    return data

def count_good_rows(data: np.array, n_configs: int) -> NoReturn:
    "Print number of data points that the script generated, removed and outputted."
    n_training = np.shape(data)[0]
    n_bad_rows = n_configs - n_training
    print("Out of " + str(n_configs) + " data points generated,\n" 
        + str(n_bad_rows) + " data points were removed \n" 
        + "to make a training set of " + str(n_training) + " points.")
    return n_training

def output_coords(data: np.array, n_atoms: int, origin: float, cluster_flag: bool, ortho_flag: bool) -> NoReturn:
    '''Outputs coords file into coords directory for each row in "data" array.'''
    n_rows = n_atoms
    if cluster_flag == False:
        n_rows += 2 # Include cell lengths and angles.
    if ortho_flag == True:
        n_rows -= 1 # Remove cell angles
    file_pos = 1
    for data_point in data:
        zeros = np.array([origin, origin, origin])
        coords = np.concatenate((zeros, data_point))
        coords = np.reshape(coords, (n_rows, 3))
        file_path = os.path.join("coords", "coords" + str(file_pos))
        np.savetxt(file_path, coords)
        file_pos += 1

def calc_E_max(response: np.array):
    '''Reads a response file and returns the magnitude of the most negative energy present.
    If all energies are positive, returns None.'''
    E_min = np.amin(response)
    if E_min >= 0:
        E_max = None # Error case
    else:
        E_max = -1 * E_min
    return E_max

def remove_large_energies(response: str, training: str, E_max: float) -> tuple:
    '''Takes paths to response and training files, reads in data, removes all rows with energies above E_max,
    and outputs modified response and training files. Returns new response and training arrays.'''
    energies = np.loadtxt(response)
    coords = np.loadtxt(training)
    large_flags = (energies == E_max)
    large_rows = []
    num_rows = np.size(large_flags)
    for row in range(num_rows):
        if (large_flags[row]):
            large_rows.append(row)
    condensed_energies = np.delete(energies, large_rows)
    condensed_coords = np.delete(coords, large_rows, 0)
    np.savetxt(response + "2", condensed_energies)
    np.savetxt(training + "2", condensed_coords)
    return training, response

def gen_cell_bounds(n_configs: int, a_min: float, a_max: float, a_bound_width: float,
                            angle_min: float, angle_max: float, angle_bound_width: float):
    '''Generate a set of acceptable configurations via latin hypercube sampling at bounds of config space.
    Currently only works for non-orthorhombic cells with one atom per unit cell.'''
    n_configs_per_bound = int(n_configs / 12) # 6 features total (3 cell lengths + 3 cell angles). 2 bounds per feature.
    angle_limits = [[angle_min, angle_max] for i in range(3)]
    length_limits = [[a_min, a_max] for i in range(3)]
    bound_training = np.zeros((n_configs_per_bound * 12, 6))
    n_rows_filled = 0
    for i in range(3): # Cell angles lower bounds.
        angle_lim_bounds = angle_limits.copy()
        angle_lim_bounds[i] = [angle_min, angle_min + angle_bound_width]
        limits = np.array(angle_lim_bounds + length_limits)
        sampling = LHS(xlimits = limits)
        bound_data = sampling(n_configs_per_bound)
        bound_training[n_rows_filled: n_rows_filled+n_configs_per_bound] = bound_data
        n_rows_filled += n_configs_per_bound
    for i in range(3): # Cell angles upper bounds.
        angle_lim_bounds = angle_limits.copy()
        angle_lim_bounds[i] = [angle_max-angle_bound_width, angle_max]
        limits = np.array(angle_lim_bounds + length_limits)
        sampling = LHS(xlimits = limits)
        bound_data = sampling(n_configs_per_bound)
        bound_training[n_rows_filled: n_rows_filled+n_configs_per_bound] = bound_data
        n_rows_filled += n_configs_per_bound
    for i in range(3): # Cell length lower bounds.
        length_lim_bounds = length_limits.copy()
        length_lim_bounds[i] = [a_min, a_min + a_bound_width]
        limits = np.array(angle_limits + length_lim_bounds)
        sampling = LHS(xlimits = limits)
        bound_data = sampling(n_configs_per_bound)
        bound_training[n_rows_filled: n_rows_filled+n_configs_per_bound] = bound_data
        n_rows_filled += n_configs_per_bound
    for i in range(3): # Cell length upper bounds.
        length_lim_bounds = length_limits.copy()
        length_lim_bounds[i] = [a_max - a_bound_width, a_max]
        limits = np.array(angle_limits + length_lim_bounds)
        sampling = LHS(xlimits = limits)
        bound_data = sampling(n_configs_per_bound)
        bound_training[n_rows_filled: n_rows_filled+n_configs_per_bound] = bound_data
        n_rows_filled += n_configs_per_bound
    return bound_training

if __name__ == "__main__":
    main()