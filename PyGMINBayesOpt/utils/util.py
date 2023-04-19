import math
import os
import numpy as np
import subprocess

def calc_cell_vol(cell_params: list) -> float:
    "Calculates cell volume from cell parameters"
    cos_alpha = math.cos(cell_params[0])
    cos_beta = math.cos(cell_params[1])
    cos_gamma = math.cos(cell_params[2])
    a = cell_params[3]
    b = cell_params[4]
    c = cell_params[5]
    p = (1 - cos_alpha ** 2 - cos_beta ** 2 - cos_gamma ** 2 
        + 2 * cos_alpha * cos_beta * cos_gamma) ** 0.5
    vol = a * b * c * p
    return vol

def calc_cell_vectors(cell_params: list) -> np.ndarray:
    "Calculates Cartesian components of lattice vectors from cell parameters"
    cos_alpha = math.cos(cell_params[0])
    cos_beta = math.cos(cell_params[1])
    cos_gamma = math.cos(cell_params[2])
    sin_gamma = math.sin(cell_params[2])
    a = cell_params[3]
    b = cell_params[4]
    c = cell_params[5]
    a_vect = (a, 0, 0)
    b_vect = (b * cos_gamma, b * sin_gamma, 0)
    c_vect = (c * cos_beta, 
    c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
    c * (1 - cos_beta ** 2 - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma) ** 2) ** 0.5)
    cell_vectors = np.array([a_vect, b_vect, c_vect])
    low_values_flags = cell_vectors < 10 ** (-12)
    cell_vectors[low_values_flags] = 0 # Truncate low value elements of array, which may arise from numerical approximation of pi
    return cell_vectors

def calc_pair_list(data_point: np.array, origin: float, n_atoms: int, cluster_flag: bool) -> np.array:
    '''Takes a line of training data and returns a list of pair distances between all atoms.
    For non-clusters, the last six values should correspond to cell params: alpha, beta, gamma, a, b, c.'''
    if (cluster_flag):
        zeros = np.array([origin, origin, origin])
        cart_coords = np.concatenate((zeros, data_point))
        cart_coords = np.reshape(cart_coords, (n_atoms, 3))
    else:
        cell_vects = calc_cell_vectors(data_point[-6:])
        fract_coords = data_point[:-6]
        zeros = np.array([origin, origin, origin])
        fract_coords = np.concatenate((zeros, fract_coords))
        fract_coords = np.reshape(fract_coords, (n_atoms, 3))
        shifted_fract_coords = fract_coords
        for vect in np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]):
            shifted_fract_coords = np.concatenate((shifted_fract_coords, fract_coords + vect))
        cart_coords = np.matmul(np.transpose(cell_vects), np.transpose(shifted_fract_coords))
        cart_coords = np.transpose(cart_coords)
    pair_list = []
    total_atoms = np.shape(cart_coords)[0]
    for i in range(total_atoms):
        for j in range(i+1, total_atoms):
            displacement = cart_coords[j] - cart_coords[i]
            distance = np.linalg.norm(displacement)
            pair_list.append(distance)
    return pair_list

def coords_to_cell_vects(path: str, ortho_flag: bool) -> tuple:
    '''Reads in a coords file (located at path). 
    Returns fract_coords (Nx3 array) and cell_vects (3x3 array; each row is a cell vector).'''
    coords = np.loadtxt(path)
    if (ortho_flag):
        fract_coords = coords[0:-1]
        cell_params = np.zeros((2,3))
        cell_params[0] = np.array([0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi]) # Cell angles
        cell_params[1] = coords[-1:] # Cell lengths
    else:
        fract_coords = coords[0:-2]
        cell_params = coords[-2:]
    cell_vects = calc_cell_vectors(cell_params.flatten())
    return fract_coords, cell_vects

def singlepoints(n_minima: int, E_max: float, ortho_flag: bool, dftbp_flag: bool, gmin_exec: str, dftbpgmin_exec: str):
    '''Performs n_minima singlepoint calculations, reading in data from coords directory.
    If a coords file is absent for a particular minima (e.g. if minima corresponds to non-reasonable cell volume),
    in place of a singlepoint calculation, the energy E_max is written to a dump.1.V file.'''
    for num in range(n_minima):
        subprocess.Popen(["cp", "-r", "singlepoint_template", "singlepoint/point" + str(num+1)]).wait()
        point_path = os.path.join("singlepoint", "point" + str(num+1))
        coords_path = os.path.join("coords", "coords" + str(num+1))
        if os.path.exists(coords_path) == False: # True if coords file absent
            dump = np.array([[E_max, E_max]])
            dump_path = os.path.join("singlepoint", "point" + str(num+1), "dump.1.V")
            np.savetxt(dump_path, dump)
            continue
        if (dftbp_flag):
            fract_coords, cell_vects = coords_to_cell_vects(coords_path, ortho_flag)
            dftbp_path = os.path.join("singlepoint", "point" + str(num+1), "dftb_in.hsd")
            edit_dftbp_in(dftbp_path, fract_coords, cell_vects)
            run_dftbp(point_path, dftbpgmin_exec)
        else:
            subprocess.Popen(["cp", "-r", "coords/coords" + str(num+1), "singlepoint/point" + str(num+1) + r"/coords"]).wait()
            subprocess.Popen([gmin_exec], cwd = point_path).wait()
    
def edit_dftbp_in(path: str, fract_coords: np.array, cell_vects: np.array):
    with open(path) as file:
        data = file.readlines()
    line_index = 0
    for line in data:
        if "genFormat" in line:
            genFormat_header_end = line_index + 2
            fract_coord_end = genFormat_header_end + np.shape(fract_coords)[0]
            fract_coord_indices = range(genFormat_header_end+1, fract_coord_end+1)
            cell_vects_indices = range(fract_coord_end+2, fract_coord_end+5)
        if line_index in fract_coord_indices:
            fract_coord_index = fract_coord_indices.index(line_index)
            changed_line = line.split()
            for i in range(3):
                changed_line[i+2] = str(fract_coords[fract_coord_index][i])
            changed_line = ' '.join(changed_line) + "\n"
            data[line_index] = changed_line
        elif line_index in cell_vects_indices:
            cell_vects_index = cell_vects_indices.index(line_index)
            changed_line = line.split()
            for i in range(3):
                changed_line[i] = str(cell_vects[cell_vects_index][i])
            changed_line = ' '.join(changed_line) + "\n"
            data[line_index] = changed_line
        line_index += 1
    with open(path, 'w') as file:
        file.writelines(data)

def run_dftbp(path: str, dftbpgmin_exec: str):
    '''Run DFTBPGMIN in specified directory until dump.1.V file isn't empty for a maximum of 20 runs.'''
    run_flag = False
    attempt_no = 0
    dump_path = os.path.join(path, "dump.1.V")
    while run_flag == False:
        subprocess.Popen([dftbpgmin_exec], cwd = path).wait()
        if os.stat(dump_path).st_size > 0 or attempt_no > 20: # True if dump file not empty or 20 attempts already taken.
            run_flag = True # Confirm that DFTBPGMIN calculation completed without backtrace error.
        attempt_no += 1

def scrape_response(n_points: int):
    '''Scrapes energies from a directory of singlepoint calculations. Returns 1D array of energies.'''
    response = np.zeros(n_points)
    for i in range(n_points):
        file_path = os.path.join("singlepoint", "point" + str(i+1), "dump.1.V")
        with open(file_path) as file:
            for line in file:
                energy = line.split()[1]
                response[i] = energy
                break # Only want energy of initial configuration
    return response