import math
import os
import numpy as np
import subprocess
from typing import TextIO
from .utils import util, batch


def main():
    params = {
    "sampling_mode": 0, # Int. Method of sampling points via AF: single global minima (0), topological batch selection (1), or lowest minima (2)
    "hyperparams_flag": True, # Bool. True if BayesOpt GMIN hyperparams file needed
    "gpplateau_flag": True, # Bool. True if BayesOpt GMIN gpplateau file needed
    "E_max": 3, # Float. Values in "response" file above E_max are truncated to E_max
    "a_max": 5, # Float. Maximum cell length.
    "a_min": 0, # Float. Minimum cell length.
    "a_bound_width": 0.5, # Float. Cell length width that is considered "bounds" region.
    "v_min": 0, # Float. Minimum cell volume.
    "angle_min": 0.1, # Float. Minimum cell angle in radians.
    "angle_max": 3, # Float. Maximum cell angle in radians.
    "angle_bound_width": 0.2, # Float. Cell angle width that is considered "bounds" region.
    "cluster_flag": False, # Bool. If true, cell information is ignored.
    "ortho_flag": False, # Bool. If true, min_angle and max_angle parameters are ignored and all cell angles are set to 90 deg.
    "dftbp_flag": True, # Bool. If true, program prepares dftbp_in.hsd file instead of coords.
    "gmin_exec": "/home/sma86/softwarewales/GMIN/builds/compiler/GMIN", # String. Path to GMIN executable.
    "dftbpgmin_exec": "/home/sma86/softwarewales/GMIN/builds/ifort_dftbp/DFTBPGMIN", # String. Path to DFTBPGMIN executable.
    "bayesopt_optim_exec": "/home/sma86/walesibm/OPTIM/builds/ifort/OPTIM", # String. Path to BayesOpt OPTIM executable.
    "bayesopt_pathsample_exec": "/home/sma86/walesibm/PATHSAMPLE/builds/ifort/PATHSAMPLE", # String. Path to BayesOpt PATHSAMPLE executable.
    "alpha": 0.8, # Float. Sets energy cutoff threshold during topological batch selection.
    "beta": 0.2, # Float. Sets barrier cutoff threshold during topological batch selection.
    "batch_size": 3 # Int. Number of minima to be sampled during topological batch selection.
    }
    bayesopt(params)

def bayesopt(params: dict):
    num_points = choose_test_points(params)
    util.singlepoints(num_points, params["E_max"], params["ortho_flag"], params["dftbp_flag"],
                       params["gmin_exec"], params["dftbpgmin_exec"]) # Runs singlepoint calculations.
    training = gen_next_training(num_points)
    response = gen_next_response(num_points, params["E_max"])
    prep_next_run(training, response, num_points, params["sampling_mode"], params["hyperparams_flag"], params["gpplateau_flag"])

def choose_test_points(params: dict):
    '''Read in lowestEI file to get list of AF minima, then choose test points.
    Coords of chosen test points are placed in coords directory.
    Returns number of test points.'''
    coords = read_lowestEI() 
    central_coords = remove_bounds_min(coords, params["a_min"], params["a_max"], params["a_bound_width"],
                           params["angle_min"], params["angle_max"], params["angle_bound_width"], params["ortho_flag"])
    if len(central_coords) == 0: # If all minima are at the bounds, output first minima in lowestEI list as "bad" coords.
        output_coords(coords[0], 1, True)
        return 1
    if params["sampling_mode"] == 0:
        prep_one_singlepoint(central_coords, params["v_min"], params["cluster_flag"], params["ortho_flag"])
        return 1
    read_lowestGP()
    num_minima = import_minima(central_coords, params)
    if (num_minima == 0): # All OPTIM minimisations failed, so sample point from lowestEI.xyz.
        prep_one_singlepoint(central_coords, params["v_min"], params["cluster_flag"], params["ortho_flag"])
        return 1
    if (num_minima == 1): # Single AF minima exists, so select single point.
        extract_batch_single(params["v_min"], params["cluster_flag"], 
                                params["ortho_flag"], params["bayesopt_pathsample_exec"]) # Extract coords from single minimum.
        return 1
    if params["sampling_mode"] == 1:
        prep_topo_batch(central_coords, params["bayesopt_optim_exec"], params["bayesopt_pathsample_exec"])
        run_topo_batch(params["alpha"], params["beta"], params["batch_size"]) # Select optimal batch of minima via topological batch selection algorithm.
        num_points = extract_topo_batch(params["batch_size"], params["v_min"], params["cluster_flag"],
                params["ortho_flag"], params["bayesopt_pathsample_exec"]) # Extract coords from the selected batch of minima.
        return num_points
    if params["sampling_mode"] == 2:
        num_points = extract_lowest_batch(params["batch_size"], params["v_min"], params["cluster_flag"],
                params["ortho_flag"], params["bayesopt_pathsample_exec"]) # Extract coords from the selected batch of minima.
        return num_points

def read_lowestEI() -> list:
    '''Read in lowestEI file. Decompose into separate AF minima. Discards minima at bounds.
    Returns list of coords for each minima in lowestEI.1'''
    line_index = -1 
    data = []
    coords = []
    with open(os.path.join("lowestEI.1")) as file:
        for line in file:
            if line_index == -1: # Read no. of features (1st line in xyz file for one minima)
                no_features = int(line.split()[0])
            if line_index >= 1 and line_index <= no_features: # Read minima data
                value = float(line.split()[0])
                data.append(value)
            line_index += 1
            if line_index == no_features + 1: # Reset for next minima.
                coords.append(data)
                data = []
                line_index = -1
    return coords

def remove_bounds_min(coords: list, a_min: float, a_max: float, a_bound_width: float,
                        angle_min: float, angle_max: float, angle_bound_width: float, ortho_flag: bool) -> list:
    '''Reads a list of coordinates for AF minima. Returns a list of coordinates 
    having removed minima that are at the bounds.
    Currently, only works for cells with one atom.'''
    central_coords = []
    for minima in coords:
        if (check_central_minima(minima, a_min, a_max, a_bound_width, angle_min, angle_max, angle_bound_width, ortho_flag)):
            central_coords.append(minima)
    return central_coords

def check_central_minima(data: np.array, a_min: float, a_max: float, a_bound_width: float,
                        angle_min: float, angle_max: float, angle_bound_width: float, ortho_flag: bool) -> bool:
    '''Takes in coordinates for a AF minima. Checks if it is not at the bounds of config space.
    Returns true if in the central region, false if at the bounds.
    Currently, only works for cells with one atom.'''
    if (ortho_flag):
        for i in range(3): # Check lengths
            if data[i] < (a_min + a_bound_width) or data[i] > (a_max - a_bound_width):
                return False
    else: 
        for i in range(3): # Check angles
            if data[i] < (angle_min + angle_bound_width) or data[i] > (angle_max - angle_bound_width):
                return False
        for i in range(3, 6): # Check lengths
            if data[i] < (a_min + a_bound_width) or data[i] > (a_max - a_bound_width):
                return False
    return True

def prep_one_singlepoint(coords: list, v_min: float, cluster_flag: bool, ortho_flag: bool):
    '''Take a list of lowestEI minima coords, choose to sample the first minima,
    then prepare directories for singlepoint calculation.'''
    for point in coords:
        if check_good_cell(point[-6:], v_min, cluster_flag, ortho_flag):
            output_coords(point, 1)
        else:
            output_coords(point, 1, True)
        break

def import_minima(coords: list, params: dict) -> bool:
    '''Import AF minima via OPTIM/PATHSAMPLE to make min.data file.
    Returns 0, 1, or 2 if min.data contains 0, 1, or more than 1 minima respectively.'''
    minima_index = 0
    for minima in coords:
        minima_index += 1
        prep_optim_min(minima, minima_index)
    run_optim_min(minima_index, params["bayesopt_optim_exec"]) # Perform minimisation for each minima in lowestEI file via OPTIM.
    check_optim_min(minima_index, params["a_min"], params["a_max"], params["a_bound_width"],
                           params["angle_min"], params["angle_max"], params["angle_bound_width"], params["ortho_flag"]) 
    import_pathsample_min(minima_index, params["bayesopt_pathsample_exec"]) # Import lowestEI minima into PATHSAMPLE.
    min_data_path = os.path.join("Topo_Batch", "import_min", "min.data")
    if os.stat(min_data_path).st_size == 0:
        return 0
    min_data = np.loadtxt(min_data_path) # Read min.data file.
    return min_data.ndim

def check_good_cell(cell_params: np.array, v_min: float, cluster_flag: bool, ortho_flag: bool) -> bool:
    '''Takes in a list of cell parameters and checks if the unit cell volume is real and greater than v_min.
    Returns true for an acceptable cell volume. Always returns true for clusters and orthorhombic cells.'''
    if cluster_flag == False and ortho_flag == False: # True for non-orthorhombic cell
        vol = util.calc_cell_vol(cell_params)
        return (np.isreal(vol) == True) and (vol > v_min) # True for physical unit cell volume.
    else:
        return True
    
def read_lowestGP():
    '''Read in lowestGP file and output first minima to gpfit.txt file in Topo_Batch_template directory.'''
    lowestGP_path = os.path.join("lowestGP.1")
    if os.stat(lowestGP_path).st_size == 0:
        lowestGP_path = os.path.join("bestGP.1.xyz")
        if os.stat(lowestGP_path).st_size == 0:
            print("ERROR. Empty lowestGP.1 and best GP.1.xyz files.")
            quit()
    with open(lowestGP_path) as file:
        line_no = 0
        GP_data = []
        for line in file:
            if line_no == 0:
                hyperparams = int(line.strip()) # Read in number of hyperparam features
            if line_no >= 2 and line_no <= (hyperparams + 1):
                GP_data.append(line) # Record data
            if line_no > (hyperparams + 1):
                break
            line_no += 1
    with open(os.path.join("Topo_Batch_template", "gpfit.txt"), "w") as file:
        file.writelines(GP_data)
    
def prep_optim_min(data: np.ndarray, index: int):
    '''Prepare OPTIM minimisation directory for particular AF minima.'''
    prep_optim_directory("min", "min" + str(index))
    odata_path = os.path.join("Topo_Batch", "min" + str(index), "odata")
    with open(odata_path, "a") as file:
        np.savetxt(file, data, fmt='%f')

def run_optim_min(num_minima: int, bayesopt_optim_exec: str):
    '''Run OPTIM minimisations for each AF minima.'''
    for minima in range(num_minima):
        directory_name = "Topo_Batch/min" + str(minima+1)
        subprocess.Popen([bayesopt_optim_exec], cwd = directory_name).wait()

def check_optim_min(num_minima: int, a_min: float, a_max: float, a_bound_width: float,
                        angle_min: float, angle_max: float, angle_bound_width: float, ortho_flag: bool):
    '''Checks if OPTIM minimisation converged to bounds. If so, renames directory so it will not be imported into PATHSAMPLE.'''
    '''Currently only works for single atom cells.'''
    for minima in range(num_minima):
        min_path = os.path.join("Topo_Batch", "min" + str(minima+1), "min.data.info")
        with open(min_path) as file: # Read min.data.info file
            min_data = file.readlines()
        cell = np.zeros(6)
        cell_info = min_data[-2:] # Last two lines of file correspond to cell lengths and cell angles.
        counter = 0
        for line in cell_info:
            for value in line.split():
                cell[counter] = float(value)
                counter += 1
        print(cell)
        if check_central_minima(cell, a_min, a_max, a_bound_width, angle_min, angle_max, angle_bound_width, ortho_flag):
            continue
        directory_old = os.path.join("Topo_Batch", "min" + str(minima+1))
        directory_new = os.path.join("Topo_Batch", "min" + str(minima+1) + "_bounds")
        os.rename(directory_old, directory_new)

def import_pathsample_min(num_minima: int, bayesopt_pathsample_exec: str):
    '''Parse OPTIM minimisation files to create min.data.info.initial file.
    Run PATHSAMPLE to import minima.'''
    prep_optim_directory("import_min", "import_min")
    min_data_path = os.path.join("Topo_Batch", "import_min", "min.data.info.initial")
    with open('min_data_path', 'w') as file: # Create empty min.data.info.initial file.
        pass
    for minima in range(num_minima):
        min_path = os.path.join("Topo_Batch", "min" + str(minima+1), "min.data.info")
        if os.path.exists(min_path) == False:
            continue
        with open(min_path) as file: # Read min.data.info file
            min_data = file.readlines()
        with open(min_data_path, "a") as file: # Add data to min.data.info.initial file.
            file.writelines(min_data)
    import_min_path = os.path.join("Topo_Batch", "import_min")
    subprocess.Popen([bayesopt_pathsample_exec], cwd = import_min_path).wait() # Run PATHSAMPLE.

def prep_topo_batch(coords: list, bayesopt_optim_exec: str, bayesopt_pathsample_exec: str):
    '''Connect AF minima via transition states using OPTIM/PATHSAMPLE.'''
    run_optim_path(bayesopt_optim_exec, bayesopt_pathsample_exec) # Connect two lowestEI minima via PATHSAMPLE.
    import_pathsample_path(bayesopt_pathsample_exec) # Import lowest EI minima, including a path between two minima, into PATHSAMPLE.
    connect_pathsample_path(bayesopt_pathsample_exec) # Connect all lowestEI minima via PATHSAMPLE.

def run_optim_path(bayesopt_optim_exec: str, bayesopt_pathsample_exec: str):
    '''Extract coordinates of first two minima in import_min output via PATHSAMPLE.
    Then connect the two minima via OPTIM.'''
    prep_optim_directory("path", "path")
    min1_data = extract_min("import_min", "extract_min1", 1, bayesopt_pathsample_exec)
    min2_data = extract_min("import_min", "extract_min2", 2, bayesopt_pathsample_exec)
    odata_path = os.path.join("Topo_Batch", "path", "odata")
    with open(odata_path, "a") as file:
        np.savetxt(file, min1_data, fmt='%f')
    finish_path = os.path.join("Topo_Batch", "path", "finish")
    with open(finish_path, "w") as file:
        np.savetxt(file, min2_data, fmt='%f')
    subprocess.Popen([bayesopt_optim_exec], cwd = "Topo_Batch/path").wait()

def import_pathsample_path(bayesopt_pathsample_exec: str):
    '''Import path into PATHSAMPLE.'''
    prep_optim_directory("import_path", "import_path")
    subprocess.Popen(["cp", "-r", "Topo_Batch/import_min/min.data", "Topo_Batch/import_path/min.data"]).wait() # Copy min.data file
    subprocess.Popen(["cp", "-r", "Topo_Batch/import_min/ts.data", "Topo_Batch/import_path/ts.data"]).wait() # Copy ts.data file
    subprocess.Popen(["cp", "-r", "Topo_Batch/import_min/points.min", "Topo_Batch/import_path/points.min"]).wait() # Copy points.min file
    subprocess.Popen(["cp", "-r", "Topo_Batch/import_min/points.ts", "Topo_Batch/import_path/points.ts"]).wait() # Copy points.ts file
    subprocess.Popen(["cp", "-r", "Topo_Batch/path/path.info", "Topo_Batch/import_path/path.info.initial"]).wait() # Copy path.info file
    import_path_path = os.path.join("Topo_Batch", "import_path") 
    subprocess.Popen([bayesopt_pathsample_exec], cwd = import_path_path).wait() # Run PATHSAMPLE.

def prep_optim_directory(template: str, target: str):
    '''Copies "template" directory from Topo_Batch_template directory to "target" directory in Topo_Batch directory.
    Then, copies response, training, funcbounds and gpfit.txt files to "target".'''
    target_directory = "Topo_Batch/" + target
    subprocess.Popen(["cp", "-r", "Topo_Batch_template/" + template, target_directory]).wait()
    subprocess.Popen(["cp", "-r", "training", target_directory + r"/training"]).wait() # Copy training file
    subprocess.Popen(["cp", "-r", "response", target_directory + r"/response"]).wait() # Copy response file
    subprocess.Popen(["cp", "-r", "funcbounds", target_directory + r"/funcbounds"]).wait() # Copy funcbounds file
    subprocess.Popen(["cp", "-r", "Topo_Batch_template/gpfit.txt", target_directory + r"/gpfit.txt"]).wait() # Copy gpfit.txt file

def connect_pathsample_path(bayesopt_pathsample_exec: str):
    '''Run PATHSAMPLE to connect minima via transition states.'''
    subprocess.Popen(["cp", "-r", "Topo_Batch/import_path", "Topo_Batch/connect_path"]).wait() # Copy import_path directory
    subprocess.Popen(["cp", "-r", "Topo_Batch_template/connect_path/pathdata",
                      "Topo_Batch/connect_path/pathdata"]).wait() # Overwrite pathdata file
    connect_path_path = os.path.join("Topo_Batch", "connect_path")
    subprocess.Popen([bayesopt_pathsample_exec], cwd = connect_path_path).wait() # Run PATHSAMPLE.

def run_topo_batch(alpha: float, beta: float, batch_size: int):
    '''Run topological batch python script to choose sampling minima from min/ts database.'''
    subprocess.Popen(["mkdir", "Topo_Batch/topo_batch"]).wait() 
    subprocess.Popen(["cp", "-r", "Topo_Batch/connect_path/min.data", "Topo_Batch/topo_batch/min.data"]).wait() # Copy min.data file
    subprocess.Popen(["cp", "-r", "Topo_Batch/connect_path/ts.data", "Topo_Batch/topo_batch/ts.data"]).wait() # Copy ts.data file
    batch.topo_batch("Topo_Batch/topo_batch", alpha, beta, batch_size) # Run batch.py script.

def extract_topo_batch(batch_size: int, v_min: float, cluster_flag: bool, ortho_flag: bool, bayesopt_pathsample_exec: str):
    '''Extract coordinates from topological batch output, then outputs corresponding coords file.
    Doesn't output coords for minima that have inappropriate cell volumes. A maximum of batch_size minima are extracted.
    Returns number of extracted minima'''
    batch = np.loadtxt(os.path.join("Topo_Batch", "topo_batch", "Monotonic"))
    minima = [] # List of indices of minima extracted so far.
    for i in range(np.size(batch)):
        if batch.ndim == 0: # Handle edge case for only 1 minima.
            min_no = batch + 1
        else:
            min_no = batch[i] + 1
        minima.append(min_no)
        coords = extract_min("connect_path", "extract_batch" + str(len(minima)), min_no, bayesopt_pathsample_exec)
        if check_good_cell(coords[-6:], v_min, cluster_flag, ortho_flag):
            output_coords(coords, len(minima))
        else:
            output_coords(coords, len(minima), True)
        if len(minima) == batch_size: # Enough minima extracted. Exit function.
            return len(minima)
    if os.path.exists(os.path.join("Topo_Batch", "topo_batch", "BarrierSelection")) == False:
        return len(minima) # Barrier selection file doesn't exist. Stop reading.
    batch = np.loadtxt(os.path.join("Topo_Batch", "topo_batch", "BarrierSelection"))
    for i in range(np.size(batch)):
        if batch.ndim == 0: # Handle edge case for only 1 minima.
            min_no = batch + 1
        else:
            min_no = batch[i] + 1
        if min_no in minima: # Ignore minima that have already been extracted.
            continue
        minima.append(min_no)
        print(minima)
        coords = extract_min("connect_path", "extract_batch" + str(len(minima)), min_no, bayesopt_pathsample_exec)
        if check_good_cell(coords[-6:], v_min, cluster_flag, ortho_flag):
            output_coords(coords, len(minima))
        else:
            output_coords(coords, len(minima), True)
        if len(minima) == batch_size:
            return len(minima) # Exit function.
    return len(minima)

def extract_lowest_batch(batch_size: int, v_min: float, cluster_flag: bool, ortho_flag: bool, bayesopt_pathsample_exec: str):
    '''Extract coordinates from lowest minima in import_min min.data file, then outputs corresponding coords file.
    Doesn't output coords for minima that have inappropriate cell volumes. A maximum of batch_size minima are extracted.
    Returns number of extracted minima'''
    min_data = np.loadtxt(os.path.join("Topo_Batch", "import_min", "min.data"))
    num_minima = np.shape(min_data)[0]
    min_energies = min_data[:,0]
    if num_minima > batch_size: # Number of minima exceeds batch size. Select lowest valued minima.
        minima = np.argsort(min_energies)[:batch_size]
        num_minima = batch_size
    else: # Number of minima doesn't exceed batch size. Select all of them.
        minima = np.array([i for i in range(num_minima)])
    for i in range(num_minima):
        coords = extract_min("import_min", "extract_batch" + str(i+1), minima[i]+1, bayesopt_pathsample_exec)
        if check_good_cell(coords[-6:], v_min, cluster_flag, ortho_flag):
            output_coords(coords, str(i+1))
        else:
            output_coords(coords, str(i+1), True)
    return num_minima

def extract_batch_single(v_min: float, cluster_flag: bool, ortho_flag: bool, bayesopt_pathsample_exec: str):
    coords = extract_min("import_min", "extract_min", 1, bayesopt_pathsample_exec)
    if check_good_cell(coords[-6:], v_min, cluster_flag, ortho_flag):
        output_coords(coords, 1)
    else:
        output_coords(coords, 1, True)

def extract_min(input_directory: str, target_directory: str, min: int, bayesopt_pathsample_exec: str):
    '''Open a points.min file from the input_directory. Extract and return coords for a specified min in target_directory.
    "Min" is an integer specifying the minima index, beginning from 1.'''
    target_path = os.path.join("Topo_Batch", target_directory)
    subprocess.Popen(["cp", "-r", "Topo_Batch_template/extract_batch", target_path]).wait() # Copy topo_batch template directory
    input_file = "Topo_Batch/" + input_directory + r"/points.min"
    subprocess.Popen(["cp", "-r", input_file, target_path + r"/points.min"]).wait() # Copy points.min file
    pathdata_path = os.path.join("Topo_Batch", target_directory, "pathdata")
    with open(pathdata_path, "a") as file: # Add EXTRACTMIN keyword to pathdata.
        file.writelines("EXTRACTMIN " + str(min))
    subprocess.Popen([bayesopt_pathsample_exec], cwd = target_path).wait() # Run PATHSAMPLE.
    coords_path = os.path.join("Topo_Batch", target_directory, "extractedmin")
    coords = np.loadtxt(coords_path) # Read extractedmin file
    return coords

def output_coords(data: np.ndarray, index: int, bad_flag: bool = False):
    '''Output coords for AF minima in GMIN format.
    If bad_flag is True, coords give inappropriate cell volume and will instead
    be output to bad_coords file.
    Index specifies the point number, beginning from 1.'''
    coords = np.concatenate((np.zeros(3), np.asarray(data)))
    rows = int(len(data) / 3) + 1
    coords = np.reshape(coords, (rows, 3))
    if (bad_flag):
        coords_path = os.path.join("coords", "coords" + str(index) +"_bad")
    else:
        coords_path = os.path.join("coords", "coords" + str(index))
    np.savetxt(coords_path, coords)

def gen_next_training(num_points: int):
    '''Reads training file and adds sampling points coordinates. Returns new training file. '''
    training = np.loadtxt(os.path.join("training"))
    for i in range(num_points):
        coords_path = os.path.join("coords", "coords" + str(i+1))
        if os.path.exists(coords_path) == False:
            coords_path = os.path.join("coords", "coords" + str(i+1) + "_bad")
        coords = np.loadtxt(coords_path)
        coords = np.delete(coords, 0, 0) # # Ignore first row corresponding to origin
        coords = np.array([coords.flatten()])
        training = np.concatenate((training, coords), axis = 0)
    return training

def gen_next_response(num_points: int, E_max: float):
    '''Reads response file and adds sampling points energies. Returns new response file. '''
    response = np.loadtxt(os.path.join("response"))
    extra_response = util.scrape_response(num_points)
    extra_response = np.clip(extra_response, None, E_max)
    response = np.concatenate((response, extra_response), axis = 0)
    return response

def prep_next_run(training: np.array, response: np.array, num_minima: int,
                   sampling_mode: int, hyperparams_flag: bool, gpplateau_flag: bool):
    subprocess.Popen(["mkdir", "next_run"]).wait() # Make next_run directory
    subprocess.Popen(["cp", "-r", "bayesopt.py", "next_run/bayesopt.py"]).wait() # Copy bayesopt script
    subprocess.Popen(["cp", "-r", "data", "next_run/data"]).wait() # Copy BayesOpt data file
    update_BayesOpt_keyword(os.path.join("next_run", "data"), num_minima) # Update BayesOpt data file
    np.savetxt(os.path.join("next_run", "training"), training) # Save new training file
    np.savetxt(os.path.join("next_run", "response"), response) # Save new response file
    subprocess.Popen(["cp", "-r", "funcbounds", "next_run/funcbounds"]).wait() # Copy BayesOpt funcbounds file
    if (hyperparams_flag):
        subprocess.Popen(["cp", "-r", "hyperparams", "next_run/hyperparams"]).wait() # Copy BayesOpt hyperparams file
    if (gpplateau_flag):
        subprocess.Popen(["cp", "-r", "gpplateau", "next_run/gpplateau"]).wait() # Copy BayesOpt gpplateau file
    subprocess.Popen(["cp", "-r", "singlepoint_template", "next_run/singlepoint_template"]).wait() # Copy singlepoint_template directory
    subprocess.Popen(["mkdir", "next_run/singlepoint"]).wait() # Make singlepoint directory
    subprocess.Popen(["mkdir", "coords"], cwd = "next_run").wait() # Make coords directory
    if sampling_mode == 0:
        return None # Exit function early if topological batch script undesired.
    subprocess.Popen(["mkdir", "Topo_Batch"], cwd = "next_run").wait() # Make Topo_Batch directory
    subprocess.Popen(["cp", "-r", "Topo_Batch_template", "next_run/Topo_Batch_template"]).wait() # Copy topo_batch_template directory
    subprocess.Popen(["rm", "next_run/Topo_Batch_template/gpfit.txt"]) # Delete topo_batch_template/gpfit.txt file.
    prep_odata_template("min", False, num_minima)
    prep_odata_template("import_min", True, num_minima)
    if sampling_mode == 1:
        prep_odata_template("path", False, num_minima)
        prep_odata_template("import_path", True, num_minima)

def update_BayesOpt_keyword(file_path: TextIO, num_minima: int) -> list:
    '''Read (o)data file and update number of data points in BayesOpt keyword line.'''
    with open(file_path) as file:
        data = file.readlines()
    keyword = "BAYESOPT"
    line_index = 0
    for line in data:
        if keyword in line:
            changed_line = line.split()
            changed_line[1] = str(int(changed_line[1]) + num_minima)
            changed_line = ' '.join(changed_line) + "\n"
            data[line_index] = changed_line
        line_index += 1
    with open(file_path, 'w') as file:
        file.writelines(data)

def prep_odata_template(directory: str, pathdata: bool, num_minima: int):
    '''Prep "directory" template in next_run/Topo_Batch_template running odata/pathsample'''
    new_directory = "next_run/Topo_Batch_template/" + directory
    subprocess.Popen(["cp", "-r", "next_run/training", new_directory + "/training"]).wait() # Overwrite training
    subprocess.Popen(["cp", "-r", "next_run/response", new_directory + "/response"]).wait() # Overwrite response
    if (pathdata): # PATHSAMPLE
        update_BayesOpt_keyword(os.path.join("next_run", "Topo_Batch_template", directory, "odata.connect"), num_minima) # Update odata.connect
    else: # OPTIM
        update_BayesOpt_keyword(os.path.join("next_run", "Topo_Batch_template", directory, "odata"), num_minima) # Update odata


if __name__ == "__main__":
    main()
    
