import numpy as np
import networkx as nx

def main():
   directory = None
   alpha = 0.8
   beta = 0.2
   batch_size = 3
   topo_batch(directory, alpha, beta, batch_size)

def topo_batch(directory: str, alpha: float, beta: float, batch_size: int):
   min_energies, ts_energies, ts_connections = read_data(directory)
   if np.size(min_energies) <= batch_size: # If fewer minima than batch size, select all minima as test points.
      batch = [i for i in range(min_energies)]
      output_batch("Monotonic", batch, directory)
      quit()
   energy_cutoff, barrier_cutoff = calc_cutoffs(min_energies, alpha, beta)
   min_energies_cut, banned_minima = cut_energies(min_energies, energy_cutoff)
   G = preprocess_network(min_energies_cut, ts_energies, ts_connections)
   batch = Monotonic(G, min_energies_cut)
   batch = order_batch(batch, min_energies)
   output_batch("Monotonic", batch, directory)
   if len(batch) < batch_size:
      batch = BarrierSelection(G, batch, banned_minima, barrier_cutoff, min_energies, ts_energies, ts_connections)
      batch = order_batch(batch, min_energies)
      output_batch("BarrierSelection", batch, directory)

def read_data(directory = None) -> tuple:
   '''Reads in min.data and ts.data files and returns tuple of minima energies, ts energies and ts connections.
   If there are three or less minima, instead return tuple of number of minima, 0 and 0.'''
   if directory == None:
      min_path = 'min.data'
      ts_path = 'ts.data'
   else:
      min_path = str(directory) + r"/min.data"
      ts_path = str(directory) + r"/ts.data"
   minima = np.genfromtxt(min_path)
   ts = np.genfromtxt(ts_path)
   if np.ndim(minima) == 1: # Handle edge case for if there is only one minima.
      num_min = 1
   else:
      num_min = np.size(minima, 0)
   if num_min <= 3: # Handle edge case for if there are three or fewer minima.
      return (num_min, 0, 0)
   min_energies = minima[:,0]
   if np.ndim(ts) == 1: # Handle edge case for if there is only one ts.
      ts_energies = np.array([[ts[0]]])
      ts_connections = np.array([ts[3:5].astype(int)])
      ts_connections = ts_connections-1
   else:
      ts_energies = ts[:,0]
      ts_connections = ts[:,3:5].astype(int)
      ts_connections = ts_connections-1
   return (min_energies, ts_energies, ts_connections)

def calc_cutoffs(min_energies: np.array, alpha: float, beta: float) -> float:
   '''Calculates energy and barrier cutoffs from minima energies.'''
   r = np.max(min_energies) - np.min(min_energies)
   energy_cutoff = np.min(min_energies)+alpha*r
   if (energy_cutoff > -0.5):
      energy_cutoff = -0.5
   barrier_cutoff = beta*r
   if (barrier_cutoff > 0.2):
      barrier_cutoff = 0.2
   return (energy_cutoff, barrier_cutoff)

def cut_energies(min_energies: np.array, energy_cutoff: float) -> tuple:
   '''Truncate all minima energies above energy cutoff to a value just above the energy cutoff.
   Returns tuple of truncated minima energies array and list of truncated minima index positions.'''
   banned_minima = []
   min_energies_cut = min_energies.copy()
   for i in range(np.size(min_energies_cut, axis=0)):
      if (min_energies_cut[i] > energy_cutoff):
         min_energies_cut[i] = energy_cutoff + 1e-3
         banned_minima.append(i)
   return min_energies_cut, banned_minima

def preprocess_network(min_energies: np.array, ts_energies: np.array, ts_connections: np.array) -> nx.Graph:
   '''Create graph object from minima energies, ts energies and ts connections.'''
   G = nx.Graph()
   for i in range(np.size(min_energies, axis=0)):
      G.add_node(i, energy=min_energies[i])
   for i in range(np.size(ts_connections, axis=0)):
      G.add_edge(int(ts_connections[i,0]), int(ts_connections[i,1]), energy=ts_energies[i])
   return G

def Monotonic(G, min_energies):
   MSB = []
   for i in range(np.size(min_energies, axis=0)): # Sum over minima
      monotonic = True
      energy_i = min_energies[i]
      i_edges = list(G.edges())
      for j in list(i_edges): # Sum over each minima pair connected via a ts.
         min1 = j[0] 
         min2 = j[1]
         if ((min1 == i) or (min2 == i)): 
            if (min1 == min2):
               continue
            if (min1 == i):
               energy_j = min_energies[min2]
            else:
               energy_j = min_energies[min1]
            if (energy_j <= energy_i): # If min i is connected to higher energy minima, label as not monotonic.
               monotonic = False
      if (monotonic):
         MSB.append(i)
   return MSB

def BarrierSelection(G, batch, banned_minima, barrier_cutoff, min_energies, ts_energies, ts_connections):
   for i in range(np.size(min_energies, axis=0)): # Sum over all minima
       if (i in batch): # Ignore lowest energy minima of each basin
           continue
       if (i in banned_minima): # Ignore high energy minima.
           continue
       allowed = True
       for j in batch:
          barrier_height = BarrierHeight(G, i, j, min_energies, ts_energies, ts_connections)
          h1 = barrier_height - min_energies[i]
          h2 = barrier_height - min_energies[j]
          if h1 < h2:
             h = h1
          else:
             h = h2
          if (h < barrier_cutoff):
             allowed = False
       if (allowed):
          batch.append(i)
   return batch

def BarrierHeight(G, i, j, min_energies, ts_energies, ts_connections):
   G_cutting = G.copy()
   start_energy = max(min_energies[i], min_energies[j]) + 2.0
   for k in range(500):
      current_energy = start_energy-(k*0.01)
      for j in range(np.size(ts_energies)):
          if (ts_energies[j] > current_energy):
              try:
                 G_cutting.remove_edge(ts_connections[j,0], ts_connections[j,1])
              except:
                 pass
          connected_minima = nx.node_connected_component(G_cutting, j)
      if (i not in connected_minima):
         return current_energy
      if (len(connected_minima) == 1):
         return current_energy
      
def order_batch(batch: list, min_energies: np.array) -> list:
   '''Orders a batch by ascending energy. Returns ordered batch.'''
   ordered_batch = []
   batch_energies = [float(min_energies[min]) for min in batch]
   print(batch_energies)
   for energy in sorted(batch_energies):
      min_index = batch[batch_energies.index(energy)]
      ordered_batch.append(min_index)
   return ordered_batch
      
def output_batch(file: str, batch: list, directory = None):
   if directory == None:
      file_path = file
   else:
      file_path = str(directory) + r"/" + file
   with open(file_path, 'w') as f:
      for min in batch:
         f.write("%s\n" % min)

if __name__ == "__main__":
    main()