# PyGMINBayesOpt

The training\_generator module assumes the following working directory structure:

\dirtree{%
.1 /.
.2 BayesOpt.
.2 coords.
.2 singlepoint.
.2 singlepoint\_template.
.3 (GMIN input files).
}
The bayesopt module assumes the following working directory structure:

\dirtree{%
.1 /.
.2 coords.
.2 singlepoint.
.2 singlepoint\_template.
.3 (GMIN input files).
.2 Topo\_Batch.
.2 Topo\_Batch\_template.
.3 connect\_path.
.4 (PATHSAMPLE input file).
.3 extract\_batch.
.4 (PATHSAMPLE input file).
.3 import\_min.
.4 (PATHSAMPLE/OPTIM input files).
.3 import\_path.
.4 (PATHSAMPLE/OPTIM input files).
.3 min.
.4 (OPTIM input file).
.3 path.
.4 (OPTIM input file).
.2 (BayesOpt GMIN input and output files).
}
