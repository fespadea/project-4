#!/bin/bash -x

module load spectrum-mpi
module load spectrum-mpi cuda/11.2

#####################################################################################################
# Launch N tasks per compute node allocated. Per below this launches 32 MPI rank per compute node.
# taskset insures that hyperthreaded cores are skipped.
#####################################################################################################

RANKS_PER_NODE=$1
sMult=$1
dataSizeValue=$1
taskset -c 0-159:4 mpirun -N $RANKS_PER_NODE ./proj $sMult $dataSizeValue