#!/bin/bash
#
# This is the shell script to run the example locally.
#
# There are 4 MPI libraries tested in this example

# (1) GNU OpenMPI
module purge
module load foss/2023a
mpicc mpi-hello.c -o hello-GNU-OpenMPI.exe 1> output 2> err
mpirun -np 8 hello-GNU-OpenMPI.exe 1>> output 2>> err
mpif77 mpi-hello.f -o hello-GNU-OpenMPI-fortran.exe 1>> output 2>> err
mpirun -np 8 hello-GNU-OpenMPI-fortran.exe 1>> output 2>> err
[ -s err ]
if [ $? -eq 0 ]
then
  echo "FAIL (GNU+OpenMPI)"
else
  echo "PASS (GNU+OpenMPI)" 
fi
rm output err


# (2) Intel impi
module purge 
module load intel/2023a
mpicc mpi-hello.c -o hello-Intel-impi.exe 1> output 2> err
mpirun -np 8 ./hello-Intel-impi.exe 1>> output 2>> err
mpiifort mpi-hello.f -o hello-Intel-impi-fortran.exe 1>> output 2>> err
mpirun -np 8 ./hello-Intel-impi-fortran.exe 1>> output 2>> err
[ -s err ]
if [ $? -eq 0 ]
then
  echo "FAIL (Intel+iMPI)"
else
  echo "PASS (Intel+iMPI)" 
fi
rm output err

# (3) Intel OpenMPI
module purge 
module load intel/2023a OpenMPI &> /dev/null
mpicc mpi-hello.c -o hello-Intel-OpenMPI.exe 1> output 2> err
mpirun -n 8 ./hello-Intel-OpenMPI.exe 1>> output 2>> err       # this combination does not work with srun
mpifort mpi-hello.f -o hello-Intel-OpenMPI-fortran.exe 1>> output 2>> err
mpirun -np 8 ./hello-Intel-OpenMPI-fortran.exe 1>> output 2>> err    #this combination does not with with srun
[ -s err ]
if [ $? -eq 0 ]
then
  echo "FAIL (Intel+OpenMPI)"
else
  echo "PASS (Intel+OpenMPI)" 
fi
rm output err

# (4) GCC MPICH
module purge
ml GCC OpenMPI
mpicc mpi-hello.c -o hello-MPICH.exe 1> output 2> err
mpirun -np 8 ./hello-MPICH.exe 1>> output 2>> err
mpif77 mpi-hello.f -o hello-MPICH-fortran.exe 1>> output 2>> err
mpirun -np 8 ./hello-MPICH-fortran.exe 1>> output 2>> err
[ -s err ]
if [ $? -eq 0 ]
then
  echo "FAIL (GNU+MPICH)"
else
  echo "PASS (GNU+MPICH)" 
fi
rm output err

