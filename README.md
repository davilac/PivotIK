# PivotIK
Python implementation of the hybrid inverse kinematics algorithm proposed in the paper "Real-time Inverse Kinematics for Robotic Manipulation under Remote Center of Motion Constraint using Memetic Evolution".


![alt text](https://raw.githubusercontent.com/davilac/PivotIK/main/docs/media/pivotik_6Dlissajous.gif "PivotIK tracking a Lissajous path")

## Directory Structure

    ├── qinits                      # Initial robot configurations for benchmarking
    ├── targets                     # Target points for benchmarking
    ├── docs                        # Figures and media 
    ├── urdfs                       # URDFs of the robots used for benchmarking
    ├── benchmark.py                # Launches the benchmark between PivotIK and other IK solvers
    ├── ik_problem.py               # Inverse Kinematics problem definition
    ├── pivotik_lib.py              # Implements the hybrid optimization algorithm


## Prerequisites
* Pinocchio (https://github.com/stack-of-tasks/pinocchio)
* Pygmo (https://github.com/esa/pygmo2)
* Numpy
* Scipy
* Pandas

Prerequisited needed for benchmarking with optimization-based solvers (For optimized PivotIK)
* CASADI (with the plugins for IPOPT and OSQP)
* IPOPT
* OSQP

## Optimized PivotIK
The optimized version of PivotIK, reimplemented in C++14 and compatible with ROS, will be released soon. 

## Maintainers
- Ana Davila: [Email](mailto:davila.ana@mein.nagoya-u.ac.jp)
- Jacinto Colan: [Email](mailto:colan@robo.mein.nagoya-u.ac.jp)