# %%
import pinocchio as pin
import numpy as np
import random
import os
import time
import csv
from scipy.spatial.transform import Rotation as R
import sys
import copy
import pandas as pd
import pygmo as pg

# Add the library path to the Python module search path
library_path = os.getcwd()
sys.path.append(library_path)

from pivotik_lib import MemOpt
from ik_problem import IKProblem


seed = 0
random.seed(seed)
np.random.seed(seed)

# Paths
cwd = os.getcwd()
save_path = os.path.join(cwd, "results")
qinits_dataset_path = os.path.join(cwd, "qinits")
targets_dataset_path = os.path.join(cwd, "targets")

# Benchmark parameters
robot_id = 1
mode = "c6d"
n_samples = 250
n_gen = 1000

# Load qinit and target files
if mode in ["c6d"] and robot_id == 1:
    q_init_filename = os.path.join(
        qinits_dataset_path, "qinit_robot1_constrained_samples250.csv"
    )
    target_filename = os.path.join(
        targets_dataset_path, "targets_robot1_constrained_samples250.csv"
    )
elif mode in ["u3d", "u6d"] and robot_id == 1:
    q_init_filename = os.path.join(
        qinits_dataset_path, "qinit_robot1_unconstrained_samples250.csv"
    )
    target_filename = os.path.join(
        targets_dataset_path, "targets_robot1_unconstrained_samples250.csv"
    )
elif mode in ["c6d"] and robot_id == 2:
    q_init_filename = os.path.join(
        qinits_dataset_path, "qinit_robot2_constrained_samples250.csv"
    )
    target_filename = os.path.join(
        targets_dataset_path, "targets_robot2_constrained_samples250.csv"
    )
elif mode in ["u3d", "u6d"] and robot_id == 2:
    q_init_filename = os.path.join(
        qinits_dataset_path, "qinit_robot2_unconstrained_samples250.csv"
    )
    target_filename = os.path.join(
        targets_dataset_path, "targets_robot2_unconstrained_samples250.csv"
    )

q_init_pd = pd.read_csv(q_init_filename)
target_pd = pd.read_csv(target_filename)

# %%
# IK solver parameters (Optimized for each solver through grid search)

# ABC
abc_params_dict = {}
abc_params_dict["pop_size"] = 500
abc_params_dict["abc_limit"] = 40

# CMAES
cmaes_params_dict = {}
cmaes_params_dict["pop_size"] = 500
cmaes_params_dict["cmaes_sigma0"] = 0.1

# COMPASS SEARCH
cs_params_dict = {}
cs_params_dict["pop_size"] = 100
cs_params_dict["cs_max_fevals"] = 100
cs_params_dict["cs_start_range"] = 0.001
cs_params_dict["cs_stop_range"] = 0.0001
cs_params_dict["cs_reduction_coeff"] = 0.5

# DE1 (best/1/exp)
de1_params_dict = {}
de1_params_dict["pop_size"] = 50
de1_params_dict["de_F"] = 0.99
de1_params_dict["de_CR"] = 0.99
de1_params_dict["de_variant"] = 1

# DE2 (best/1/bin)
de2_params_dict = {}
de2_params_dict["pop_size"] = 50
de2_params_dict["de_F"] = 0.99
de2_params_dict["de_CR"] = 0.99
de2_params_dict["de_variant"] = 6

# PSO
pso_params_dict = {}
pso_params_dict["pop_size"] = 100
pso_params_dict["pso_omega"] = 0.75
pso_params_dict["pso_eta1"] = 2.0
pso_params_dict["pso_eta2"] = 0.5
pso_params_dict["pso_max_vel"] = 1.0
pso_params_dict["pso_variant"] = 5
pso_params_dict["pso_neighb_type"] = 1
pso_params_dict["pso_neighb_param"] = 4

# SA
sa_params_dict = {}
sa_params_dict["pop_size"] = 10
sa_params_dict["sa_Ts"] = 10
sa_params_dict["sa_Tf"] = 0.001
sa_params_dict["sa_n_T_adj"] = 50
sa_params_dict["sa_n_range_adj"] = 10
sa_params_dict["sa_bin_size"] = 20
sa_params_dict["sa_start_range"] = 0.01

# SADE
sade_params_dict = {}
sade_params_dict["pop_size"] = 50
sade_params_dict["sade_variant"] = 6
sade_params_dict["sade_variant_adptv"] = 2

# SGA
sga_params_dict = {}
sga_params_dict["pop_size"] = 1000
sga_params_dict["sga_cr"] = 0.95
sga_params_dict["sga_m"] = 0.7
sga_params_dict["sga_param_m"] = 0.9
sga_params_dict["sga_param_s"] = 2
sga_params_dict["sga_crossover"] = "binomial"
sga_params_dict["sga_mutation"] = "uniform"
sga_params_dict["sga_selection"] = "tournament"


# xNES
xnes_params_dict = {}
xnes_params_dict["pop_size"] = 500
xnes_params_dict["xnes_eta_mu"] = 0.9
xnes_params_dict["xnes_eta_sigma"] = 0.9
xnes_params_dict["xnes_eta_b"] = 0.5
xnes_params_dict["xnes_sigma0"] = 0.1

# IPOPT
ipopt_params_dict = {}
ipopt_params_dict["pop_size"] = 1

# NLOPT
cobyla_params_dict = {}
cobyla_params_dict["pop_size"] = 1
bobyqa_params_dict = {}
bobyqa_params_dict["pop_size"] = 1
sbplx_params_dict = {}
sbplx_params_dict["pop_size"] = 1
slsqp_params_dict = {}
slsqp_params_dict["pop_size"] = 1
lbfgs_params_dict = {}
lbfgs_params_dict["pop_size"] = 1

# PivotIK
pivotik_params_dict = {}
pivotik_params_dict["pop_size"] = 10
pivotik_params_dict["elite_size"] = 4


def restart_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def perr(model, data, q_it, B_H_des):
    pin.framesForwardKinematics(model, data, q_it)
    p_FeeDes = B_H_des.translation
    p_FeeAct = data.oMf[ee_id].translation
    perr = p_FeeDes - p_FeeAct
    return np.linalg.norm(perr)


def oerr(model, data, q_it, B_H_des):
    pin.framesForwardKinematics(model, data, q_it)
    o_FeeDes = R.from_matrix(B_H_des.rotation).as_quat()
    o_FeeAct = R.from_matrix(data.oMf[ee_id].rotation).as_quat()
    o_FeeAct_conj = o_FeeAct
    alpha = np.abs(np.dot(o_FeeDes, o_FeeAct_conj))
    oerr = 0 if alpha >= 1 else 2 * np.arccos(alpha)
    return np.linalg.norm(oerr)


def compute_residual_EE_log6(model, data, q_it, B_X_EEdes):
    pin.framesForwardKinematics(model, data, q_it)
    B_H_act_Fee = data.oMf[ee_id]

    Fee_Hact_B = pin.SE3(
        B_H_act_Fee.rotation.transpose(),
        np.matmul(
            -B_H_act_Fee.rotation.transpose(),
            B_H_act_Fee.translation,
        ),
    )
    B_M_delta = B_X_EEdes.act(Fee_Hact_B)
    res_ee = pin.log(B_M_delta).vector
    err_ee = np.linalg.norm(res_ee)

    return res_ee, err_ee


def compute_residual_RCM(
    model,
    data,
    q_it,
):
    pin.framesForwardKinematics(model, data, q_it)
    B_H_Fprercm = data.oMf[pre_rcm_joint_id]
    B_H_Fpostrcm = data.oMf[post_rcm_joint_id]

    # Computing RCM error
    ps = B_H_Fpostrcm.translation - B_H_Fprercm.translation
    pr = B_X_IP.translation - B_H_Fprercm.translation
    ps_hat = ps / np.linalg.norm(ps)

    B_p_Frcm = B_H_Fprercm.translation + np.transpose(pr) @ np.outer(ps_hat, ps_hat)

    pe = B_X_IP.translation - B_p_Frcm
    res_rcm = -np.dot(pe, pe)
    err_rcm = np.linalg.norm(pe)

    return res_rcm, ps, pr, pe, err_rcm


# %%
# Benchmark loop
# Uncomment the solver you want to benchmark

for solver in [
    # "abc",
    # "cmaes",
    # "cs",
    # "de1",
    # "de2",
    # "pso",
    # "sade",
    # "sa",
    # "sga",
    # "xnes",
    # "ipopt",
    # "nlopt_cobyla",
    # "nlopt_bobyqa",
    # "nlopt_sbplx",
    # "nlopt_slsqp",
    # "nlopt_lbfgs",
    "pivotik",
]:
    if solver == "abc":
        params = abc_params_dict
    elif solver == "cmaes":
        params = cmaes_params_dict
    elif solver == "cs":
        params = cs_params_dict
    elif solver == "de1":
        params = de1_params_dict
    elif solver == "de2":
        params = de2_params_dict
    elif solver == "pso":
        params = pso_params_dict
    elif solver == "sade":
        params = sade_params_dict
    elif solver == "sga":
        params = sga_params_dict
    elif solver == "sa":
        params = sa_params_dict
    elif solver == "xnes":
        params = xnes_params_dict
    elif solver == "ipopt":
        params = ipopt_params_dict
    elif solver == "nlopt_cobyla":
        params = cobyla_params_dict
    elif solver == "nlopt_bobyqa":
        params = bobyqa_params_dict
    elif solver == "nlopt_sbplx":
        params = sbplx_params_dict
    elif solver == "nlopt_slsqp":
        params = slsqp_params_dict
    elif solver == "nlopt_lbfgs":
        params = lbfgs_params_dict
    elif solver == "pivotik":
        params = pivotik_params_dict
    else:
        raise ValueError("Invalid solver")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    csv_filename = os.path.join(
        save_path,
        f"benchmark_{solver}_robot{robot_id}_m{mode}.csv",
    )

    # Initialize CSV file header
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(
            [
                "iteration",
                "solver",
                "robot_id",
                "time",
                "success",
                "perror",
                "oerror",
                "log6_error",
                "rerror",
                "n_gen",
            ]
        )

    if robot_id == 1:
        urdf_path = os.path.join(cwd, "urdfs/robot1_openrst.urdf")
    elif robot_id == 2:
        urdf_path = os.path.join(cwd, "urdfs/robot2_hyperrst.urdf")
    else:
        raise ValueError("Invalid robot id")

    # Load urdf model
    model = pin.buildModelFromUrdf(urdf_path)

    # Create data required by the algorithms
    data = model.createData()

    # Set joint limits
    q_max = model.upperPositionLimit
    q_min = model.lowerPositionLimit

    # Print model information
    print(f"model name: {model.name}")
    print("Number of frames: {0:2d}".format(model.nframes))
    print("Number of Joints: {0:2d}".format(model.njoints))
    print("Number of DOF: {0:2d}".format(model.nq))

    # Set default robot parameters
    if robot_id == 1:
        # ?For robot1
        ee_id = model.getFrameId("link_ee")
        pre_rcm_joint_id = model.getFrameId("joint_interface")
        post_rcm_joint_id = model.getFrameId("joint_pitch")
        # Target Pose
        q_init = np.array([0.0, -0.17, 0.0, 1.31, 0.0, 1.57, 0.0, 0.0, 0.0])
        # RCM Pose
        B_X_IP = pin.SE3(np.eye(3), np.array([0.406, -0.024, 0.415]))
        # Initial constrained Pose
        path_center = np.array([0.488, -0.024, 0.279])

    elif robot_id == 2:
        # ?For robot2
        ee_id = model.getFrameId("link_ee")
        pre_rcm_joint_id = model.getFrameId("joint_interface")
        post_rcm_joint_id = model.getFrameId("joint_pitch_1")
        # Target Pose
        q_init = np.array([0.0, -0.17, 0.0, 1.31, 0.0, 1.57, 0.0, 0.0, 0.0, 0.0, 0.0])
        # RCM Pose
        B_X_IP = pin.SE3(np.eye(3), np.array([0.406, -0.024, 0.415]))
        # Initial constrained Pose
        path_center = np.array([0.476, -0.025, 0.264])
    else:
        raise ValueError("Invalid robot id")

    # Solving parameters
    n_q = model.nq

    ik_params = {}
    ik_params["seed"] = seed
    ik_params["base_link"] = "base_link"
    ik_params["ee_link"] = "link_ee"
    ik_params["ee_id"] = ee_id
    ik_params["pre_rcm_id"] = pre_rcm_joint_id
    ik_params["post_rcm_id"] = post_rcm_joint_id
    ik_params["B_X_IP"] = B_X_IP
    ik_params["mode"] = mode
    ik_params["eps.pos"] = 1e-4
    ik_params["eps.ori"] = 1e-4
    ik_params["eps.log6"] = 1e-4
    ik_params["eps.rcm"] = 1e-4
    ik_params["print_level"] = 0

    # Selection of fitness function [log6, pos+log3, pos+euler, pos+rotmat, pos+quat]
    ik_params["fitness"] = "log6"

    # %%
    # Benchmark loop

    count_success = 0
    time_total = 0
    success_time_total = 0

    for it in range(n_samples):
        # print("Iteration: ", it)
        succeed = False

        # Extract initial guess from q_init_pd file
        q_init = np.zeros(model.nq)
        for id in range(model.nq):
            q_init[id] = q_init_pd.iloc[it, id + 1]

        # Extract target pose from target_pd file
        pos_target = np.array(
            [
                target_pd.iloc[it, 1],
                target_pd.iloc[it, 2],
                target_pd.iloc[it, 3],
            ]
        )
        # Convert target from quaternion to rotation matrix
        quat_target = np.array(
            [
                target_pd.iloc[it, 4],
                target_pd.iloc[it, 5],
                target_pd.iloc[it, 6],
                target_pd.iloc[it, 7],
            ]
        )
        ori_target = R.from_quat(quat_target).as_matrix()

        B_X_EEdes = pin.SE3(ori_target, pos_target)

        # Problem definition
        prob = pg.problem(IKProblem(model, data, B_X_EEdes, q_init, ik_params))

        # Create initial population
        restart_seed(seed)
        if mode in ["u3d", "u6d"]:
            pop = pg.population(prob, size=params["pop_size"] - 1, seed=seed)
            pop.push_back(q_init)
        elif mode in ["c6d"]:
            pop = pg.population(prob, size=params["pop_size"] - 1, seed=seed)
            pop.push_back(q_init)

        # Create algorithm
        if solver == "abc":
            algo = pg.algorithm(
                pg.bee_colony(gen=1, limit=params["abc_limit"], seed=seed)
            )
        elif solver == "cmaes":
            algo = pg.algorithm(
                pg.cmaes(
                    gen=1,
                    cc=-1,
                    cs=-1,
                    c1=-1,
                    cmu=-1,
                    sigma0=params["cmaes_sigma0"],
                    ftol=1e-6,
                    xtol=1e-6,
                    memory=True,
                    force_bounds=False,
                    seed=seed,
                )
            )
        elif solver == "cs":
            algo = pg.algorithm(
                pg.compass_search(
                    max_fevals=params["cs_max_fevals"],
                    start_range=params["cs_start_range"],
                    stop_range=params["cs_stop_range"],
                    reduction_coeff=params["cs_reduction_coeff"],
                )
            )
        elif solver == "de1":
            algo = pg.algorithm(
                pg.de(
                    gen=1,
                    F=params["de_F"],
                    CR=params["de_CR"],
                    variant=params["de_variant"],
                    ftol=1e-6,
                    xtol=1e-6,
                    seed=seed,
                )
            )
        elif solver == "de2":
            algo = pg.algorithm(
                pg.de(
                    gen=1,
                    F=params["de_F"],
                    CR=params["de_CR"],
                    variant=params["de_variant"],
                    ftol=1e-6,
                    xtol=1e-6,
                    seed=seed,
                )
            )
        elif solver == "pso":
            algo = pg.algorithm(
                pg.pso(
                    gen=1,
                    omega=params["pso_omega"],
                    eta1=params["pso_eta1"],
                    eta2=params["pso_eta2"],
                    max_vel=params["pso_max_vel"],
                    variant=params["pso_variant"],
                    neighb_type=params["pso_neighb_type"],
                    neighb_param=params["pso_neighb_param"],
                    memory=True,
                    seed=seed,
                )
            )
        elif solver == "sade":
            algo = pg.algorithm(
                pg.sade(
                    gen=1,
                    variant=params["sade_variant"],
                    variant_adptv=params["sade_variant_adptv"],
                    ftol=1e-6,
                    xtol=1e-6,
                    memory=True,
                    seed=seed,
                )
            )
        elif solver == "sa":
            algo = pg.algorithm(
                pg.simulated_annealing(
                    Ts=params["sa_Ts"],
                    Tf=params["sa_Tf"],
                    n_T_adj=params["sa_n_T_adj"],
                    n_range_adj=params["sa_n_range_adj"],
                    bin_size=params["sa_bin_size"],
                    start_range=params["sa_start_range"],
                    seed=seed,
                )
            )
        elif solver == "sga":
            algo = pg.algorithm(
                pg.sga(
                    gen=1,
                    cr=params["sga_cr"],
                    m=params["sga_m"],
                    param_m=params["sga_param_m"],
                    param_s=params["sga_param_s"],
                    crossover=params["sga_crossover"],
                    mutation=params["sga_mutation"],
                    selection=params["sga_selection"],
                    seed=seed,
                )
            )
        elif solver == "xnes":
            algo = pg.algorithm(
                pg.xnes(
                    gen=1,
                    eta_mu=params["xnes_eta_mu"],
                    eta_sigma=params["xnes_eta_sigma"],
                    eta_b=params["xnes_eta_b"],
                    sigma0=params["xnes_sigma0"],
                    ftol=1e-6,
                    xtol=1e-6,
                    memory=True,
                    force_bounds=False,
                    seed=0,
                )
            )
        elif solver == "ipopt":
            ip = pg.ipopt()
            ip.set_numeric_option("tol", 1e-9)
            ip.get_numeric_options()
            algo = pg.algorithm(ip)
        elif solver == "nlopt_cobyla":
            nl = pg.nlopt("cobyla")
            nl.xtol_rel = 1e-6
            algo = pg.algorithm(nl)
        elif solver == "nlopt_bobyqa":
            nl = pg.nlopt("bobyqa")
            nl.xtol_rel = 1e-6
            algo = pg.algorithm(nl)
        elif solver == "nlopt_sbplx":
            nl = pg.nlopt("sbplx")
            nl.xtol_rel = 1e-6
            algo = pg.algorithm(nl)
        elif solver == "nlopt_slsqp":
            nl = pg.nlopt("slsqp")
            nl.xtol_rel = 1e-6
            algo = pg.algorithm(nl)
        elif solver == "nlopt_lbfgs":
            nl = pg.nlopt("lbfgs")
            nl.xtol_rel = 1e-6
            algo = pg.algorithm(nl)
        elif solver == "pivotik":
            bounds = np.array([prob.get_bounds()[0], prob.get_bounds()[1]]).T
            algo = MemOpt(
                bounds=bounds,
                initial_guess=q_init,
                n_eval=n_gen,
                population_size=params["pop_size"],
                elite_size=params["elite_size"],
                problem=prob,
                params=ik_params,
            )
        else:
            raise ValueError("Invalid solver")

        # Set verbosity level
        algo.set_verbosity(0)

        # Measure computation time
        start_time = time.time()
        if solver in ["ipopt"]:
            n_gen = 1
        for gen in range(n_gen):
            try:
                pop = algo.evolve(pop)
            except:
                print("Error in evolution")
                break

            prob_extracted = prob.extract(IKProblem)

            prob_extracted.update_model(pop.champion_x)
            err_p = prob_extracted.perr()
            err_o = prob_extracted.oerr()
            _, _, _, _, err_r = prob_extracted.compute_residual_RCM()

            # Verify convergence to threshold
            if mode in ["u3d"]:
                if err_p < ik_params["eps.pos"]:
                    succeed = True
                    break
            elif mode in ["u6d"]:
                if err_p < ik_params["eps.pos"] and err_o < ik_params["eps.ori"]:
                    succeed = True
                    break
            elif mode in ["c6d"]:
                if (
                    err_p < ik_params["eps.pos"]
                    and err_o < ik_params["eps.ori"]
                    and err_r < ik_params["eps.rcm"]
                ):
                    succeed = True
                    break
        end_time = time.time()
        time_it = end_time - start_time
        time_total += time_it
        n_iter = gen + 1

        # Extract best individual from last generation
        q_sol = copy.deepcopy(pop.champion_x)

        if succeed:
            success_time_total += time_it
            count_success += 1
            print(
                "[",
                solver,
                "/",
                mode,
                "/",
                robot_id,
                "] Prob:\t",
                it,
                " solved. So far",
                count_success,
                "/",
                it + 1,
                " successes!",
            )
        else:
            print("[", solver, "/", mode, "/", robot_id, "] Prob:\t", it, " failed.")

        res_ee, err_ee = compute_residual_EE_log6(model, data, q_sol, B_X_EEdes)
        if ik_params["mode"] in ["c6d"]:
            res_rcm, ps, pr, pe, err_rcm = compute_residual_RCM(model, data, q_sol)
        else:
            err_rcm = 0.0
        err_p = perr(model, data, q_sol, B_X_EEdes)
        err_o = oerr(model, data, q_sol, B_X_EEdes)

        # Save results in CSV file
        with open(csv_filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    it,
                    solver,
                    robot_id,
                    time_it,
                    succeed,
                    err_p,
                    err_o,
                    err_ee,
                    err_rcm,
                    n_iter,
                ]
            )

    if count_success > 0:
        avg_success_time = success_time_total / count_success
    else:
        avg_success_time = 0

    print(
        "Success rate:",
        count_success / n_samples,
        "Total time:",
        time_total,
        "Avg success time:",
        avg_success_time,
        "Population size:",
        params["pop_size"],
        "Number of problems evaluated:",
        n_samples,
    )

# %%
