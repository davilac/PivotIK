# %%
import numpy as np
import pinocchio as pin
import pygmo as pg
from scipy.spatial.transform import Rotation as R


class IKProblem:
    def __init__(self, model, data, B_X_EEdes, q_init, params):
        self.model = model
        self.data = data
        self.B_X_EEdes = B_X_EEdes
        self.q_init = q_init
        self.ee_id = params.get("ee_id")
        self.pre_rcm_id = params.get("pre_rcm_id")
        self.post_rcm_id = params.get("post_rcm_id")
        self.B_X_IP = params.get("B_X_IP")
        self.mode = params.get("mode")
        self.fitness_type = params.get("fitness")
        self.nq = model.nq
        self.q_min = model.lowerPositionLimit
        self.q_max = model.upperPositionLimit
        self.Kee = 1.0
        self.Krcm = 100

    def fitness(self, x):
        self.update_model(x)

        if self.mode == "u3d":
            fitness = self.perr()
        elif self.mode == "u6d":
            res_ee, err_ee = self.compute_residual_EE_log6()
            fitness = err_ee
        elif self.mode == "c6d":
            res_rcm, ps, pr, pe, err_rcm = self.compute_residual_RCM()
            if self.fitness_type == "log6":
                res_ee, err_ee = self.compute_residual_EE_log6()
                fitness = err_ee + err_rcm
            elif self.fitness_type == "pos+log3":
                fitness = self.perr() + self.oerr("log3") + err_rcm
            elif self.fitness_type == "pos+euler":
                fitness = self.perr() + self.oerr("euler") + err_rcm
            elif self.fitness_type == "pos+rotmat":
                fitness = self.perr() + self.oerr("rotmat") + err_rcm
            elif self.fitness_type == "pos+quat":
                fitness = self.perr_norm() + self.oerr("quat") + err_rcm
            else:
                raise ValueError("Invalid fitness type")
        return [fitness]

    def get_bounds(self):
        return (self.q_min, self.q_max)

    def compute_grad_unconstrained(self, q_it):
        # Compute EE error
        res_ee, err_ee = self.compute_residual_EE_log6()

        # Compute EE Jacobian
        B_Jb_EE = self.compute_Jacobian_EE(q_it)

        # Compute Gradient
        qdot = np.linalg.pinv(B_Jb_EE) @ (self.Kee * res_ee)

        return -qdot

    def compute_grad_constrained(self, q_it):
        # Compute residuals
        res_ee, err_ee = self.compute_residual_EE_log6()
        res_rcm, ps, pr, pe, err_rcm = self.compute_residual_RCM()

        # Compute task Jacobians
        B_Jb_EE = self.compute_Jacobian_EE(q_it)
        B_Jb_RCM = self.compute_Jacobian_RCM(q_it, ps, pr, pe)
        B_Jb_RCM = B_Jb_RCM.reshape(self.nq, 1)

        pinv_B_Jb_RCM = np.linalg.pinv(B_Jb_RCM)
        pinv_B_Jb_EE = np.linalg.pinv(B_Jb_EE)

        # Compute Jacobian-based gradient
        task_ee_v = pinv_B_Jb_EE @ (self.Kee * res_ee)
        task_rcm_v = pinv_B_Jb_RCM * (self.Krcm * res_rcm)

        task_ee_null = np.eye(self.nq) - pinv_B_Jb_EE @ B_Jb_EE
        task_rcm_null = np.eye(self.nq) - np.outer(pinv_B_Jb_RCM, B_Jb_RCM)
        task_ee_v = task_ee_v.reshape(self.nq, 1)
        task_rcm_v = task_rcm_v.reshape(self.nq, 1)

        qdot = task_ee_v + task_ee_null @ task_rcm_v
        qdot = qdot.reshape(self.nq)

        return -qdot

    # Jacobian-based gradient computation
    def gradient(self, x):
        self.update_model(x)
        if self.mode in ["c6d"]:
            return self.compute_grad_constrained(x)
        elif self.mode in ["u3d", "u6d"]:
            return self.compute_grad_unconstrained(x)
        else:
            raise ValueError("Invalid mode")

    # First-order finite difference gradient computation
    # def gradient(self, x):
    #     return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    def update_model(self, q_it):
        pin.framesForwardKinematics(self.model, self.data, q_it)

    def perr(self):
        p_FeeDes = self.B_X_EEdes.translation
        p_FeeAct = self.data.oMf[self.ee_id].translation

        perr = p_FeeDes - p_FeeAct
        return np.linalg.norm(perr)

    def perr_norm(self):
        p_FeeDes = self.B_X_EEdes.translation
        p_FeeAct = self.data.oMf[self.ee_id].translation

        d = np.linalg.norm(p_FeeDes - p_FeeAct)
        L = 1.627  # robot1
        # L = 1.643 # robot2
        lambda_val = np.linalg.norm(p_FeeAct)

        return (np.pi * d) / (np.sqrt((L + d) * (lambda_val + d)))

    def oerr(self, type="quat"):
        if type == "log3":
            return self.oerr_log3()
        elif type == "euler":
            return self.oerr_euler()
        elif type == "rotmat":
            return self.oerr_rotmat("trans")
        elif type == "quat":
            return self.oerr_quat()
        else:
            raise ValueError("Invalid fitness type to compute orientation error")

    def oerr_log3(self):
        B_R_EEact = self.data.oMf[self.ee_id].rotation
        oerr = pin.log3(self.B_X_EEdes.rotation @ B_R_EEact.transpose())
        return np.linalg.norm(oerr)

    def oerr_euler(self):
        o_FeeDes = R.from_matrix(self.B_X_EEdes.rotation).as_euler("zyx")
        o_FeeAct = R.from_matrix(self.data.oMf[self.ee_id].rotation).as_euler("zyx")
        oerr = o_FeeDes - o_FeeAct
        return np.linalg.norm(oerr)

    def oerr_rotmat(self, mode="diff"):
        o_FeeDes = self.B_X_EEdes.rotation
        o_FeeAct = self.data.oMf[self.ee_id].rotation
        if mode == "diff":
            oerr = o_FeeDes - o_FeeAct
        elif mode == "trans":
            oerr = o_FeeDes @ o_FeeAct.transpose()
        else:
            raise ValueError("Invalid mode")
        return np.linalg.norm(oerr)

    def oerr_quat(self):
        o_FeeDes = R.from_matrix(self.B_X_EEdes.rotation).as_quat()
        o_FeeAct = R.from_matrix(self.data.oMf[self.ee_id].rotation).as_quat()
        o_FeeAct_conj = o_FeeAct
        alpha = np.abs(np.dot(o_FeeDes, o_FeeAct_conj))
        oerr = 0 if alpha >= 1 else 2 * np.arccos(alpha)
        return np.linalg.norm(oerr)

    def compute_residual_EE_log6(self):
        B_X_EEact = self.data.oMf[self.ee_id]
        res_ee = pin.log6(self.B_X_EEdes.act(B_X_EEact.inverse()))
        err_ee = np.linalg.norm(res_ee)
        return res_ee, err_ee

    def compute_residual_RCM(self):
        B_H_Fprercm = self.data.oMf[self.pre_rcm_id]
        B_H_Fpostrcm = self.data.oMf[self.post_rcm_id]

        ps = B_H_Fpostrcm.translation - B_H_Fprercm.translation
        pr = self.B_X_IP.translation - B_H_Fprercm.translation
        ps_hat = ps / np.linalg.norm(ps)

        B_p_Frcm = B_H_Fprercm.translation + np.transpose(pr) @ np.outer(ps_hat, ps_hat)

        pe = self.B_X_IP.translation - B_p_Frcm
        res_rcm = -np.dot(pe, pe)

        err_rcm = np.linalg.norm(pe)

        return res_rcm, ps, pr, pe, err_rcm

    def compute_Jacobian_EE(self, q_it):
        B_Jb_EE = pin.computeFrameJacobian(
            self.model, self.data, q_it, self.ee_id, pin.WORLD
        )
        return B_Jb_EE

    def compute_Jacobian_RCM(self, q_it, ps, pr, pe):
        ps_hat = ps / np.linalg.norm(ps)
        pe_hat = pe / np.linalg.norm(pe)

        B_Jb_Fprercm = pin.computeFrameJacobian(
            self.model, self.data, q_it, self.pre_rcm_id, pin.LOCAL_WORLD_ALIGNED
        )
        B_Jb_Fpostrcm = pin.computeFrameJacobian(
            self.model, self.data, q_it, self.post_rcm_id, pin.LOCAL_WORLD_ALIGNED
        )

        Jb_ps_hat = (
            (1 / np.linalg.norm(ps))
            * (np.eye(3) - (1 / np.dot(ps, ps)) * np.outer(ps, np.transpose(ps)))
            @ (B_Jb_Fpostrcm - B_Jb_Fprercm)[:3, :]
        )

        B_Jb_RCM = (
            -2
            * np.transpose(pe)
            @ (
                (np.eye(3) - np.outer(ps_hat, np.transpose(ps_hat)))
                @ B_Jb_Fprercm[:3, :]
                + (
                    np.outer(ps_hat, np.transpose(pr))
                    + np.dot(np.transpose(pr), ps_hat) * np.eye(3)
                )
                @ Jb_ps_hat
            )
        )

        B_Jb_RCM = B_Jb_RCM.reshape(self.nq, 1)

        return B_Jb_RCM
