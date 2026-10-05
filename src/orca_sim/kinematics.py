"""Small site-based inverse-kinematics helper shared by task environments and
scripted demonstration generators.

This is a plain damped-least-squares solver operating on a MuJoCo ``MjData``'s
``qpos`` for a chosen subset of degrees of freedom. It is intentionally
dependency-free (no external IK library) since the arms used here (a single
7-DoF Franka Panda) are small enough that a few dozen Newton-style iterations
converge in well under a millisecond.
"""

from __future__ import annotations

import mujoco
import numpy as np


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def axis_angle_increment(rot_vec: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector -> quaternion, for incrementally nudging an
    orientation target (used by delta-pose action spaces)."""
    angle = float(np.linalg.norm(rot_vec))
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rot_vec / angle
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q1, q2)
    return out


class SiteIK:
    """Damped-least-squares IK for a single MuJoCo site, solving over a
    contiguous slice of joint-space degrees of freedom (e.g. the first 7 dofs
    of a Panda arm mounted at the start of the kinematic chain).

    Parameters
    ----------
    model, data:
        The live MjModel / MjData. ``data.qpos`` is read and written in place
        by :meth:`step` and :meth:`solve`.
    site_name:
        Name of the site to drive (e.g. a grasp-point site on the hand).
    dof_ids:
        Indices into ``qpos`` / ``qvel`` (they coincide for hinge/slide
        joints, which is all the Panda uses) that the solver is allowed to
        move. Must be a chain starting at the robot's base joint.
    joint_range:
        (n, 2) array of [lo, hi] limits for the same dofs, used to clip each
        iteration so the solver never proposes an infeasible pose.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_name: str,
        dof_ids: np.ndarray,
        joint_range: np.ndarray,
        *,
        damping: float = 1e-4,
        nullspace_gain: float = 0.05,
        home_qpos: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.data = data
        self.site_id = model.site(site_name).id
        self.dof_ids = np.asarray(dof_ids, dtype=np.int64)
        self.joint_range = np.asarray(joint_range, dtype=np.float64)
        self.damping = damping
        self.nullspace_gain = nullspace_gain
        self.home_qpos = (
            None if home_qpos is None else np.asarray(home_qpos, dtype=np.float64)
        )
        self._ndof = len(self.dof_ids)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Current (position, 3x3 rotation matrix) of the driven site."""
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        rot = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, rot

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        *,
        q0: np.ndarray | None = None,
        iters: int = 150,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
    ) -> tuple[np.ndarray, float, float]:
        """Solve for the dof values that drive the site to the target pose.

        Does not mutate ``self.data.qpos`` outside the solved dofs, and does
        not step physics. Returns (qpos_values, pos_error, rot_error) for the
        solved dofs; the caller decides whether to accept/apply them.
        """
        model, data = self.model, self.data
        q = (
            data.qpos[self.dof_ids].copy()
            if q0 is None
            else np.asarray(q0, dtype=np.float64).copy()
        )
        saved = data.qpos[self.dof_ids].copy()
        pos_err = rot_err = 0.0
        for _ in range(iters):
            data.qpos[self.dof_ids] = q
            mujoco.mj_kinematics(model, data)
            # mj_jacSite reads d.cdof, which mj_kinematics does not populate;
            # without this the Jacobian silently uses whatever cdof happened
            # to be left over from the last unrelated mj_forward() call.
            mujoco.mj_comPos(model, data)
            cur_pos = data.site_xpos[self.site_id]
            cur_rot = data.site_xmat[self.site_id].reshape(3, 3)
            pos_vec = target_pos - cur_pos
            rot_vec = 0.5 * (
                np.cross(cur_rot[:, 0], target_rot[:, 0])
                + np.cross(cur_rot[:, 1], target_rot[:, 1])
                + np.cross(cur_rot[:, 2], target_rot[:, 2])
            )
            pos_err = float(np.linalg.norm(pos_vec))
            rot_err = float(np.linalg.norm(rot_vec))
            if pos_err < pos_tol and rot_err < rot_tol:
                break
            mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self.site_id)
            jac = np.vstack([self._jacp, self._jacr])[:, self.dof_ids]
            err = np.concatenate([pos_vec, rot_vec])
            lam = self.damping * np.eye(6)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + lam, err)
            if self.home_qpos is not None:
                nullspace = np.eye(self._ndof) - np.linalg.pinv(jac) @ jac
                dq += nullspace @ (
                    self.nullspace_gain * (self.home_qpos - q)
                )
            q = np.clip(q + dq, self.joint_range[:, 0], self.joint_range[:, 1])
        data.qpos[self.dof_ids] = saved  # caller applies the result explicitly
        return q, pos_err, rot_err
