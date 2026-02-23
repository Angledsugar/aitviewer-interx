"""
Body-attached camera system for SMPLX visualization.

Computes per-frame camera positions and targets from joint positions,
then creates aitviewer PinholeCamera objects.

Camera types:
  - Head Camera:  1st-person view from between the eyes, looking forward (face direction).
  - Hand Camera:  View from the wrist joint, looking along the forearm direction.
  - Top Camera:   Bird's-eye view looking down at the scene center (one shared camera).
"""

from typing import Dict, Tuple

import numpy as np
import scipy.ndimage


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vectors along the last axis."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return v / norm


def _smooth_trajectory(positions: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Smooth a trajectory with a Gaussian filter."""
    if sigma > 0 and positions.shape[0] > 1:
        return scipy.ndimage.gaussian_filter1d(positions, sigma, axis=0, mode="nearest")
    return positions


def compute_head_camera(
    joints: np.ndarray,
    jm: dict,
    offset_forward: float = 0.10,
    smooth_sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute head-mounted camera positioned between the eyes.

    The camera is placed at the head joint, offset forward to sit between
    the eyes, and looks in the direction the person is facing.

    Args:
        joints: Joint positions (F, J, 3).
        jm: Joint index mapping dict.
        offset_forward: Forward offset from head joint toward face (between eyes).
        smooth_sigma: Gaussian smoothing sigma for trajectory.

    Returns:
        (positions, targets) each of shape (F, 3).
    """
    head = joints[:, jm["head"]]
    neck = joints[:, jm["neck"]]
    l_shoulder = joints[:, jm["left_shoulder"]]
    r_shoulder = joints[:, jm["right_shoulder"]]

    # Spine direction (up).
    spine_up = _normalize(head - neck)

    # Shoulder direction (right).
    shoulder_right = _normalize(r_shoulder - l_shoulder)

    # Forward direction = up x right (face direction in Y-up system).
    # cross(up, right) points forward (out of the face).
    forward = _normalize(np.cross(spine_up, shoulder_right))

    # Camera position: at head, offset forward to sit between the eyes.
    positions = head + forward * offset_forward

    # Camera target: looking forward from the face.
    targets = positions + forward * 1.0

    positions = _smooth_trajectory(positions, smooth_sigma)
    targets = _smooth_trajectory(targets, smooth_sigma)

    return positions, targets


def compute_hand_camera(
    joints: np.ndarray,
    jm: dict,
    hand: str = "right",
    offset_forward: float = 0.05,
    smooth_sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute hand-mounted camera looking along the forearm.

    Args:
        joints: Joint positions (F, J, 3).
        jm: Joint index mapping dict.
        hand: 'left' or 'right'.
        offset_forward: Forward offset from wrist along forearm direction.
        smooth_sigma: Gaussian smoothing sigma for trajectory.

    Returns:
        (positions, targets) each of shape (F, 3).
    """
    if hand == "right":
        wrist = joints[:, jm["right_wrist"]]
        elbow = joints[:, jm["right_elbow"]]
    else:
        wrist = joints[:, jm["left_wrist"]]
        elbow = joints[:, jm["left_elbow"]]

    # Forward direction: from elbow to wrist.
    forearm_dir = _normalize(wrist - elbow)

    # Camera position: at wrist, slightly forward.
    positions = wrist + forearm_dir * offset_forward

    # Camera target: further along the forearm direction.
    targets = positions + forearm_dir * 1.0

    positions = _smooth_trajectory(positions, smooth_sigma)
    targets = _smooth_trajectory(targets, smooth_sigma)

    return positions, targets


def compute_top_camera(
    n_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return fixed top-view camera position and target, broadcast across all frames.

    Uses hardcoded values for a stable bird's-eye view of the scene.

    Args:
        n_frames: Number of frames to broadcast across.

    Returns:
        (positions, targets) each of shape (F, 3).
    """
    positions = np.tile(np.array([0.018, 2.972, 3.818]), (n_frames, 1))
    targets = np.tile(np.array([-0.052, 0.835, 0.580]), (n_frames, 1))
    return positions, targets


def _create_person_cameras(
    joints: np.ndarray,
    person_label: str,
    jm: dict,
    viewer=None,
    width: int = 640,
    height: int = 480,
    head_fov: float = 90.0,
    hand_fov: float = 90.0,
    smooth_sigma: float = 2.0,
) -> Dict[str, "PinholeCamera"]:
    """Create head and hand cameras attached to a single person's body."""
    from aitviewer.scene.camera import PinholeCamera

    label = person_label.upper()
    cameras = {}

    # Head camera (1st person view, between the eyes).
    pos, tar = compute_head_camera(joints, jm, smooth_sigma=smooth_sigma)
    cameras[f"{label}_head_cam"] = PinholeCamera(
        pos, tar, width, height, fov=head_fov, viewer=viewer,
        name=f"{label} Head Camera",
    )

    # Right hand camera.
    pos, tar = compute_hand_camera(joints, jm, hand="right", smooth_sigma=smooth_sigma)
    cameras[f"{label}_right_hand_cam"] = PinholeCamera(
        pos, tar, width, height, fov=hand_fov, viewer=viewer,
        name=f"{label} Right Hand Camera",
    )

    # Left hand camera.
    pos, tar = compute_hand_camera(joints, jm, hand="left", smooth_sigma=smooth_sigma)
    cameras[f"{label}_left_hand_cam"] = PinholeCamera(
        pos, tar, width, height, fov=hand_fov, viewer=viewer,
        name=f"{label} Left Hand Camera",
    )

    return cameras


def create_body_cameras(
    joints_p1: np.ndarray,
    joints_p2: np.ndarray,
    joint_map: dict,
    viewer=None,
    width: int = 640,
    height: int = 480,
    head_fov: float = 90.0,
    hand_fov: float = 90.0,
    top_fov: float = 45.0,
    smooth_sigma: float = 2.0,
) -> Dict[str, "PinholeCamera"]:
    """Create body-attached PinholeCamera objects for both persons.

    Creates head, left hand, right hand cameras per person, plus one
    shared top-down camera.

    Args:
        joints_p1: Person 1 joints (F, J, 3).
        joints_p2: Person 2 joints (F, J, 3).
        joint_map: Joint index mapping (SMPLX_JOINT_MAP or OPTITRACK_JOINT_MAP).
        viewer: aitviewer Viewer instance (for interactive camera viewing).
        width: Camera image width.
        height: Camera image height.
        head_fov: Head camera FOV.
        hand_fov: Hand camera FOV.
        top_fov: Top camera FOV.
        smooth_sigma: Smoothing sigma for trajectories.

    Returns:
        Dict mapping camera names to PinholeCamera objects.
    """
    from aitviewer.scene.camera import PinholeCamera

    cameras = {}

    # P1 cameras (head + hands).
    cameras.update(_create_person_cameras(
        joints_p1, "P1", jm=joint_map,
        viewer=viewer, width=width, height=height,
        head_fov=head_fov, hand_fov=hand_fov,
        smooth_sigma=smooth_sigma,
    ))

    # P2 cameras (head + hands).
    cameras.update(_create_person_cameras(
        joints_p2, "P2", jm=joint_map,
        viewer=viewer, width=width, height=height,
        head_fov=head_fov, hand_fov=hand_fov,
        smooth_sigma=smooth_sigma,
    ))

    # Shared top-down camera (fixed position).
    n_frames = joints_p1.shape[0]
    pos, tar = compute_top_camera(n_frames)
    cameras["top_cam"] = PinholeCamera(
        pos, tar, width, height, fov=top_fov, viewer=viewer,
        name="Top View Camera",
    )

    return cameras
