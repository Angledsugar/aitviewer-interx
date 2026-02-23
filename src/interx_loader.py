"""
Inter-X dataset loader for SMPLX visualization.

Supports two data formats:
  1. NPZ files with SMPLX parameters (pose_body, root_orient, betas, trans, etc.)
     -> Full mesh + skeleton visualization via SMPLSequence
  2. NPY files with Optitrack joint positions (T, 64, 3)
     -> Skeleton-only visualization via Skeletons renderable

NPY Optitrack joint layout (64 raw markers, 61 after selection):
  After SELECTED_JOINTS filtering, the indices used for skeleton connectivity
  and camera placement follow the OPTITRACK_LIMBS definition from Inter-X.

  Key body landmarks (indices in the *selected* 61-joint array):
    0: hip center,  1-4: right leg,  5-8: left leg (was 6-9 in raw, shifted)
    9: spine (was 10 in raw),  10: spine1 (was 11)
    11-13: neck/head chain (was 12-14)
    14: right shoulder (was 14),  35: left shoulder (was 35 in selected)
    22: right wrist,  46: left wrist
    38: left hand root,  14: right hand root region
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Optitrack skeleton (from Inter-X joint_viewer_tool/data_viewer.py)
# ──────────────────────────────────────────────────────────────────────

# Joints to select from raw 64-marker data (skip indices 5 and 10).
SELECTED_JOINTS = np.concatenate([range(0, 5), range(6, 10), range(11, 63)])

# Skeleton connectivity in the *selected* joint space.
OPTITRACK_LIMBS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10],
    [10, 11], [11, 12], [12, 13], [13, 14],
        [14, 15], [15, 16], [16, 17], [17, 18],
        [14, 19], [19, 20], [20, 21], [21, 22],
        [14, 23], [23, 24], [24, 25], [25, 26],
        [14, 27], [27, 28], [28, 29], [29, 30],
        [14, 31], [31, 32], [32, 33], [33, 34],
    [10, 35], [35, 36], [36, 37], [37, 38],
        [38, 39], [39, 40], [40, 41], [41, 42],
        [38, 43], [43, 44], [44, 45], [45, 46],
        [38, 47], [47, 48], [48, 49], [49, 50],
        [38, 51], [51, 52], [52, 53], [53, 54],
        [38, 55], [55, 56], [56, 57], [57, 58],
    [10, 59], [59, 60],
]

# ──────────────────────────────────────────────────────────────────────
# Joint index mappings for camera placement.
# Two systems depending on data format:
#   - SMPLX (NPZ): 22 body joints from FK
#   - Optitrack (NPY): 61 selected joints
# ──────────────────────────────────────────────────────────────────────

# Optitrack joint indices (NPY, 61 selected joints).
OPTITRACK_JOINT_MAP = {
    "pelvis": 0,
    "spine": 10,
    "neck": 59,
    "head": 60,
    "left_shoulder": 35,
    "right_shoulder": 11,
    "left_elbow": 36,
    "right_elbow": 12,
    "left_wrist": 37,
    "right_wrist": 13,
}

# SMPLX joint indices (NPZ, 22 body joints from FK).
SMPLX_JOINT_MAP = {
    "pelvis": 0,
    "spine": 9,       # spine3
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

# Legacy constants (Optitrack) for backward compat.
JOINT_PELVIS = 0
JOINT_SPINE3 = 10
JOINT_NECK = 59
JOINT_HEAD = 60
JOINT_RIGHT_SHOULDER = 11
JOINT_LEFT_SHOULDER = 35
JOINT_RIGHT_ELBOW = 12
JOINT_LEFT_ELBOW = 36
JOINT_RIGHT_WRIST = 13
JOINT_LEFT_WRIST = 37


@dataclass
class InterXSequence:
    """Container for a loaded Inter-X sequence."""
    name: str
    # Joint positions (F, J, 3) - always available.
    joints_p1: np.ndarray
    joints_p2: np.ndarray
    n_frames: int
    fps: int
    # Joint index mapping for camera placement (keys: pelvis, head, neck, etc.)
    joint_map: dict = None
    # Text descriptions (if available).
    text: Optional[str] = None
    # SMPLX parameters (if loaded from NPZ).
    smplx_params_p1: Optional[dict] = None
    smplx_params_p2: Optional[dict] = None


def list_sequences(data_dir: str) -> List[str]:
    """List available sequence names in the dataset directory.

    Searches both motions/ and skeletons/ subdirectories.
    """
    sequences = set()
    for subdir in ["motions", "skeletons"]:
        folder = os.path.join(data_dir, subdir)
        if os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                if not name.startswith("."):
                    sequences.add(name)
    return sorted(sequences)


def load_text(data_dir: str, sequence_name: str) -> Optional[str]:
    """Load text description for a sequence."""
    text_path = os.path.join(data_dir, "texts", f"{sequence_name}.txt")
    if os.path.exists(text_path):
        with open(text_path, "r") as f:
            return f.read().strip()
    return None


def _detect_data_format(data_dir: str, sequence_name: str) -> str:
    """Detect whether SMPLX parameters (NPZ) or joint positions (NPY) are available.

    Returns 'npz' if NPZ SMPLX parameter files exist, 'npy' otherwise.
    """
    npz_path = os.path.join(data_dir, "motions", sequence_name, "P1.npz")
    if os.path.exists(npz_path):
        return "npz"
    return "npy"


def load_sequence(data_dir: str, sequence_name: str, fps: int = 30) -> InterXSequence:
    """Load an Inter-X sequence, auto-detecting the data format.

    Args:
        data_dir: Root dataset directory (e.g., datasets/interx/).
        sequence_name: Sequence identifier (e.g., G001T000A000R000).
        fps: Target FPS for the sequence.

    Returns:
        InterXSequence with joint positions and optionally SMPLX parameters.
    """
    fmt = _detect_data_format(data_dir, sequence_name)
    if fmt == "npz":
        return _load_from_npz(data_dir, sequence_name, fps)
    else:
        return _load_from_npy(data_dir, sequence_name, fps)


def _load_from_npz(data_dir: str, sequence_name: str, fps: int) -> InterXSequence:
    """Load from NPZ files containing SMPLX parameters."""
    motions_dir = os.path.join(data_dir, "motions", sequence_name)

    params_p1 = dict(np.load(os.path.join(motions_dir, "P1.npz"), allow_pickle=True))
    params_p2 = dict(np.load(os.path.join(motions_dir, "P2.npz"), allow_pickle=True))

    nf = params_p1["pose_body"].shape[0]

    # Downsample from original FPS (120) to target FPS if needed.
    original_fps = 120
    downsample = max(1, original_fps // fps)

    for key in params_p1:
        if isinstance(params_p1[key], np.ndarray) and params_p1[key].ndim >= 1:
            if params_p1[key].shape[0] == nf:
                params_p1[key] = params_p1[key][::downsample]
                params_p2[key] = params_p2[key][::downsample]

    text = load_text(data_dir, sequence_name)

    # We'll compute joint positions when creating SMPLSequence (lazy).
    # For now, store None; joints will be filled in by create_smpl_sequences().
    return InterXSequence(
        name=sequence_name,
        joints_p1=None,  # Will be filled after FK.
        joints_p2=None,
        n_frames=params_p1["pose_body"].shape[0],
        fps=fps,
        joint_map=SMPLX_JOINT_MAP,
        text=text,
        smplx_params_p1=params_p1,
        smplx_params_p2=params_p2,
    )


def _load_from_npy(data_dir: str, sequence_name: str, fps: int) -> InterXSequence:
    """Load from NPY files containing Optitrack joint positions."""
    # Try motions/ first, then skeletons/.
    for subdir in ["motions", "skeletons"]:
        p1_path = os.path.join(data_dir, subdir, sequence_name, "P1.npy")
        p2_path = os.path.join(data_dir, subdir, sequence_name, "P2.npy")
        if os.path.exists(p1_path) and os.path.exists(p2_path):
            break
    else:
        raise FileNotFoundError(
            f"No data found for sequence '{sequence_name}' in {data_dir}"
        )

    joints_p1 = np.load(p1_path)  # (T, 64, 3)
    joints_p2 = np.load(p2_path)

    # Select the same joints as Inter-X viewer (skip raw indices 5 and 10).
    joints_p1 = joints_p1[:, SELECTED_JOINTS]  # (T, 61, 3)
    joints_p2 = joints_p2[:, SELECTED_JOINTS]

    # Downsample from 120 fps to target fps.
    original_fps = 120
    downsample = max(1, original_fps // fps)
    joints_p1 = joints_p1[::downsample]
    joints_p2 = joints_p2[::downsample]

    text = load_text(data_dir, sequence_name)

    return InterXSequence(
        name=sequence_name,
        joints_p1=joints_p1,
        joints_p2=joints_p2,
        n_frames=joints_p1.shape[0],
        fps=fps,
        joint_map=OPTITRACK_JOINT_MAP,
        text=text,
    )


def create_smpl_sequences(seq: InterXSequence, device=None):
    """Create aitviewer SMPLSequence objects from SMPLX parameters.

    Requires NPZ data with SMPLX parameters. Returns (smpl_seq_p1, smpl_seq_p2).
    Also fills in seq.joints_p1/p2 from the FK computation.

    Args:
        seq: InterXSequence with smplx_params_p1/p2 set.
        device: PyTorch device (defaults to aitviewer CONFIG.device).
    """
    from aitviewer.configuration import CONFIG as C
    from aitviewer.models.smpl import SMPLLayer
    from aitviewer.renderables.smpl import SMPLSequence

    if device is None:
        device = C.device

    if seq.smplx_params_p1 is None:
        raise ValueError("NPZ SMPLX parameters required. Current data is NPY (joint positions only).")

    sequences = []
    colors = [
        (0.11, 0.53, 0.8, 1.0),   # P1: blue
        (1.0, 0.27, 0.0, 1.0),    # P2: red/orange
    ]

    for i, (params, color) in enumerate(zip(
        [seq.smplx_params_p1, seq.smplx_params_p2], colors
    )):
        nf = params["pose_body"].shape[0]
        gender = str(params.get("gender", "neutral"))

        smplx_layer = SMPLLayer(
            model_type="smplx",
            gender=gender,
            num_betas=10,
            device=device,
        )

        smpl_seq = SMPLSequence(
            poses_body=params["pose_body"].reshape(nf, -1),
            smpl_layer=smplx_layer,
            poses_root=params["root_orient"],
            betas=params["betas"],
            trans=params["trans"],
            poses_left_hand=params["pose_lhand"].reshape(nf, -1),
            poses_right_hand=params["pose_rhand"].reshape(nf, -1),
            device=device,
            color=color,
            name=f"P{i+1}",
        )
        sequences.append(smpl_seq)

    # Fill in joint positions from FK results.
    seq.joints_p1 = sequences[0].joints  # (F, J, 3)
    seq.joints_p2 = sequences[1].joints

    return sequences[0], sequences[1]


def create_skeleton_nodes(seq: InterXSequence):
    """Create aitviewer Skeletons nodes from Optitrack joint data (NPY).

    Uses the OPTITRACK_LIMBS connectivity matching Inter-X's data_viewer.py.
    Returns (skeleton_p1, skeleton_p2).
    """
    from aitviewer.renderables.skeletons import Skeletons

    if seq.joints_p1 is None:
        raise ValueError("Joint position data not available.")

    skel_p1 = Skeletons(
        seq.joints_p1,
        OPTITRACK_LIMBS,
        color=(0.11, 0.53, 0.8, 1.0),
        name="P1 Skeleton",
    )

    skel_p2 = Skeletons(
        seq.joints_p2,
        OPTITRACK_LIMBS,
        color=(1.0, 0.27, 0.0, 1.0),
        name="P2 Skeleton",
    )

    return skel_p1, skel_p2
