#!/usr/bin/env python3
"""
image_exporter.py

Inter-X SMPL-X (motions/<clip_id>/P1.npz,P2.npz) -> 7-camera RGB frame export (LeRobot-like)

Output layout (default):
  exports/<clip_id>/observations/images/<cam_name>/<frame:06d>.png

Cameras (7):
  p1_wrist_l, p1_wrist_r, p1_head,
  p2_wrist_l, p2_wrist_r, p2_head,
  topdown_all

Key points
- Headless rendering via AITViewer HeadlessRenderer.
- Two modes:
  1) Fixed cameras (default): 7 static cameras looking at scene center.
  2) Joint-follow cameras (--follow-joints): per-frame positions follow wrist/head joints.
- Works without joint names:
  If SMPL-X model does not expose JOINT_NAMES/joint_names, falls back to indices:
    root=0, left_wrist=20, right_wrist=21, head=15 (override via CLI)
- Head camera "forward" can be improved:
  - Uses root_orient (axis-angle) to orient p1_head/p2_head forward (approx gaze).
  - Wrist cameras still use a simple forward unless you extend to elbow->wrist direction.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as SciR

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence

# Headless renderer (preferred)
try:
    from aitviewer.headless import HeadlessRenderer
except Exception as e:
    raise RuntimeError(
        "HeadlessRenderer import failed. If you're on a headless machine, install Xvfb and run with xvfb-run."
    ) from e

# Cameras
try:
    from aitviewer.scene.camera import PinholeCamera
except Exception as e:
    raise RuntimeError("Cannot import PinholeCamera from aitviewer.scene.camera") from e


# -------------------------
# Utilities
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def np_load_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return dict(np.load(str(path), allow_pickle=True))


def to_numpy(x):
    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return x


def find_seq_joints(seq) -> Optional[np.ndarray]:
    """
    Best-effort extraction of joints (T,J,3) from SMPLSequence.
    Different AITViewer versions use different attribute names.
    """
    candidates = [
        "joints",
        "_joints",
        "J",
        "Jtr",
        "joints_world",
        "joints_global",
        "joint_positions",
    ]
    for attr in candidates:
        if hasattr(seq, attr):
            j = getattr(seq, attr)
            j = to_numpy(j)
            if isinstance(j, np.ndarray) and j.ndim == 3 and j.shape[-1] == 3:
                return j
    return None


def get_joint_names_from_layer(smpl_layer: SMPLLayer) -> Optional[list]:
    bm = getattr(smpl_layer, "bm", None)
    if bm is None:
        return None
    names = getattr(bm, "JOINT_NAMES", None)
    if names is None:
        names = getattr(bm, "joint_names", None)
    if names is None:
        return None
    return list(names)


def resolve_joint_index(joint_names: list, candidates: list) -> Optional[int]:
    name2idx = {n: i for i, n in enumerate(joint_names)}
    for c in candidates:
        if c in name2idx:
            return name2idx[c]
    return None


def normalize_rotvec_array(x, T: int) -> np.ndarray:
    """
    Normalize root_orient-like array to shape (T,3) float32.
    Handles shapes: (T,3), (T,1,3), (T,3,1), (3,), object arrays, etc.
    """
    x = np.asarray(x)

    # object array -> numeric
    if x.dtype == object:
        x = np.array([np.asarray(v, dtype=np.float32).reshape(-1) for v in x], dtype=np.float32)

    x = x.astype(np.float32)

    if x.ndim == 1:
        # could be (3,) or (T*3,)
        if x.size == 3:
            x = np.tile(x.reshape(1, 3), (T, 1))
        else:
            x = x.reshape(-1, 3)

    if x.ndim == 2 and x.shape[1] == 3:
        pass
    elif x.ndim == 3:
        x = np.squeeze(x)
        if x.ndim == 1 and x.size == 3:
            x = np.tile(x.reshape(1, 3), (T, 1))
        elif x.ndim == 2 and x.shape[1] == 3:
            pass
        else:
            x = x.reshape(-1, 3)
    else:
        x = x.reshape(-1, 3)

    # match length T
    if x.shape[0] < T:
        pad = np.repeat(x[-1:].copy(), T - x.shape[0], axis=0)
        x = np.concatenate([x, pad], axis=0)
    elif x.shape[0] > T:
        x = x[:T]

    return x


def update_camera_pose(cam: PinholeCamera, pos: np.ndarray, tgt: np.ndarray) -> None:
    """
    Update camera pose for current frame; robust across AITViewer versions.
    Some camera attributes are read-only (e.g., center), so we set only writable fields.
    """
    pos = np.asarray(pos, dtype=np.float32)
    tgt = np.asarray(tgt, dtype=np.float32)

    # Preferred: look_at if available (do not early-return; still try setting fields)
    if hasattr(cam, "look_at") and callable(getattr(cam, "look_at")):
        try:
            cam.look_at(pos, tgt)
        except Exception:
            pass

    # Best-effort writable attributes
    for attr, val in (("position", pos), ("target", tgt), ("eye", pos)):
        try:
            if hasattr(cam, attr):
                setattr(cam, attr, val)
        except Exception:
            pass

    # Do NOT set cam.center (often read-only)


def safe_get_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return None


# -------------------------
# SMPL-X loading (matches your data_viewer.py)
# -------------------------
def build_smplx_sequence(
    npz_params: dict,
    color: Tuple[float, float, float, float],
    device: str,
):
    nf = npz_params["pose_body"].shape[0]

    betas = npz_params["betas"]
    poses_root = npz_params["root_orient"]
    poses_body = npz_params["pose_body"].reshape(nf, -1)
    poses_lhand = npz_params["pose_lhand"].reshape(nf, -1)
    poses_rhand = npz_params["pose_rhand"].reshape(nf, -1)
    transl = npz_params["trans"]
    gender = str(npz_params["gender"])

    smplx_layer = SMPLLayer(model_type="smplx", gender=gender, num_betas=10, device=device)

    seq = SMPLSequence(
        poses_body=poses_body,
        smpl_layer=smplx_layer,
        poses_root=poses_root,
        betas=betas,
        trans=transl,
        poses_left_hand=poses_lhand,
        poses_right_hand=poses_rhand,
        device=device,
        color=color,
    )
    return seq, smplx_layer, nf


# -------------------------
# Camera builders
# -------------------------
def build_fixed_7_cameras(scene_center: np.ndarray, w: int, h: int) -> Dict[str, PinholeCamera]:
    c = np.asarray(scene_center, dtype=np.float32)

    def cam(pos, tgt, fov):
        return PinholeCamera(np.asarray(pos, np.float32), np.asarray(tgt, np.float32), w, h, fov=float(fov))

    return {
        "p1_wrist_l": cam([+1.0, +0.35, 1.15], c, 80),
        "p1_wrist_r": cam([+1.0, -0.35, 1.15], c, 80),
        "p1_head":    cam([+1.2,  0.00, 1.60], c, 70),

        "p2_wrist_l": cam([-1.0, +0.35, 1.15], c, 80),
        "p2_wrist_r": cam([-1.0, -0.35, 1.15], c, 80),
        "p2_head":    cam([-1.2,  0.00, 1.60], c, 80),

        "topdown_all": cam([0.0,  3.50, 0.0], c, 65),
    }


def build_joint_follow_specs_from_names(joint_names: list) -> Dict[str, Dict]:
    # Candidate names (varies across SMPL-X releases)
    cand_left_wrist = ["left_wrist", "lwrist", "L_Wrist", "Lwrist"]
    cand_right_wrist = ["right_wrist", "rwrist", "R_Wrist", "Rwrist"]
    cand_head = ["head", "Head", "neck", "Neck"]
    cand_root = ["pelvis", "Pelvis", "hips", "Hips", "root", "Root"]
    cand_left_elbow = ["left_elbow", "lelbow", "L_Elbow", "Lelbow"]
    cand_right_elbow = ["right_elbow", "relbow", "R_Elbow", "Relbow"]
    cand_left_shoulder = ["left_shoulder", "lshoulder", "L_Shoulder", "Lshoulder"]
    cand_right_shoulder = ["right_shoulder", "rshoulder", "R_Shoulder", "Rshoulder"]

    li = resolve_joint_index(joint_names, cand_left_wrist)
    ri = resolve_joint_index(joint_names, cand_right_wrist)
    hi = resolve_joint_index(joint_names, cand_head)
    rooti = resolve_joint_index(joint_names, cand_root)
    lei = resolve_joint_index(joint_names, cand_left_elbow)
    rei = resolve_joint_index(joint_names, cand_right_elbow)
    lshi = resolve_joint_index(joint_names, cand_left_shoulder)
    rshi = resolve_joint_index(joint_names, cand_right_shoulder)

    if None in (li, ri, hi, rooti):
        raise RuntimeError(f"Cannot resolve joints by name: left={li}, right={ri}, head/neck={hi}, root={rooti}")

    return build_joint_follow_specs_from_indices(
        idx_root=int(rooti),
        idx_left_wrist=int(li),
        idx_right_wrist=int(ri),
        idx_head=int(hi),
        idx_left_elbow=int(lei) if lei is not None else 18,
        idx_right_elbow=int(rei) if rei is not None else 19,
        idx_left_shoulder=int(lshi) if lshi is not None else 16,
        idx_right_shoulder=int(rshi) if rshi is not None else 17,
    )


def build_joint_follow_specs_from_indices(
    idx_root: int,
    idx_left_wrist: int,
    idx_right_wrist: int,
    idx_head: int,
    idx_left_elbow: int = 18,
    idx_right_elbow: int = 19,
    idx_left_shoulder: int = 16,
    idx_right_shoulder: int = 17,
) -> Dict[str, Dict]:
    off_wrist = np.array([0.03, 0.0, 0.02], dtype=np.float32)
    off_head = np.array([0.00, 0.0, 0.08], dtype=np.float32)  # eye-height-ish

    return {
        "p1_wrist_l": {"who": "p1", "j": idx_left_wrist,  "j_elbow": idx_left_elbow,  "off": off_wrist, "fov": 80},
        "p1_wrist_r": {"who": "p1", "j": idx_right_wrist, "j_elbow": idx_right_elbow, "off": off_wrist, "fov": 80},
        "p1_head":    {"who": "p1", "j": idx_head, "j_lsh": idx_left_shoulder, "j_rsh": idx_right_shoulder, "off": off_head, "fov": 70},

        "p2_wrist_l": {"who": "p2", "j": idx_left_wrist,  "j_elbow": idx_left_elbow,  "off": off_wrist, "fov": 80},
        "p2_wrist_r": {"who": "p2", "j": idx_right_wrist, "j_elbow": idx_right_elbow, "off": off_wrist, "fov": 80},
        "p2_head":    {"who": "p2", "j": idx_head, "j_lsh": idx_left_shoulder, "j_rsh": idx_right_shoulder, "off": off_head, "fov": 70},

        # topdown: 두 사람 root 중간점의 Z=3.5 위에서 -Z 방향으로 내려다봄 (Z-up 좌표계)
        "topdown_all": {"who": "both_root", "j": idx_root,
                        "off": np.array([0.0, 0.0, 3.5], dtype=np.float32),
                        "fov": 65},
    }


def init_cameras_from_specs(specs: Dict[str, Dict], w: int, h: int) -> Dict[str, PinholeCamera]:
    cams = {}
    dummy_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    dummy_tgt = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    for name, sp in specs.items():
        cams[name] = PinholeCamera(dummy_pos.copy(), dummy_tgt.copy(), w, h, fov=float(sp["fov"]))
    return cams


# -------------------------
# Headless backend
# -------------------------
class HeadlessBackend:
    def __init__(self, w: int, h: int, fps: int):
        self.r = HeadlessRenderer(window_size=(w, h))
        self.r.scene.fps = fps
        self.r.playback_fps = fps
        self.w = w
        self.h = h
        self._scene_initialized = False

    def add(self, node):
        self.r.scene.add(node)

    def _ensure_init(self):
        """Initialize scene once (make renderables, set floor, etc.)."""
        if not self._scene_initialized:
            self.r._init_scene()
            self._scene_initialized = True

    def set_frame(self, t: int):
        self.r.scene.current_frame_id = t

    def set_camera(self, cam: PinholeCamera):
        # Some AITViewer versions render from viewport camera, others from scene.camera.
        self.r.viewports[0].camera = cam
        if hasattr(self.r.scene, "camera"):
            self.r.scene.camera = cam

    def export_frame(self, path: Path):
        ensure_dir(path.parent)
        self._ensure_init()
        self.r.export_frame(str(path))

    def render_current_frame(self) -> np.ndarray:
        """Render current frame and return as numpy array (H, W, 3)."""
        self._ensure_init()
        run_animations = self.r.run_animations
        self.r.run_animations = False
        self.r.render(0, 0, export=True)
        self.r.run_animations = run_animations
        img = self.r.get_current_frame_as_image()
        return np.array(img)

    def save_video(self, output_path: str, fps: int, quality: str = "medium"):
        """
        Export the full animation as a single video (fixed camera).
        Uses HeadlessRenderer.save_video() internally.
        """
        ensure_dir(Path(output_path).parent)
        self.r.save_video(
            video_dir=output_path,
            output_fps=fps,
            quality=quality,
            ensure_no_overwrite=False,
        )

    def close(self):
        try:
            self.r.close()
        except Exception:
            pass


# -------------------------
# Export procedure
# -------------------------
def export_clip(
    clip_id: str,
    motions_dir: Path,
    out_dir: Path,
    body_models_dir: Path,
    w: int,
    h: int,
    fps: int,
    every_n: int,
    max_frames: Optional[int],
    device: str,
    follow_joints: bool,
    idx_root: int,
    idx_left_wrist: int,
    idx_right_wrist: int,
    idx_head: int,
    idx_left_elbow: int,
    idx_right_elbow: int,
    idx_left_shoulder: int,
    idx_right_shoulder: int,
    debug_print_cam0: bool,
    export_video: bool = False,
    video_format: str = "mp4",
    video_quality: str = "medium",
) -> None:
    clip_dir = motions_dir / clip_id
    p1_path = clip_dir / "P1.npz"
    p2_path = clip_dir / "P2.npz"

    params1 = np_load_npz(p1_path)
    params2 = np_load_npz(p2_path)

    # Configure AITViewer (SMPL-X model path + device)
    C.update_conf({"smplx_models": str(body_models_dir)})
    C.update_conf({"device": device})

    seq1, layer1, nf1 = build_smplx_sequence(params1, color=(0.11, 0.53, 0.80, 1.0), device=device)
    seq2, layer2, nf2 = build_smplx_sequence(params2, color=(1.00, 0.27, 0.00, 1.0), device=device)

    # Determine T first
    T = min(nf1, nf2)
    if max_frames is not None:
        T = min(T, int(max_frames))
    if T <= 0:
        raise RuntimeError(f"No frames to export for clip={clip_id}")

    # Normalize root_orient AFTER nf1/nf2 exist (axis-angle rotvec)
    root1 = normalize_rotvec_array(params1["root_orient"], nf1)[:T]
    root2 = normalize_rotvec_array(params2["root_orient"], nf2)[:T]

    backend = HeadlessBackend(w, h, fps)
    backend.add(seq1)
    backend.add(seq2)

    joints1 = find_seq_joints(seq1)
    joints2 = find_seq_joints(seq2)

    specs = None

    if follow_joints:
        if joints1 is None or joints2 is None:
            print("[WARN] --follow-joints requested, but joints not accessible. Falling back to fixed cameras.")
            follow_joints = False
        else:
            joints1 = joints1[:T]
            joints2 = joints2[:T]
            joint_names = get_joint_names_from_layer(layer1)

            if joint_names is None:
                print("[WARN] joint names not accessible. Using fallback joint indices.")
                specs = build_joint_follow_specs_from_indices(
                    idx_root=idx_root,
                    idx_left_wrist=idx_left_wrist,
                    idx_right_wrist=idx_right_wrist,
                    idx_head=idx_head,
                    idx_left_elbow=idx_left_elbow,
                    idx_right_elbow=idx_right_elbow,
                    idx_left_shoulder=idx_left_shoulder,
                    idx_right_shoulder=idx_right_shoulder,
                )
            else:
                try:
                    specs = build_joint_follow_specs_from_names(joint_names)
                except Exception as e:
                    print(f"[WARN] joint-name mapping failed ({e}). Using fallback joint indices.")
                    specs = build_joint_follow_specs_from_indices(
                        idx_root=idx_root,
                        idx_left_wrist=idx_left_wrist,
                        idx_right_wrist=idx_right_wrist,
                        idx_head=idx_head,
                        idx_left_elbow=idx_left_elbow,
                        idx_right_elbow=idx_right_elbow,
                        idx_left_shoulder=idx_left_shoulder,
                        idx_right_shoulder=idx_right_shoulder,
                    )

    # If not follow joints, make a fixed spec table (also rendered via new camera objects each time)
    if not follow_joints:
        center = 0.5 * (np.asarray(params1["trans"][0]) + np.asarray(params2["trans"][0])).astype(np.float32)
        center[2] = max(center[2], 1.0)
        fixed = {
            "p1_wrist_l": (np.array([+1.0, +0.35, 1.15], np.float32), center, 80),
            "p1_wrist_r": (np.array([+1.0, -0.35, 1.15], np.float32), center, 80),
            "p1_head":    (np.array([+1.2,  0.00, 1.60], np.float32), center, 70),
            "p2_wrist_l": (np.array([-1.0, +0.35, 1.15], np.float32), center, 80),
            "p2_wrist_r": (np.array([-1.0, -0.35, 1.15], np.float32), center, 80),
            "p2_head":    (np.array([-1.2,  0.00, 1.60], np.float32), center, 70),
            "topdown_all":(np.array([0.0, 0.0, 3.5], np.float32),     center, 65),
        }
        cam_names = list(fixed.keys())
    else:
        cam_names = list(specs.keys())  # type: ignore

    out_root = out_dir / clip_id / "observations" / "images"
    if export_video:
        video_root = out_dir / clip_id / "observations" / "videos"
    for cam_name in cam_names:
        ensure_dir(out_root / cam_name)
        if export_video:
            ensure_dir(video_root)

    mode_str = "video" if export_video else "images"
    print(f"[INFO] Export clip={clip_id}, T={T}, cams={len(cam_names)}, every_n={every_n}, device={device}, follow={follow_joints}, mode={mode_str}")

    # Optional debug: print first-frame computed poses (computed, not camera object attrs)
    if debug_print_cam0:
        print("=== computed camera poses at frame 0 ===")

    # Initialize video writers (one per camera)
    video_writers = {}
    if export_video:
        import skvideo.io

        output_fps = fps // every_n if every_n > 1 else fps
        quality_to_crf = {"high": "23", "medium": "28", "low": "33"}
        for cam_name in cam_names:
            video_path = str(video_root / f"{cam_name}.{video_format}")
            outputdict = {
                "-pix_fmt": "yuv420p",
                "-vf": "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-r": str(output_fps),
            }
            if video_format == "mp4":
                outputdict.update({
                    "-c:v": "libx264",
                    "-preset": "slow",
                    "-profile:v": "high",
                    "-level:v": "4.0",
                    "-crf": quality_to_crf.get(video_quality, "28"),
                })
            video_writers[cam_name] = skvideo.io.FFmpegWriter(
                video_path,
                inputdict={"-framerate": str(output_fps)},
                outputdict=outputdict,
            )

    # Export loop
    for t in range(0, T, every_n):
        backend.set_frame(t)

        if not follow_joints:
            # Fixed cams: create a NEW camera object per render (cache-proof)
            if debug_print_cam0 and t == 0:
                for k, (p, q, fov) in fixed.items():
                    print(k, "pos=", p, "tgt=", q, "fov=", fov)

            for cam_name, (pos, tgt, fov) in fixed.items():
                cam = PinholeCamera(pos, tgt, w, h, fov=float(fov))
                backend.set_camera(cam)
                if export_video:
                    frame = backend.render_current_frame()
                    video_writers[cam_name].writeFrame(frame)
                else:
                    frame_path = out_root / cam_name / f"{t:06d}.png"
                    backend.export_frame(frame_path)

        else:
            # Joint-follow: compute pos/tgt per cam per frame, create NEW camera object each time
            assert specs is not None and joints1 is not None and joints2 is not None

            Rw1 = SciR.from_rotvec(root1[t]).as_matrix().astype(np.float32)
            Rw2 = SciR.from_rotvec(root2[t]).as_matrix().astype(np.float32)

            world_up = np.array([0., 1., 0.], dtype=np.float32)  # Z-up 좌표계

            for cam_name, sp in specs.items():
                who = sp["who"]
                j = int(sp["j"])
                off = sp["off"].astype(np.float32)
                fov = float(sp["fov"])

                if who in ("p1", "p2"):
                    joints_p = joints1 if who == "p1" else joints2
                    Rw = Rw1 if who == "p1" else Rw2
                    joint = joints_p[t, j, :].astype(np.float32)

                    if cam_name in ("p1_head", "p2_head"):
                        # Head: 어깨 벡터와 world_up 의 외적으로 정면 방향 계산
                        lsh = joints_p[t, int(sp["j_lsh"]), :].astype(np.float32)
                        rsh = joints_p[t, int(sp["j_rsh"]), :].astype(np.float32)
                        right_vec = rsh - lsh
                        fwd = np.cross(world_up, right_vec)
                        fwd_len = np.linalg.norm(fwd)
                        if fwd_len < 1e-6:
                            # fallback: root_orient 기반 +Z 방향
                            fwd = (Rw @ np.array([0., 0., 1.], dtype=np.float32))
                        else:
                            fwd = (fwd / fwd_len).astype(np.float32)
                        pos = joint + (Rw @ off) + fwd * 0.15
                        tgt = joint + fwd * 1.0

                    else:
                        # Wrist: elbow -> wrist 팔 벡터 방향으로 바라봄
                        elbow = joints_p[t, int(sp["j_elbow"]), :].astype(np.float32)
                        arm_vec = joint - elbow
                        arm_len = np.linalg.norm(arm_vec)
                        arm_dir = (arm_vec / arm_len).astype(np.float32) if arm_len > 1e-6 else np.array([1., 0., 0.], dtype=np.float32)
                        # 팔 방향 기준 5cm 앞: 어느 방향을 향하든 항상 손목 mesh 바깥에 위치
                        pos = joint + arm_dir * 0.15
                        tgt = joint + arm_dir * 0.6

                else:  # both_root (topdown)
                    rootj1 = joints1[t, j, :].astype(np.float32)
                    rootj2 = joints2[t, j, :].astype(np.float32)
                    center = 0.5 * (rootj1 + rootj2)
                    pos = center + off   # off = [0, 0, 3.5] → Z=3.5 위에서
                    tgt = center         # 중간점을 향해 내려다봄 (-Z 방향)

                if debug_print_cam0 and t == 0:
                    print(cam_name, "pos=", pos, "tgt=", tgt, "fov=", fov)

                # ✅ 핵심: NEW camera object for every (t, cam) to avoid internal caching
                cam = PinholeCamera(pos, tgt, w, h, fov=fov)
                backend.set_camera(cam)
                if export_video:
                    frame = backend.render_current_frame()
                    video_writers[cam_name].writeFrame(frame)
                else:
                    frame_path = out_root / cam_name / f"{t:06d}.png"
                    backend.export_frame(frame_path)

        if t == 0 or (t % max(50 * every_n, 1) == 0):
            print(f"  frame {t:06d}/{T-1:06d}")

    # Close video writers
    if export_video:
        for cam_name, writer in video_writers.items():
            writer.close()
        print(f"[OK] Done. Videos saved under: {video_root}")
    else:
        print(f"[OK] Done. Saved under: {out_root}")

    backend.close()


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", type=str, default=None, help="Clip folder name (e.g., G001T000A000R000). Omit to process all clips.")
    ap.add_argument("--motions-dir", type=str, default="./motions", help="Directory containing clip folders.")
    ap.add_argument("--out-dir", type=str, default="./exports", help="Output directory root.")
    ap.add_argument("--body-models-dir", type=str, default="./body_models", help="SMPL-X model root (must contain ./smplx).")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--every-n", type=int, default=1, help="Frame stride (1=every frame).")
    ap.add_argument("--max-frames", type=int, default=None, help="Export at most this many frames.")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="SMPL device. Start with cpu.")
    ap.add_argument("--follow-joints", action="store_true", help="Follow wrist/head joints per frame.")
    # Fallback indices
    ap.add_argument("--idx-root", type=int, default=0, help="Root/pelvis joint index (fallback).")
    ap.add_argument("--idx-left-wrist", type=int, default=20, help="Left wrist joint index (fallback).")
    ap.add_argument("--idx-right-wrist", type=int, default=21, help="Right wrist joint index (fallback).")
    ap.add_argument("--idx-head", type=int, default=15, help="Head (or neck) joint index (fallback). Try 12 if 15 is wrong.")
    ap.add_argument("--idx-left-elbow", type=int, default=18, help="Left elbow joint index (fallback).")
    ap.add_argument("--idx-right-elbow", type=int, default=19, help="Right elbow joint index (fallback).")
    ap.add_argument("--idx-left-shoulder", type=int, default=16, help="Left shoulder joint index (fallback).")
    ap.add_argument("--idx-right-shoulder", type=int, default=17, help="Right shoulder joint index (fallback).")
    ap.add_argument("--debug-cam0", action="store_true", help="Print camera poses at frame 0.")
    # Video export options
    ap.add_argument("--video", action="store_true", help="Export as video instead of per-frame images.")
    ap.add_argument("--video-format", type=str, default="mp4", choices=["mp4", "webm"], help="Video format (default: mp4).")
    ap.add_argument("--video-quality", type=str, default="medium", choices=["high", "medium", "low"], help="Video quality (default: medium). MP4 only.")
    return ap.parse_args()


def discover_clips(motions_dir: Path) -> list:
    """Find all clip directories that contain both P1.npz and P2.npz."""
    clips = sorted([
        d.name for d in motions_dir.iterdir()
        if d.is_dir() and (d / "P1.npz").exists() and (d / "P2.npz").exists()
    ])
    return clips


def main():
    args = parse_args()

    motions_dir = Path(args.motions_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    body_models_dir = Path(args.body_models_dir).resolve()

    if not (body_models_dir / "smplx").exists():
        raise FileNotFoundError(f"Expected SMPL-X folder not found: {body_models_dir / 'smplx'}")

    # Determine clip list
    if args.clip_id:
        clip_ids = [args.clip_id]
        clip_dir = motions_dir / args.clip_id
        if not clip_dir.exists():
            raise FileNotFoundError(f"Clip dir not found: {clip_dir}")
    else:
        clip_ids = discover_clips(motions_dir)
        if not clip_ids:
            raise FileNotFoundError(f"No clips found (P1.npz + P2.npz) under: {motions_dir}")
        print(f"[INFO] Found {len(clip_ids)} clips to export.")

    ensure_dir(out_dir)

    for i, clip_id in enumerate(clip_ids):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(clip_ids)}] {clip_id}")
        print(f"{'='*60}")
        try:
            export_clip(
                clip_id=clip_id,
                motions_dir=motions_dir,
                out_dir=out_dir,
                body_models_dir=body_models_dir,
                w=int(args.width),
                h=int(args.height),
                fps=int(args.fps),
                every_n=max(1, int(args.every_n)),
                max_frames=args.max_frames,
                device=args.device,
                follow_joints=bool(args.follow_joints),
                idx_root=int(args.idx_root),
                idx_left_wrist=int(args.idx_left_wrist),
                idx_right_wrist=int(args.idx_right_wrist),
                idx_head=int(args.idx_head),
                idx_left_elbow=int(args.idx_left_elbow),
                idx_right_elbow=int(args.idx_right_elbow),
                idx_left_shoulder=int(args.idx_left_shoulder),
                idx_right_shoulder=int(args.idx_right_shoulder),
                debug_print_cam0=bool(args.debug_cam0),
                export_video=bool(args.video),
                video_format=args.video_format,
                video_quality=args.video_quality,
            )
        except Exception as e:
            print(f"[ERROR] Failed to export {clip_id}: {e}")
            continue

    print(f"\n[DONE] Exported {len(clip_ids)} clip(s) to: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(130)