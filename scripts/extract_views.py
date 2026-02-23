"""
Headless multi-camera image extraction for Inter-X SMPLX data.

Renders each camera view (head, hand, top) for both persons across all frames
and saves the images to disk. Uses GPU for both SMPLX FK and OpenGL rendering.

Usage:
    # Extract all camera views for a single sequence:
    python scripts/extract_views.py --data_dir datasets/interx --sequence G001T000A000R000

    # Extract only P1 head camera with custom resolution:
    python scripts/extract_views.py --data_dir datasets/interx --sequence G001T000A000R000 \
        --cameras P1_head_cam --width 640 --height 480

    # Extract all sequences as videos:
    python scripts/extract_views.py --data_dir datasets/interx --output_format video
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

# Pre-load system GL libraries before moderngl import (workaround for missing -dev symlinks).
import src.gl_setup  # noqa: F401, E402

import numpy as np
from tqdm import tqdm

from aitviewer.configuration import CONFIG as C

from src.interx_loader import (
    create_skeleton_nodes,
    create_smpl_sequences,
    list_sequences,
    load_sequence,
)
from src.body_cameras import create_body_cameras


AVAILABLE_CAMERAS = [
    "P1_head_cam", "P1_right_hand_cam", "P1_left_hand_cam",
    "P2_head_cam", "P2_right_hand_cam", "P2_left_hand_cam",
    "top_cam",
]


def extract_frames(
    data_dir: str,
    sequence_name: str,
    output_dir: str,
    cameras_to_extract: list,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    output_format: str = "frames",
):
    """Extract camera view images for a single sequence.

    Args:
        data_dir: Path to Inter-X dataset directory.
        sequence_name: Sequence identifier.
        output_dir: Root output directory.
        cameras_to_extract: List of camera names to extract.
        width: Output image width.
        height: Output image height.
        fps: Target FPS.
        output_format: 'frames' for individual images, 'video' for MP4.
    """
    from aitviewer.headless import HeadlessRenderer

    print(f"\n{'='*60}")
    print(f"Extracting: {sequence_name}")
    print(f"{'='*60}")

    # Load sequence data.
    seq = load_sequence(data_dir, sequence_name, fps=fps)

    # Configure headless renderer.
    C.update_conf({"window_width": width, "window_height": height})

    v = HeadlessRenderer()

    # Add scene objects based on data format.
    if seq.smplx_params_p1 is not None:
        print("  Mode: SMPLX mesh (NPZ)")
        seq_p1, seq_p2 = create_smpl_sequences(seq, device=C.device)
        v.scene.add(seq_p1)
        v.scene.add(seq_p2)
    else:
        print("  Mode: Skeleton (NPY)")
        skel_p1, skel_p2 = create_skeleton_nodes(seq)
        v.scene.add(skel_p1)
        v.scene.add(skel_p2)

    v.scene.fps = fps

    # Create cameras for both persons.
    if seq.joints_p1 is None:
        print("  Error: No joint data available for camera computation.")
        return

    cameras = create_body_cameras(
        seq.joints_p1,
        seq.joints_p2,
        joint_map=seq.joint_map,
        viewer=v,
        width=width,
        height=height,
    )

    # Filter to requested cameras.
    cameras = {k: v_cam for k, v_cam in cameras.items() if k in cameras_to_extract}

    if not cameras:
        print(f"  No valid cameras selected. Available: {AVAILABLE_CAMERAS}")
        return

    # Extract each camera view.
    for cam_name, cam in cameras.items():
        print(f"\n  Camera: {cam_name}")
        cam_output_dir = os.path.join(output_dir, sequence_name, cam_name)

        # Add camera to scene and set as active.
        v.scene.add(cam)
        v.set_temp_camera(cam)

        if output_format == "video":
            # Render as video.
            video_path = os.path.join(cam_output_dir, f"{cam_name}.mp4")
            os.makedirs(cam_output_dir, exist_ok=True)
            print(f"    Saving video to: {video_path}")
            v.save_video(video_dir=video_path, output_fps=fps)
        else:
            # Render individual frames.
            os.makedirs(cam_output_dir, exist_ok=True)
            print(f"    Saving {seq.n_frames} frames to: {cam_output_dir}")

            v._init_scene()
            for frame_idx in tqdm(range(seq.n_frames), desc=f"    {cam_name}"):
                v.scene.current_frame_id = frame_idx
                frame_path = os.path.join(cam_output_dir, f"frame_{frame_idx:06d}.png")
                v.export_frame(frame_path)

        # Remove camera and reset for next camera.
        v.scene.remove(cam)
        v.reset_camera()

    print(f"\n  Done: {sequence_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Headless multi-camera image extraction for Inter-X SMPLX data"
    )
    parser.add_argument("--data_dir", type=str, default="datasets/interx",
                        help="Path to Inter-X dataset directory")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Specific sequence (or 'all' for all sequences)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for extracted images/videos")
    parser.add_argument("--cameras", type=str, nargs="+", default=AVAILABLE_CAMERAS,
                        choices=AVAILABLE_CAMERAS,
                        help="Camera views to extract")
    parser.add_argument("--width", type=int, default=640,
                        help="Output image width")
    parser.add_argument("--height", type=int, default=480,
                        help="Output image height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS")
    parser.add_argument("--output_format", type=str, default="frames",
                        choices=["frames", "video"],
                        help="'frames' for PNG images, 'video' for MP4")
    args = parser.parse_args()

    # Resolve paths relative to project root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir)
    output_dir = os.path.join(project_root, args.output_dir)

    if not os.path.isdir(data_dir):
        print(f"Error: Dataset directory not found: {data_dir}")
        sys.exit(1)

    # Get sequence list.
    if args.sequence and args.sequence != "all":
        sequences = [args.sequence]
    else:
        sequences = list_sequences(data_dir)
        if not sequences:
            print(f"Error: No sequences found in {data_dir}")
            sys.exit(1)

    print(f"Sequences to process: {len(sequences)}")
    print(f"Cameras: {args.cameras}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Output format: {args.output_format}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {C.device}")

    for seq_name in sequences:
        extract_frames(
            data_dir=data_dir,
            sequence_name=seq_name,
            output_dir=output_dir,
            cameras_to_extract=args.cameras,
            width=args.width,
            height=args.height,
            fps=args.fps,
            output_format=args.output_format,
        )

    print(f"\nAll done! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
