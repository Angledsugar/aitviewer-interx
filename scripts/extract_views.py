"""
Headless multi-camera image extraction for Inter-X SMPLX data.

Renders each camera view (head, hand, top) for both persons across all frames
and saves the images to disk. Uses GPU for both SMPLX FK and OpenGL rendering.

Usage:
    # Extract all camera views for ALL sequences:
    uv run python scripts/extract_views.py --data_dir datasets/interx

    # Extract a single sequence:
    uv run python scripts/extract_views.py --data_dir datasets/interx --sequence G001T000A000R000

    # Extract only specific cameras with custom resolution:
    uv run python scripts/extract_views.py --data_dir datasets/interx \
        --cameras P1_head_cam P2_head_cam top_cam --width 640 --height 480

    # Extract as MP4 video instead of frames:
    uv run python scripts/extract_views.py --data_dir datasets/interx --output_format video

    # Skip already-extracted sequences (for resuming interrupted runs):
    uv run python scripts/extract_views.py --data_dir datasets/interx --skip_existing
"""

import argparse
import os
import sys
import traceback
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


def _is_sequence_extracted(
    output_dir: str,
    sequence_name: str,
    cameras: list,
    output_format: str,
) -> bool:
    """Check if a sequence has already been fully extracted."""
    for cam_name in cameras:
        cam_dir = os.path.join(output_dir, sequence_name, cam_name)
        if not os.path.isdir(cam_dir):
            return False
        if output_format == "video":
            video_path = os.path.join(cam_dir, f"{cam_name}.mp4")
            if not os.path.isfile(video_path):
                return False
        else:
            # Check that at least one frame exists.
            has_frames = any(f.endswith(".png") for f in os.listdir(cam_dir))
            if not has_frames:
                return False
    return True


def _clear_scene(viewer):
    """Remove all user-added nodes from the scene."""
    for node in viewer.scene.nodes.copy():
        viewer.scene.remove(node)


def extract_sequence(
    viewer,
    data_dir: str,
    sequence_name: str,
    output_dir: str,
    cameras_to_extract: list,
    width: int,
    height: int,
    fps: int,
    output_format: str,
):
    """Extract camera view images for a single sequence using an existing renderer.

    Args:
        viewer: HeadlessRenderer instance (reused across sequences).
        data_dir: Path to Inter-X dataset directory.
        sequence_name: Sequence identifier.
        output_dir: Root output directory.
        cameras_to_extract: List of camera names to extract.
        width: Output image width.
        height: Output image height.
        fps: Target FPS.
        output_format: 'frames' for individual images, 'video' for MP4.
    """
    # Load sequence data.
    seq = load_sequence(data_dir, sequence_name, fps=fps)

    # Add scene objects based on data format.
    if seq.smplx_params_p1 is not None:
        seq_p1, seq_p2 = create_smpl_sequences(seq, device=C.device)
        viewer.scene.add(seq_p1)
        viewer.scene.add(seq_p2)
        mode = "SMPLX mesh (NPZ)"
    else:
        skel_p1, skel_p2 = create_skeleton_nodes(seq)
        viewer.scene.add(skel_p1)
        viewer.scene.add(skel_p2)
        mode = "Skeleton (NPY)"

    viewer.scene.fps = fps

    # Create cameras from joint data.
    if seq.joints_p1 is None:
        print(f"  [SKIP] No joint data for camera computation.")
        _clear_scene(viewer)
        return

    cameras = create_body_cameras(
        seq.joints_p1,
        seq.joints_p2,
        joint_map=seq.joint_map,
        viewer=viewer,
        width=width,
        height=height,
    )

    # Filter to requested cameras.
    cameras = {k: cam for k, cam in cameras.items() if k in cameras_to_extract}
    if not cameras:
        print(f"  [SKIP] No valid cameras. Available: {AVAILABLE_CAMERAS}")
        _clear_scene(viewer)
        return

    print(f"  Mode: {mode} | Frames: {seq.n_frames} | Cameras: {list(cameras.keys())}")

    # Extract each camera view.
    for cam_name, cam in cameras.items():
        cam_output_dir = os.path.join(output_dir, sequence_name, cam_name)
        os.makedirs(cam_output_dir, exist_ok=True)

        # Add camera to scene and set as active.
        viewer.scene.add(cam)
        viewer.set_temp_camera(cam)

        if output_format == "video":
            video_path = os.path.join(cam_output_dir, f"{cam_name}.mp4")
            print(f"    {cam_name} -> {video_path}")
            viewer.save_video(video_dir=video_path, output_fps=fps)
        else:
            print(f"    {cam_name} -> {cam_output_dir}/ ({seq.n_frames} frames)")
            viewer._init_scene()
            for frame_idx in tqdm(
                range(seq.n_frames),
                desc=f"    {cam_name}",
                leave=False,
            ):
                viewer.scene.current_frame_id = frame_idx
                frame_path = os.path.join(cam_output_dir, f"frame_{frame_idx:06d}.png")
                viewer.export_frame(frame_path)

        # Remove camera and reset for next camera.
        viewer.scene.remove(cam)
        viewer.reset_camera()

    # Clear scene for next sequence.
    _clear_scene(viewer)


def main():
    parser = argparse.ArgumentParser(
        description="Headless multi-camera image extraction for Inter-X SMPLX data"
    )
    parser.add_argument(
        "--data_dir", type=str, default="datasets/interx",
        help="Path to dataset directory (default: datasets/interx)",
    )
    parser.add_argument(
        "--sequence", type=str, default=None,
        help="Specific sequence name, or omit for all sequences",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Output directory for extracted images/videos (default: output)",
    )
    parser.add_argument(
        "--cameras", type=str, nargs="+", default=AVAILABLE_CAMERAS,
        choices=AVAILABLE_CAMERAS,
        help="Camera views to extract (default: all 7 cameras)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Output image width (default: 640)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Output image height (default: 480)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Target FPS (default: 30, original data is 120 FPS)",
    )
    parser.add_argument(
        "--output_format", type=str, default="frames",
        choices=["frames", "video"],
        help="'frames' for PNG images, 'video' for MP4 (default: frames)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip sequences that have already been extracted",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir)
    output_dir = os.path.join(project_root, args.output_dir)

    if not os.path.isdir(data_dir):
        print(f"Error: Dataset directory not found: {data_dir}")
        sys.exit(1)

    # Get sequence list.
    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = list_sequences(data_dir)
        if not sequences:
            print(f"Error: No sequences found in {data_dir}")
            sys.exit(1)

    # Filter out already-extracted sequences.
    if args.skip_existing:
        original_count = len(sequences)
        sequences = [
            s for s in sequences
            if not _is_sequence_extracted(output_dir, s, args.cameras, args.output_format)
        ]
        skipped = original_count - len(sequences)
        if skipped > 0:
            print(f"Skipping {skipped} already-extracted sequence(s).")
        if not sequences:
            print("All sequences already extracted. Nothing to do.")
            return

    print(f"{'='*60}")
    print(f"Camera Image Extraction")
    print(f"{'='*60}")
    print(f"  Dataset:     {data_dir}")
    print(f"  Sequences:   {len(sequences)}")
    print(f"  Cameras:     {args.cameras}")
    print(f"  Resolution:  {args.width}x{args.height}")
    print(f"  FPS:         {args.fps}")
    print(f"  Format:      {args.output_format}")
    print(f"  Output:      {output_dir}")
    print(f"  Device:      {C.device}")
    print(f"{'='*60}\n")

    # Create headless renderer once and reuse across sequences.
    from aitviewer.headless import HeadlessRenderer

    C.update_conf({"window_width": args.width, "window_height": args.height})
    viewer = HeadlessRenderer()

    # Process all sequences.
    succeeded = 0
    failed = []

    for seq_idx, seq_name in enumerate(
        tqdm(sequences, desc="Sequences", unit="seq")
    ):
        print(f"\n[{seq_idx + 1}/{len(sequences)}] {seq_name}")
        try:
            extract_sequence(
                viewer=viewer,
                data_dir=data_dir,
                sequence_name=seq_name,
                output_dir=output_dir,
                cameras_to_extract=args.cameras,
                width=args.width,
                height=args.height,
                fps=args.fps,
                output_format=args.output_format,
            )
            succeeded += 1
        except Exception as e:
            print(f"  [ERROR] {seq_name}: {e}")
            traceback.print_exc()
            failed.append((seq_name, str(e)))
            # Clear scene to recover for next sequence.
            try:
                _clear_scene(viewer)
            except Exception:
                pass

    # Summary.
    print(f"\n{'='*60}")
    print(f"Extraction Complete")
    print(f"{'='*60}")
    print(f"  Succeeded: {succeeded}/{len(sequences)}")
    if failed:
        print(f"  Failed:    {len(failed)}")
        for name, err in failed:
            print(f"    - {name}: {err}")
    print(f"  Output:    {output_dir}")


if __name__ == "__main__":
    main()
