"""
Interactive SMPLX viewer for Inter-X dataset.

Supports:
  - Full SMPLX mesh visualization (when NPZ data available)
  - Skeleton-only visualization (when only NPY data available)
  - Body-attached cameras for both P1 and P2 (head, hand, top view)
  - Navigation between sequences with UP/DOWN arrow keys
  - View from any camera by selecting it in the scene hierarchy and clicking
    "View from camera" in the properties panel.

Usage:
    python scripts/visualize.py --data_dir datasets/interx --sequence G001T000A000R000

    # Visualize all sequences (navigate with UP/DOWN):
    python scripts/visualize.py --data_dir datasets/interx
"""

import argparse
import os
import sys

# Add project root to path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

# Pre-load system GL libraries before moderngl import (workaround for missing -dev symlinks).
import src.gl_setup  # noqa: F401, E402

# Patch PyQt5Window.resize to handle missing _ctx before GL context init.
import moderngl_window.context.pyqt5.window as _pyqt5_win

_orig_resize = _pyqt5_win.Window.resize

def _safe_resize(self, width, height):
    if not hasattr(self, "_ctx"):
        self._width = width // self._widget.devicePixelRatio()
        self._height = height // self._widget.devicePixelRatio()
        self._buffer_width = width
        self._buffer_height = height
        return
    _orig_resize(self, width, height)

_pyqt5_win.Window.resize = _safe_resize

import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.viewer import Viewer

from src.interx_loader import (
    InterXSequence,
    create_skeleton_nodes,
    create_smpl_sequences,
    list_sequences,
    load_sequence,
)
from src.body_cameras import create_body_cameras


class InterXViewer(Viewer):
    """Interactive viewer for Inter-X SMPLX data with body-attached cameras."""

    title = "Inter-X SMPLX Viewer"

    def __init__(
        self,
        data_dir: str,
        sequences: list,
        show_cameras: bool = True,
        fps: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.sequence_names = sequences
        self.show_cameras = show_cameras
        self.fps = fps
        self.current_idx = 0

        # Key bindings for sequence navigation.
        self._prev_key = self.wnd.keys.UP
        self._next_key = self.wnd.keys.DOWN

        # Load first sequence.
        self._load_current_sequence()

    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self._prev_key:
                self._navigate(-1)
                return
            elif key == self._next_key:
                self._navigate(1)
                return
        super().key_event(key, action, modifiers)

    def _navigate(self, direction: int):
        """Navigate to the previous/next sequence."""
        if len(self.sequence_names) <= 1:
            return
        self.current_idx = (self.current_idx + direction) % len(self.sequence_names)
        self._clear_scene()
        self._load_current_sequence()
        self.scene.current_frame_id = 0

    def _clear_scene(self):
        """Remove all user-added nodes from the scene."""
        for node in self.scene.nodes.copy():
            self.scene.remove(node)

    def _load_current_sequence(self):
        """Load and display the current sequence."""
        seq_name = self.sequence_names[self.current_idx]
        print(f"\nLoading sequence: {seq_name} ({self.current_idx + 1}/{len(self.sequence_names)})")

        seq = load_sequence(self.data_dir, seq_name, fps=self.fps)

        if seq.text:
            print(f"Description: {seq.text[:200]}...")

        # Create visualization nodes based on available data format.
        if seq.smplx_params_p1 is not None:
            # Full SMPLX mesh visualization.
            print("  Mode: SMPLX mesh (NPZ)")
            seq_p1, seq_p2 = create_smpl_sequences(seq, device=C.device)
            self.scene.add(seq_p1)
            self.scene.add(seq_p2)
        else:
            # Skeleton-only visualization.
            print("  Mode: Skeleton (NPY)")
            skel_p1, skel_p2 = create_skeleton_nodes(seq)
            self.scene.add(skel_p1)
            self.scene.add(skel_p2)

        # Add body-attached cameras for both persons.
        if self.show_cameras and seq.joints_p1 is not None:
            cameras = create_body_cameras(
                seq.joints_p1,
                seq.joints_p2,
                joint_map=seq.joint_map,
                viewer=self,
                width=self.window_size[0],
                height=self.window_size[1],
            )
            for cam_name, cam in cameras.items():
                self.scene.add(cam)
                print(f"  Added camera: {cam_name}")

        self.scene.fps = self.fps
        self.playback_fps = self.fps
        print(f"  Frames: {seq.n_frames}, FPS: {self.fps}")


def main():
    parser = argparse.ArgumentParser(description="Interactive SMPLX viewer for Inter-X dataset")
    parser.add_argument("--data_dir", type=str, default="datasets/interx",
                        help="Path to Inter-X dataset directory")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Specific sequence to visualize (e.g., G001T000A000R000)")
    parser.add_argument("--no_cameras", action="store_true",
                        help="Do not add body-attached cameras")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target playback FPS")
    args = parser.parse_args()

    # Resolve data directory relative to project root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, args.data_dir)

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
        print(f"Found {len(sequences)} sequences. Use UP/DOWN arrows to navigate.")

    viewer = InterXViewer(
        data_dir=data_dir,
        sequences=sequences,
        show_cameras=not args.no_cameras,
        fps=args.fps,
    )
    viewer.run()


if __name__ == "__main__":
    main()
