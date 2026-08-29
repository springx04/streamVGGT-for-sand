"""Tests for the MVS recorder's JSON camera sidecar format."""

from __future__ import annotations

import json

import numpy as np

from stream_omnivggt.cli.run_stream_demo import _load_camera


def test_load_camera_json_sidecar(tmp_path) -> None:
    camera_dir = tmp_path / "cameras"
    camera_dir.mkdir()
    (camera_dir / "frame_000000.json").write_text(
        json.dumps(
            {
                "intrinsic": [[100.0, 0.0, 32.0], [0.0, 101.0, 24.0], [0.0, 0.0, 1.0]],
                "distortion": [0.1, -0.02, 0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )

    intrinsic, extrinsic = _load_camera(tmp_path / "images" / "frame_000000.png", camera_dir)

    assert extrinsic is None
    assert intrinsic is not None
    assert intrinsic.shape == (3, 3)
    assert np.allclose(intrinsic, [[100.0, 0.0, 32.0], [0.0, 101.0, 24.0], [0.0, 0.0, 1.0]])
