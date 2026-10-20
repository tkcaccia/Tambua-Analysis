import numpy as np
from PIL import Image
import tifffile

from analysis.stain_reference_preflight import (
    lab_stats, normalize, reference_profile, write_calibrated_ome,
)


def test_reference_transform_preserves_rgb_geometry_and_moves_tissue_toward_reference(tmp_path):
    yy, xx = np.indices((64, 64))
    variation = ((xx + 2 * yy) % 25).astype(np.int16)
    ref = np.stack([215 + variation // 3, 185 + variation, 210 + variation // 2], axis=-1)
    source = np.stack([150 + variation, 90 + variation, 145 + variation], axis=-1)
    ref = np.clip(ref, 0, 255).astype(np.uint8)
    source = np.clip(source, 0, 255).astype(np.uint8)
    ref[:8, :8] = 255
    source[:8, :8] = 255

    folder = tmp_path / "reference"
    folder.mkdir()
    paths = []
    for idx in range(10):
        path = folder / f"{idx:02d}_he.png"
        Image.fromarray(ref, mode="RGB").save(path)
        paths.append(path)
    profile = reference_profile(paths)
    transformed, report = normalize(source, profile)

    source_mean, _, _ = lab_stats(source)
    changed_mean, _, _ = lab_stats(transformed)
    target = np.asarray(profile["target_lab_mean"])
    assert transformed.shape == source.shape
    assert transformed.dtype == np.uint8
    assert np.all(transformed[:8, :8] == 255)
    assert report["pure_white_unchanged"]
    assert abs(changed_mean[0] - target[0]) < abs(source_mean[0] - target[0])


def test_calibrated_ome_export_is_lossless_and_preserves_spacing(tmp_path):
    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[..., 0] = 210
    rgb[..., 1] = 145
    rgb[..., 2] = 190
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "normalized.ome.tif"
    tifffile.imwrite(source, rgb, photometric="rgb", ome=True,
                     metadata={"axes": "YXS", "PhysicalSizeX": .252,
                               "PhysicalSizeXUnit": "µm", "PhysicalSizeY": .252,
                               "PhysicalSizeYUnit": "µm"})
    recorded = write_calibrated_ome(source, rgb, output)
    assert recorded["x_um_per_px"] == .252
    assert recorded["y_um_per_px"] == .252
    with tifffile.TiffFile(output) as saved:
        assert np.array_equal(saved.asarray(), rgb)
