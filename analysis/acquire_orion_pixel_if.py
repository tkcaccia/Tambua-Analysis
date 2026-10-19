#!/usr/bin/env python3
"""Range-read registered ORION fluorescence pixels for a saved H&E window.

This CPU-only acquisition reads only the requested channels and XY window from
the public tiled OME-TIFF. It never downloads a whole multiplex WSI. The
measured signal is preserved as 16-bit pixels for later visual and numerical
comparison with virtual markers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import fsspec
import numpy as np
import requests
import tifffile
import zarr

BASE = "https://lin-2023-orion-crc.s3.amazonaws.com/"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def source_for_patient(patient: str) -> tuple[str, str, list[str]]:
    response = requests.get(BASE, params={"list-type": "2", "prefix": f"data/{patient}/"},
                            timeout=(20, 60))
    response.raise_for_status()
    xml = ET.fromstring(response.content)
    if xml.findtext("s3:IsTruncated", namespaces=S3_NS) != "false":
        raise ValueError("S3 listing was truncated")
    keys = [entry.findtext("s3:Key", namespaces=S3_NS)
            for entry in xml.findall("s3:Contents", S3_NS)]
    candidates = [key for key in keys if key.endswith(".ome.tiff")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one ORION IF image for {patient}: {candidates}")
    marker_key = f"data/{patient}/markers.csv"
    if marker_key not in keys:
        raise ValueError("ORION marker list absent")
    marker_response = requests.get(BASE + marker_key, timeout=(20, 60))
    marker_response.raise_for_status()
    markers = [name.strip() for name in marker_response.text.splitlines() if name.strip()]
    return BASE + candidates[0], marker_key, markers


def acquire(prepared: Path, output: Path, markers: list[str]) -> dict:
    provenance = json.loads((prepared / "PROVENANCE.json").read_text())
    patient = provenance["patient"]
    url, marker_key, channel_names = source_for_patient(patient)
    if len(channel_names) != 19 or len(set(channel_names)) != 19:
        raise ValueError("Unexpected ORION channel list")
    wanted = list(dict.fromkeys(markers))
    if not wanted or not set(wanted) <= set(channel_names):
        raise ValueError(f"Unavailable measured markers: {set(wanted) - set(channel_names)}")
    x, y, size = (int(provenance[key]) for key in ("x", "y", "size_px"))
    if size <= 0 or min(x, y) < 0:
        raise ValueError("Invalid registered window")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with fsspec.open(url, "rb", block_size=1 << 20, cache_type="bytes").open() as handle:
        with tifffile.TiffFile(handle) as image:
            series = image.series[0]
            if series.axes != "IYX" or series.shape[0] != len(channel_names):
                raise ValueError(f"Unexpected ORION image axes/shape: {series.axes}, {series.shape}")
            if y + size > series.shape[1] or x + size > series.shape[2]:
                raise ValueError("Window outside measured image")
            ome = ET.fromstring(image.ome_metadata)
            pixels = next(node for node in ome.iter() if node.tag.endswith("Pixels"))
            mpp_x, mpp_y = float(pixels.attrib["PhysicalSizeX"]), float(pixels.attrib["PhysicalSizeY"])
            if abs(mpp_x - provenance["mpp_x"]) > 1e-6 or \
                    abs(mpp_y - provenance["mpp_y"]) > 1e-6:
                raise ValueError("Measured fluorescence and registered H&E pixel spacing differ")
            named = [node.attrib["Name"] for node in pixels if node.tag.endswith("Channel")]
            normalize = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())
            if len(named) != len(channel_names) or any(
                    normalize(marker) not in normalize(name)
                    for marker, name in zip(channel_names, named)):
                raise ValueError("OME channel metadata and released marker list disagree")
            data = zarr.open(series.aszarr(level=0), mode="r")
            for marker in wanted:
                channel = channel_names.index(marker)
                window = np.asarray(data[channel, y:y+size, x:x+size])
                if window.shape != (size, size) or window.dtype != np.uint16:
                    raise ValueError(f"Invalid measured pixels for {marker}")
                destination = output / f"{marker}.tif"
                tifffile.imwrite(destination, window, compression="zlib",
                                 metadata={"axes": "YX", "PhysicalSizeX": mpp_x,
                                           "PhysicalSizeY": mpp_y,
                                           "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"})
                records.append({"marker": marker, "channel": channel,
                                "sha256": hashlib.sha256(window.tobytes()).hexdigest(),
                                "min": int(window.min()), "max": int(window.max()),
                                "mean": float(window.mean())})
    manifest = {"patient": patient, "source_url": url, "marker_list_key": marker_key,
                "x": x, "y": y, "size_px": size, "mpp_x": mpp_x, "mpp_y": mpp_y,
                "channels": records,
                "interpretation": "Measured ORION fluorescence pixels, same-section registered H&E window; not a painted cell table."}
    (output / "MEASURED_PIXEL_SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markers", nargs="+", default=["Hoechst", "CD68", "CD3e", "Pan-CK"])
    args = parser.parse_args()
    result = acquire(args.prepared, args.output, args.markers)
    print(json.dumps({"patient": result["patient"], "channels": result["channels"]}))
