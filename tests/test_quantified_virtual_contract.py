import csv
import json

import pytest

from analysis.quantify_detected_monusac_cells import (
    GIGATIME_CHANNELS, VALUE_SEMANTICS, validate_quantified_table,
)


def test_quantified_table_requires_virtual_score_provenance(tmp_path):
    table = tmp_path / "mean.csv"
    summary = tmp_path / "summary.json"
    fields = ["label_id", "mask_name", "value_semantics",
              "calibrated_probability", "measured_protein_abundance",
              *GIGATIME_CHANNELS]
    row = {name: "0.5" for name in GIGATIME_CHANNELS}
    row.update(label_id="1", mask_name="nuclei",
               value_semantics=VALUE_SEMANTICS,
               calibrated_probability="False", measured_protein_abundance="False")
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    summary.write_text(json.dumps({
        "mask_name": "nuclei", "channel_names": list(GIGATIME_CHANNELS),
        "objects_quantified": 1,
        "value_semantics": {"value_type": VALUE_SEMANTICS,
                            "calibrated_probability": False,
                            "measured_protein_abundance": False}}))
    assert validate_quantified_table(table, summary, "nuclei") == 1
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[name for name in fields
                                                    if name != "value_semantics"])
        writer.writeheader()
        writer.writerow({key: value for key, value in row.items()
                         if key != "value_semantics"})
    with pytest.raises(ValueError, match="omitted virtual-score columns"):
        validate_quantified_table(table, summary, "nuclei")
