import pytest

from analysis.score_orion_spatial_pixels import measured_patients


def test_measured_cohort_uses_record_list_not_patient_count():
    cohort = {"status": "complete", "device": "CPU", "patients": 10,
              "measured_marker_images": 100,
              "records": [{"patient": f"CRC{i:02d}"} for i in range(10)]}
    assert measured_patients(cohort) == [f"CRC{i:02d}" for i in range(10)]
    cohort["records"][-1]["patient"] = "CRC00"
    with pytest.raises(ValueError, match="distinct"):
        measured_patients(cohort)
