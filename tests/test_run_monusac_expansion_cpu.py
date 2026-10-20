"""CPU deadlines scale with physically calibrated expansion image size."""

import pytest

from analysis.run_monusac_expansion_cpu import inference_timeout


def test_deadline_covers_large_registered_roi_but_is_bounded():
    small = {"width": 311, "height": 341, "mpp": .2527}
    large = {"width": 475, "height": 2000, "mpp": .252}
    assert inference_timeout(small) == 240
    assert 1000 < inference_timeout(large) < 1200
    with pytest.raises(ValueError, match="audited scope"):
        inference_timeout({"width": 4000, "height": 4000, "mpp": .25})
