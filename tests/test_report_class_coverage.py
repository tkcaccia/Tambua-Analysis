from analysis.report_class_coverage import report


def case(name, matrix):
    classes = ["epithelial", "lymphocyte", "macrophage"]
    return {
        "sample_id": name,
        "organ": "lung",
        "n_cells": sum(map(sum, matrix)),
        "support": {name: sum(row) for name, row in zip(classes, matrix)},
        "confusion_matrix": matrix,
    }


def test_single_class_cannot_contribute_to_multiclass_image_score():
    classes = ["epithelial", "lymphocyte", "macrophage"]
    result = report([
        case("single", [[100, 0, 0], [0, 0, 0], [0, 0, 0]]),
        case("mixed", [[3, 2, 0], [3, 2, 0], [0, 0, 0]]),
        case("rare", [[9, 1, 0], [1, 0, 0], [0, 0, 0]]),
    ], classes)
    assert result["single_class_images"] == 1
    assert result["multiclass"]["images"] == 2
    assert result["multiclass_minimum_five"]["images"] == 1
    assert result["per_image"][0]["observed_class_macro_recall"] is None
    assert result["per_image"][1]["observed_class_macro_recall"] == 0.5
    assert result["per_image"][2]["minimum_five_class_macro_recall"] is None
