import json
import pandas as pd
import pytest

import src.gradient_boosting_comparison as module


def _metrics(value):
    return {key: value for key, _ in module._METRICS}


def test_build_comparison_has_expected_changes():
    frame = module.build_hist_gradient_boosting_comparison(_metrics(.5), _metrics(.7))
    assert module.HIST_GRADIENT_BOOSTING_COMPARISON_ID == "M04-HGB-COMP-001"
    assert list(frame) == ["metric", "baseline_value", "tuned_value", "absolute_change"]
    assert (frame["absolute_change"] == frame["tuned_value"] - frame["baseline_value"]).all()


def test_baseline_loader_validates_temporary_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "get_project_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError): module.load_hist_gradient_boosting_baseline_metrics("missing.json")
    path = tmp_path / "bad.json"; path.write_text(json.dumps({"experiment_id": "wrong", "metrics": {}}))
    with pytest.raises(ValueError): module.load_hist_gradient_boosting_baseline_metrics(path)


def test_comparison_csv_uses_temporary_path(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "get_project_root", lambda: tmp_path)
    frame = module.build_hist_gradient_boosting_comparison(_metrics(.5), _metrics(.7))
    path = module.save_hist_gradient_boosting_comparison_table(frame, "comparison.csv")
    assert pd.read_csv(path).shape == (7, 4)
    with pytest.raises(FileExistsError): module.save_hist_gradient_boosting_comparison_table(frame, "comparison.csv")
