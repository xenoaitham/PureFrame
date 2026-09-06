"""Tests for the eval-parity gate (scripts/check_eval_parity.py)."""

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = (Path(__file__).parent.parent / "scripts" / "check_eval_parity.py").resolve()
_spec = importlib.util.spec_from_file_location("check_eval_parity", _SCRIPT)
_parity = importlib.util.module_from_spec(_spec)
sys.modules["check_eval_parity"] = _parity
_spec.loader.exec_module(_parity)


def _report(labels_by_scene, f1=0.0):
    return {
        "aggregate_metrics": {"precision": f1, "recall": f1, "f1_score": f1},
        "results": [
            {"scene_id": sid, "labels": labels}
            for sid, labels in labels_by_scene.items()
        ],
    }


def test_signature_identical_within_tolerance():
    base = _report(
        {"LA-001": {"FACE_FEMALE": 0.91, "BELLY_EXPOSED": 0.42}, "LA-002": {}}
    )
    cur = _report(
        {"LA-001": {"FACE_FEMALE": 0.93, "BELLY_EXPOSED": 0.41}, "LA-002": {}}
    )
    assert _parity.compare_signatures(base, cur) == []


def test_signature_detects_score_drift():
    base = _report({"LA-001": {"FACE_FEMALE": 0.91}})
    cur = _report({"LA-001": {"FACE_FEMALE": 0.97}})
    problems = _parity.compare_signatures(base, cur)
    assert len(problems) == 1 and "FACE_FEMALE" in problems[0]


def test_signature_detects_missing_and_new_labels():
    base = _report({"LA-001": {"FACE_FEMALE": 0.91}})
    cur = _report({"LA-001": {"BELLY_EXPOSED": 0.6}})
    problems = _parity.compare_signatures(base, cur)
    assert any("missing" in p for p in problems)
    assert any("appeared" in p for p in problems)


def test_signature_ignores_subfloor_scores():
    base = _report({"LA-001": {"FACE_FEMALE": 0.91, "ARMPITS": 0.01}})
    cur = _report({"LA-001": {"FACE_FEMALE": 0.91}})
    assert _parity.compare_signatures(base, cur) == []


def test_signature_reports_scene_set_changes():
    base = _report({"LA-001": {}})
    cur = _report({"LA-001": {}, "LA-002": {}})
    assert _parity.compare_signatures(base, cur) == [
        "LA-002: present in only one report"
    ]


def test_main_bootstrap_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_eval_parity.py", "evaluation_report.json"])
    report = _report({}, f1=0.0)
    (tmp_path / "evaluation_report.json").write_text(json.dumps(report))
    rc = _parity.main()
    assert rc == 0
    assert "bootstrap" in capsys.readouterr().err


def test_main_fails_on_signature_drift(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _report({"LA-001": {"FACE_FEMALE": 0.91}}, f1=0.8)
    cur = _report({"LA-001": {"FACE_FEMALE": 0.97}}, f1=0.8)
    (tmp_path / "eval-baseline.json").write_text(json.dumps(base))
    (tmp_path / "evaluation_report.json").write_text(json.dumps(cur))
    assert _parity.main() == 1
