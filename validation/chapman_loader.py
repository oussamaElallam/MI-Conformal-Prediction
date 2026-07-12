"""Chapman-Shaoxing loader with explicit cohort accounting and preprocessing.

The public ``load_chapman_data`` API remains backward compatible: by default it
returns ``(X, y)``.  Set ``return_metadata=True`` to additionally receive the
record manifest and exclusion log used by the in-distribution reviewer experiment.
"""

from __future__ import annotations

import math
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

MI_CODES = {
    "426177001",  # Old myocardial infarction
    "164865005",  # Myocardial infarction
    "427395009",  # Acute myocardial infarction
}
NORMAL_CODE = "426783006"  # Sinus rhythm
DATA_DIR = "chapman_shaoxing"
STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _normalise_lead_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
    aliases = {"AVR": "AVR", "AVL": "AVL", "AVF": "AVF"}
    return aliases.get(cleaned, cleaned)


def _comment_value(comments: Iterable[str], keys: Iterable[str]) -> str | None:
    patterns = [re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", re.I) for key in keys]
    for comment in comments:
        text = str(comment)
        for pattern in patterns:
            match = pattern.match(text)
            if match:
                return match.group(1).strip()
    return None


def _parse_diagnosis(comments: Iterable[str]) -> tuple[int | None, str]:
    diagnosis = _comment_value(comments, ["Dx", "Diagnosis"])
    if not diagnosis:
        return None, "missing_diagnosis"
    codes = {code.strip() for code in diagnosis.split(",") if code.strip()}
    is_mi = bool(codes & MI_CODES)
    is_normal = NORMAL_CODE in codes
    if is_mi and not is_normal:
        return 1, "mi"
    if is_normal and not is_mi:
        return 0, "normal"
    if is_mi and is_normal:
        return None, "conflicting_mi_and_normal_labels"
    return None, "outside_binary_mi_normal_cohort"


def _parse_age(comments: Iterable[str]) -> float:
    value = _comment_value(comments, ["Age", "Patient Age"])
    match = re.search(r"\d+(?:\.\d+)?", value or "")
    return float(match.group(0)) if match else float("nan")


def _parse_group_id(record_name: str, comments: Iterable[str]) -> tuple[str, str]:
    value = _comment_value(
        comments,
        ["Patient ID", "PatientID", "Subject ID", "SubjectID", "Study ID", "StudyID"],
    )
    if value:
        return str(value), "header_patient_id"
    # The PhysioNet Chapman release is normally one record per participant.  We
    # still record this fallback explicitly so the manuscript does not overstate
    # patient-level evidence when an alternative export lacks patient identifiers.
    return os.path.basename(record_name), "record_id_fallback"


def _reorder_leads(signal: np.ndarray, signal_names: Iterable[str]) -> np.ndarray:
    names = [_normalise_lead_name(name) for name in signal_names]
    expected = [_normalise_lead_name(name) for name in STANDARD_LEADS]
    missing = [STANDARD_LEADS[i] for i, lead in enumerate(expected) if lead not in names]
    if missing:
        raise ValueError("missing_leads=" + ",".join(missing))
    indices = [names.index(lead) for lead in expected]
    return signal[:, indices]


def _prepare_signal(
    signal: np.ndarray,
    source_frequency: float,
    *,
    target_frequency: int,
    target_samples: int,
    short_record_policy: str,
) -> np.ndarray:
    if source_frequency <= 0:
        raise ValueError(f"invalid_sampling_frequency={source_frequency}")
    if short_record_policy not in {"skip", "pad"}:
        raise ValueError("short_record_policy must be 'skip' or 'pad'")

    required_source_samples = int(math.ceil(target_samples * source_frequency / target_frequency))
    if len(signal) < required_source_samples:
        if short_record_policy == "skip":
            raise ValueError(f"short_record={len(signal)}<{required_source_samples}")
        signal = np.pad(signal, ((0, required_source_samples - len(signal)), (0, 0)))
    else:
        signal = signal[:required_source_samples]

    source_hz = int(round(source_frequency))
    divisor = math.gcd(source_hz, int(target_frequency))
    resampled = resample_poly(
        signal,
        up=int(target_frequency) // divisor,
        down=source_hz // divisor,
        axis=0,
    )
    if len(resampled) < target_samples:
        resampled = np.pad(resampled, ((0, target_samples - len(resampled)), (0, 0)))
    return np.asarray(resampled[:target_samples], dtype=np.float32)


def load_chapman_data(
    data_dir: str = DATA_DIR,
    max_records: int | None = None,
    *,
    target_frequency: int = 100,
    target_samples: int = 1000,
    short_record_policy: str = "pad",
    return_metadata: bool = False,
):
    """Load the clean binary MI/Normal Chapman-Shaoxing cohort.

    Parameters
    ----------
    short_record_policy:
        ``"pad"`` preserves the historical external-validation behaviour.
        ``"skip"`` is recommended for the reviewer in-distribution experiment
        and records every exclusion in the returned log.
    return_metadata:
        When true, return ``(X, y, manifest, exclusions)``.
    """
    records_file = os.path.join(data_dir, "RECORDS")
    if not os.path.exists(records_file):
        empty_x = np.empty((0, target_samples, len(STANDARD_LEADS)), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)
        if return_metadata:
            return empty_x, empty_y, pd.DataFrame(), pd.DataFrame(
                [{"record": None, "reason": f"missing_records_file:{records_file}"}]
            )
        return empty_x, empty_y

    with open(records_file, encoding="utf-8") as handle:
        record_names = [line.strip() for line in handle if line.strip()]
    if max_records is not None:
        record_names = record_names[:max_records]

    signals: list[np.ndarray] = []
    labels: list[int] = []
    manifest_rows: list[dict] = []
    exclusion_rows: list[dict] = []

    for record_name in record_names:
        try:
            record = wfdb.rdrecord(os.path.join(data_dir, record_name))
            comments = record.comments or []
            label, label_reason = _parse_diagnosis(comments)
            if label is None:
                exclusion_rows.append({"record": record_name, "reason": label_reason})
                continue

            signal = np.asarray(record.p_signal, dtype=np.float32)
            signal = _reorder_leads(signal, record.sig_name)
            signal = _prepare_signal(
                signal,
                float(record.fs),
                target_frequency=target_frequency,
                target_samples=target_samples,
                short_record_policy=short_record_policy,
            )
            group_id, group_source = _parse_group_id(record_name, comments)
            sex = _comment_value(comments, ["Sex", "Gender"]) or "Unknown"

            array_index = len(signals)
            signals.append(signal)
            labels.append(int(label))
            manifest_rows.append(
                {
                    "array_index": array_index,
                    "record_id": record_name,
                    "group_id": group_id,
                    "group_id_source": group_source,
                    "label": int(label),
                    "class_name": "MI" if label == 1 else "Normal",
                    "age": _parse_age(comments),
                    "sex": sex,
                    "source_frequency_hz": float(record.fs),
                    "source_samples": int(record.sig_len),
                    "target_frequency_hz": int(target_frequency),
                    "target_samples": int(target_samples),
                }
            )
        except Exception as exc:
            exclusion_rows.append(
                {"record": record_name, "reason": f"processing_error:{type(exc).__name__}:{exc}"}
            )

    if signals:
        x = np.stack(signals).astype(np.float32, copy=False)
        y = np.asarray(labels, dtype=np.int64)
    else:
        x = np.empty((0, target_samples, len(STANDARD_LEADS)), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)

    manifest = pd.DataFrame(manifest_rows)
    exclusions = pd.DataFrame(exclusion_rows)
    print(
        f"Chapman clean cohort: n={len(y)}, MI={int(y.sum()) if len(y) else 0}, "
        f"Normal={int(len(y) - y.sum()) if len(y) else 0}, exclusions={len(exclusions)}"
    )
    if return_metadata:
        return x, y, manifest, exclusions
    return x, y


if __name__ == "__main__":
    X, y, metadata, excluded = load_chapman_data(max_records=500, return_metadata=True)
    print(f"X={X.shape}, y={y.shape}, metadata={metadata.shape}, exclusions={excluded.shape}")
