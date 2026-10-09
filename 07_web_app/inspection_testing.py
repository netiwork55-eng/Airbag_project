"""Honest metrics for labelled, same-image model comparisons."""
import math
from statistics import median


def summarize_test(rows):
    # Development images and unknown labels remain visible, not independent accuracy.
    eligible = [r for r in rows if r.get("overlap_check_available") and not r.get("development_image")]
    good = [r for r in eligible if r["expected"] == "OK"]
    bad = [r for r in eligible if r["expected"] == "NG"]
    count = lambda items, status: sum(r.get("decision") == status for r in items)
    times = sorted(r["test_ms"] for r in rows if isinstance(r.get("test_ms"), (int, float)) and math.isfinite(r["test_ms"]))
    return {"total": len(rows), "labelled_good": len(good), "labelled_bad": len(bad),
        "development_images": sum(bool(r.get("development_image")) for r in rows),
        "unknown_labels": sum(r["expected"] == "UNKNOWN" for r in rows),
        "unchecked_overlap": sum(not r.get("overlap_check_available") for r in rows),
        "decisions": {key: count(rows, key) for key in ("OK", "NG", "REVIEW", "NO_PART", "ERROR")},
        "good_acceptance": count(good, "OK") / len(good) if good else None,
        "false_reject": count(good, "NG") / len(good) if good else None,
        "defect_detection": count(bad, "NG") / len(bad) if bad else None,
        "defect_miss": count(bad, "OK") / len(bad) if bad else None,
        "good_unresolved": len(good) - count(good, "OK") - count(good, "NG"),
        "bad_unresolved": len(bad) - count(bad, "OK") - count(bad, "NG"),
        "median_ms": median(times) if times else None,
        "limits": "User labels; file-hash overlap check cannot prove separate physical parts or capture sessions. No defect-location metric without verified regions."}
