
from __future__ import annotations

from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    f = float(v)
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    samples = []

    for i, value in enumerate(ms): #1 validation and ingestion
        t = _finite(value, f"ms[{i}]")

        if t < 0:
            raise AgreementError("Latency cannot be negative.")

        samples.append(t)

    if not samples:
        raise AgreementError("No latency samples provided.") #2 non-empty guard

    n = len(samples)
    mean = sum(samples) / n
    worst = max(samples) #3 summary metrics, worst is max because higher latency is bad

    percentiles = {}

    for q in LATENCY_PERCENTILES:
        percentiles[int(q)] = _percentile(samples, q)

    p50 = percentiles[50]
    p95 = percentiles[95]
    p99 = percentiles[99] #4 percentile computation

    jitter = p95 - p50 #5 jitter calc

    return {
        "n": n,
        "mean": mean,
        "worst": worst,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "jitter": jitter,
    }




def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    base = _finite(base_ms, "base_ms")
    other = _finite(other_ms, "other_ms")

    if base <= 0 or other <= 0:
        raise AgreementError("Execution times must be positive.")

    factor = base / other

    faster = factor > 1.0

    saved_ms = base - other

    return {
        "factor": factor,
        "faster": faster,
        "saved_ms": saved_ms,
    }



#Max Absolute Deviation
def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    if not a or not b:
        raise AgreementError("Output sequences cannot be empty.")

    if len(a) != len(b):
        raise AgreementError("Output sequences must have matching lengths.")

    a_values = []
    b_values = []

    for i in range(len(a)):
        a_values.append(_finite(a[i], f"a[{i}]"))
        b_values.append(_finite(b[i], f"b[{i}]"))

    best_i = 0
    best = abs(a_values[0] - b_values[0])

    for i in range(1, len(a_values)):
        diff = abs(a_values[i] - b_values[i])

        if diff > best:
            best = diff
            best_i = i

    return {
        "max_abs": best,
        "index": best_i, #index is first max_abs if there are 2 instances of same difference
        "a": a_values[best_i],
        "b": b_values[best_i],
        "n": len(a_values),
    }

#Max Relative Deviation
def max_rel_dev(a: Sequence[float], b: Sequence[float], 
                floor: float = REL_FLOOR) -> dict[str, Any]:

    floor_value = _finite(floor, "floor")

    if floor_value <= 0:
        raise AgreementError("Floor must be positive.")

    if not a or not b:
        raise AgreementError("Output sequences cannot be empty.")

    if len(a) != len(b):
        raise AgreementError("Output sequences must have matching lengths.")

    a_values = []
    b_values = []

    for i in range(len(a)):
        a_values.append(_finite(a[i], f"a[{i}]"))
        b_values.append(_finite(b[i], f"b[{i}]"))

    best = -1.0 #neg so rel>best is true everytime
    best_i = 0
    floored = 0
    denom_min = abs(b_values[0])

    for i in range(len(a_values)): #2 state tracking
        d = abs(b_values[i])
        denom_min = min(denom_min, d)

        if d < floor_value:
            floored += 1

        rel = abs(a_values[i] - b_values[i]) / max(d, floor_value)

        if rel > best:
            best = rel
            best_i = i

    return {
        "max_rel": best,
        "index": best_i,
        "floored": floored, #floored: Counts how many reference values require the floor, smaller than our floor
        "denom_min": denom_min, #smallest original absolute reference value
        "n": len(a_values),
    }


#top1_agreement performs instance-by-instance argmax comparisons to verify that both backends predict the identical top-ranking category
def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:

    if not rows_a or not rows_b:
        raise AgreementError("Score rows cannot be empty.")

    if len(rows_a) != len(rows_b):
        raise AgreementError("Row counts must match.")

    n = len(rows_a)
    agree = 0
    first_disagreement = None

    for k in range(n): #process 1 image at a time
        row_a = rows_a[k]
        row_b = rows_b[k]

        if not row_a or not row_b: #row validations
            raise AgreementError("Score vectors cannot be empty.")

        if len(row_a) != len(row_b):
            raise AgreementError("Class counts must match.") 

        
        a_values = []
        b_values = []

        for i in range(len(row_a)): #validate and collecting every class score for image
            a_values.append(_finite(row_a[i], f"rows_a[{k}][{i}]")) #k identifies which input image we're processing
            b_values.append(_finite(row_b[i], f"rows_b[{k}][{i}]")) #i identifies which class score we're checking

        pred_a = a_values.index(max(a_values))
        pred_b = b_values.index(max(b_values))

        if pred_a == pred_b: #comparing predictions
            agree += 1
        elif first_disagreement is None:
            first_disagreement = k

    disagree = n - agree
    rate = agree / n

    return {
        "n": n,
        "agree": agree,
        "disagree": disagree,
        "rate": rate,
        "first_disagreement": first_disagreement,
    }



#decision margin: how close the highest-scoring class is to the second-highest-scoring class

#tol: numerical deviation tolerance
def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:

    tol_value = _finite(tol, "tol")

    if tol_value < 0:
        raise AgreementError("Tolerance cannot be negative.")

    if not rows:
        raise AgreementError("Score rows cannot be empty.")

    margins = []

    for k, row in enumerate(rows):
        if len(row) < 2:
            raise AgreementError("Each row must have at least two classes.") #at least 2 to compare

        scores = []

        for i, value in enumerate(row):
            scores.append(_finite(value, f"rows[{k}][{i}]"))

        scores.sort(reverse=True) #sort from largest to smallest

        margin = scores[0] - scores[1] #largest - 2nd largest
        margins.append(margin)

    min_margin = min(margins)
    argmin = margins.index(min_margin) #automatically selects first index if tie

    median = _percentile(margins, 50)
    mean = sum(margins) / len(margins)

    below_tol = 0

    for margin in margins:
        if margin < tol_value:
            below_tol += 1

    safe = below_tol == 0 #true if no margins below tol, false if there are margins below tol

    return {
        "n": len(margins),
        "min": min_margin,
        "argmin": argmin,
        "median": median,
        "mean": mean,
        "below_tol": below_tol,
        "safe": safe,
    }



#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()