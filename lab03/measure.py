from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    samples = []

    bench.workload.synchronize()

    for _ in range(repeats):
        start = bench.clock()

        bench.workload.run()
        bench.workload.synchronize()

        end = bench.clock()

        elapsed_ms = (end - start) / 1_000_000.0
        samples.append(elapsed_ms)

    return samples


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 4:
        return unknown("samples", "too few samples to determine warm-up")

    second_half = samples[len(samples) // 2:]
    settled_median = statistics.median(second_half)

    if settled_median <= 0:
        return unknown("samples", "settled median is not positive")

    threshold = settled_median * (1 + WARMUP_TOL)

    discarded = 0

    for sample in samples:
        if sample > threshold:
            discarded += 1
        else:
            break

    return measured(
        discarded,
        f"leading prefix above (1 + {WARMUP_TOL}) x median of the run's second half",
        settled_rate_ms=round(settled_median, 4),
        threshold_ms=round(threshold, 4),
        tolerance=WARMUP_TOL,
        retained=len(samples) - discarded,
    )



def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }

    sorted_samples = sorted(samples)
    n = len(sorted_samples)

    mean = statistics.fmean(sorted_samples)
    minimum = min(sorted_samples)
    maximum = max(sorted_samples)

    if n > 2:
        std = statistics.stdev(sorted_samples)
    else:
        std = 0.0

    def percentile(q: int) -> float:
        h = (n - 1) * q / 100
        i = int(h)

        if i == n - 1:
            return sorted_samples[i]

        return sorted_samples[i] + (h - i) * (
            sorted_samples[i + 1] - sorted_samples[i]
        )

    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(minimum, 4),
        "max": round(maximum, 4),
        "p50": round(percentile(50), 4),
        "p95": round(percentile(95), 4),
        "p99": round(percentile(99), 4),
    }

def is_multimodal(samples: list[float]) -> dict[str, Any]:
    if len(samples) < MIN_SAMPLES_FOR_MODALITY:
        return unknown("samples", "not enough samples to check for multiple modes")

    sorted_samples = sorted(samples)

    # Remove the lowest 5% and highest 5%
    trim_count = int(len(sorted_samples) * 0.05)
    trimmed = sorted_samples[trim_count:-trim_count]

    # Find the gaps between neighboring measurements
    gaps = []

    for i in range(len(trimmed) - 1):
        gap = trimmed[i + 1] - trimmed[i]
        gaps.append(gap)

    typical_gap = statistics.median(gaps)

    if typical_gap <= 0:
        return unknown("samples", "timer resolution is too coarse")

    widest_gap = max(gaps)
    gap_ratio = widest_gap / typical_gap

    # Find where the widest gap occurs
    split_index = gaps.index(widest_gap)

    # Use the gap to separate the original samples into two groups
    split_value = (
        trimmed[split_index] + trimmed[split_index + 1]
    ) / 2

    left = [x for x in sorted_samples if x <= split_value]
    right = [x for x in sorted_samples if x > split_value]

    left_share = len(left) / len(samples)
    right_share = len(right) / len(samples)

    multimodal = ( # true only if all 3 conditions are true
        gap_ratio >= MULTIMODAL_GAP_RATIO
        and left_share >= MIN_MODE_FRACTION
        and right_share >= MIN_MODE_FRACTION
    )

    return measured(
        multimodal,
        f"widest trimmed gap >= {MULTIMODAL_GAP_RATIO}x the median gap, "
        f"with >= {MIN_MODE_FRACTION * 100:.0f}% of samples on each side",
        gap_ratio=round(gap_ratio, 2),
        widest_gap_ms=round(widest_gap, 5),
        typical_gap_ms=round(typical_gap, 5),
        modes=[
            {
                "n": len(left),
                "share": round(left_share, 2),
                "median_ms": round(statistics.median(left), 4),
            },
            {
                "n": len(right),
                "share": round(right_share, 2),
                "median_ms": round(statistics.median(right), 4),
            },
        ],
    )

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    result = bench.runner(["nvpmodel", "-q"])

    # Check whether the command itself worked
    if not result.ok:
        return unknown(result.source, result.error)

    if result.returncode != 0:
        return unknown(
            result.source,
            f"command returned exit code {result.returncode}"
        )

    # Find the power mode
    lines = result.stdout.splitlines()

    mode_name = None
    mode_index = None

    for i, line in enumerate(lines):
        if "NV Power Mode:" in line:
            mode_name = line.split("NV Power Mode:", 1)[1].strip()

            if i + 1 < len(lines):
                try:
                    mode_index = int(lines[i + 1].strip())
                except ValueError:
                    pass

            break

    if mode_name is None or mode_index is None:
        return unknown(result.source, "could not parse NVIDIA power mode")

    # Read CPU frequency limits
    min_freq = read_text(bench.telemetry, CPUFREQ_MIN)
    max_freq = read_text(bench.telemetry, CPUFREQ_MAX)

    clock_source = f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}"

    if min_freq is None or max_freq is None:
        jetson_clocks = None
        jetson_clocks_source = unknown(
            clock_source,
            "could not read CPU frequency limits"
        )
    else:
        jetson_clocks = min_freq == max_freq

        jetson_clocks_source = measured(
            f"scaling_min_freq={min_freq}, scaling_max_freq={max_freq}",
            clock_source
        )

    return measured(
        mode_name,
        result.source,
        mode_index=mode_index,
        jetson_clocks=jetson_clocks,
        jetson_clocks_source=jetson_clocks_source,
    )



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    # -------------------------
    # Temperature
    # -------------------------
    temperatures = []

    thermal_root = bench.telemetry / THERMAL_ZONES

    for zone in thermal_root.glob("thermal_zone*"):
        try:
            temp_text = read_text(
                bench.telemetry,
                f"{THERMAL_ZONES}/{zone.name}/temp"
            )
        except TypeError:
            continue

        try:
            zone_name = read_text(
                bench.telemetry,
                f"{THERMAL_ZONES}/{zone.name}/type"
            )
        except TypeError:
            zone_name = zone.name

        if temp_text is None:
            continue

        try:
            raw_temp = int(temp_text)
        except ValueError:
            continue

        if raw_temp <= -1000:
            continue

        temp_c = raw_temp / 1000.0
        temperatures.append((temp_c, zone_name))

    if temperatures:
        hottest_temp, hottest_zone = max(
            temperatures,
            key=lambda item: item[0]
        )

        temperature = measured(
            round(hottest_temp, 2),
            "sys/devices/virtual/thermal/*/temp",
            zone=hottest_zone,
            zones_read=len(temperatures),
        )
    else:
        temperature = unknown(
            "sys/devices/virtual/thermal/*/temp",
            "no readable thermal zones"
        )

    # -------------------------
    # Power
    # -------------------------
    power_result = read_first(
        bench.telemetry,
        POWER_RAIL_CANDIDATES
    )

    if power_result is None:
        power = unknown(
            " | ".join(POWER_RAIL_CANDIDATES),
            "none of the documented INA3221 rail paths could be read"
        )
    else:
        power_path, power_text = power_result

        try:
            power_value = int(power_text)
            power = measured(power_value, power_path)
        except ValueError:
            power = unknown(
                power_path,
                "power value could not be parsed as an integer"
            )

    # -------------------------
    # GPU utilization
    # -------------------------
    gpu_result = read_first(
        bench.telemetry,
        GPU_LOAD_CANDIDATES
    )

    if gpu_result is None:
        gpu_utilization = unknown(
            " | ".join(GPU_LOAD_CANDIDATES),
            "none of the documented GPU load paths could be read"
        )
    else:
        gpu_path, gpu_text = gpu_result

        try:
            gpu_value = int(gpu_text) / 10.0

            gpu_utilization = measured(
                gpu_value,
                gpu_path,
                units="per-mille / 10"
            )
        except ValueError:
            gpu_utilization = unknown(
                gpu_path,
                "GPU load could not be parsed as an integer"
            )

    return {
        "temperature_c": temperature,
        "power_mw": power,
        "gpu_utilization_percent": gpu_utilization,
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)