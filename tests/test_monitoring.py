"""System resource monitoring: informative, never crashing, never faking."""

from __future__ import annotations

import os

from tmai.monitoring import flat_system_metrics, system_metrics


class TestSystemMetrics:
    def test_returns_expected_keys(self):
        metrics = system_metrics()
        for key in (
            "python",
            "platform",
            "cpu_count",
            "load_average",
            "memory_total_bytes",
            "memory_available_bytes",
            "memory_used_fraction",
            "process_memory_bytes",
            "disk_free_bytes",
            "disk_total_bytes",
        ):
            assert key in metrics, f"missing key {key}"

    def test_cpu_count_matches_os(self):
        assert system_metrics()["cpu_count"] == os.cpu_count()

    def test_memory_fraction_is_between_zero_and_one(self):
        metrics = system_metrics()
        if metrics["memory_used_fraction"] is not None:
            assert 0.0 <= metrics["memory_used_fraction"] <= 1.0

    def test_process_memory_is_positive(self):
        assert system_metrics()["process_memory_bytes"] > 0

    def test_disk_usage_matches_shutil(self):
        import shutil

        metrics = system_metrics()
        usage = shutil.disk_usage(os.getcwd())
        assert metrics["disk_free_bytes"] == usage.free
        assert metrics["disk_total_bytes"] == usage.total

    def test_load_average_has_three_entries_on_linux(self):
        if hasattr(os, "getloadavg"):
            assert len(system_metrics()["load_average"]) == 3

    def test_no_cuda_keys_without_gpu(self):
        metrics = system_metrics()
        # On a machine without CUDA these keys are absent, not zero -- a missing counter
        # must not be reported as a fake 0.
        if not metrics.get("cuda_device_count"):
            assert "cuda_memory_allocated" not in metrics


class TestFlatSystemMetrics:
    def test_only_scalars_are_flattened(self):
        flat = flat_system_metrics()
        assert flat, "expected at least some scalar metrics"
        for key, value in flat.items():
            assert key.startswith("system/")
            assert isinstance(value, float)
            assert value == value  # not NaN

    def test_load_average_is_flattened_per_entry(self):
        flat = flat_system_metrics()
        if hasattr(os, "getloadavg"):
            assert "system/load1" in flat
            assert "system/load2" in flat
            assert "system/load3" in flat

    def test_strings_and_lists_are_skipped(self):
        flat = flat_system_metrics()
        assert "system/python" not in flat  # a string, not a number
        assert "system/load_average" not in flat  # a list, flattened as load1..3 instead

    def test_custom_prefix(self):
        flat = flat_system_metrics(prefix="resources")
        assert all(key.startswith("resources/") for key in flat)
