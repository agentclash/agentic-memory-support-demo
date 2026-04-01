from agentic_memory_support_demo.benchmark import run_benchmark


def test_deterministic_benchmark_shows_memory_lift():
    summary = run_benchmark()

    baseline = summary["baseline"]
    memory_enabled = summary["memory_enabled"]

    assert baseline["accuracy"] == 33.3
    assert memory_enabled["accuracy"] == 100.0

    assert baseline["by_bucket"]["current_turn"]["accuracy"] == 100.0
    assert baseline["by_bucket"]["profile_recall"]["accuracy"] == 0.0
    assert baseline["by_bucket"]["troubleshooting_continuity"]["accuracy"] == 0.0
    assert baseline["by_bucket"]["procedure_retrieval"]["accuracy"] == 0.0

    assert memory_enabled["by_bucket"]["current_turn"]["accuracy"] == 100.0
    assert memory_enabled["by_bucket"]["profile_recall"]["accuracy"] == 100.0
    assert memory_enabled["by_bucket"]["troubleshooting_continuity"]["accuracy"] == 100.0
    assert memory_enabled["by_bucket"]["procedure_retrieval"]["accuracy"] == 100.0
