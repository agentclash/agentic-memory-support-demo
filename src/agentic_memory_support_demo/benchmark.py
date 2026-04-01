from __future__ import annotations

from dataclasses import dataclass

from .chatbot import SupportChatbot
from .deterministic import DeterministicSupportLLM


@dataclass(frozen=True)
class Scenario:
    name: str
    bucket: str
    setup_turns: list[str]
    question: str
    expected_substrings: list[str]


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    mode: str
    answer: str
    passed: bool


SCENARIOS = [
    Scenario(
        name="control_name_same_turn",
        bucket="current_turn",
        setup_turns=[],
        question="My name is Priya. What is my name?",
        expected_substrings=["priya"],
    ),
    Scenario(
        name="control_plan_same_turn",
        bucket="current_turn",
        setup_turns=[],
        question="I'm on the enterprise plan. What plan am I on?",
        expected_substrings=["enterprise"],
    ),
    Scenario(
        name="control_theme_same_turn",
        bucket="current_turn",
        setup_turns=[],
        question="I prefer the dark theme. Which theme do I prefer?",
        expected_substrings=["dark"],
    ),
    Scenario(
        name="control_issue_same_turn",
        bucket="current_turn",
        setup_turns=[],
        question="My issue is a login loop after SSO rotation. What issue am I reporting?",
        expected_substrings=["login loop", "sso"],
    ),
    Scenario(
        name="profile_name_cross_turn",
        bucket="profile_recall",
        setup_turns=["My name is Priya."],
        question="What is my name?",
        expected_substrings=["priya"],
    ),
    Scenario(
        name="profile_plan_cross_turn",
        bucket="profile_recall",
        setup_turns=["I'm on the enterprise plan."],
        question="What plan am I on?",
        expected_substrings=["enterprise"],
    ),
    Scenario(
        name="profile_timezone_cross_turn",
        bucket="profile_recall",
        setup_turns=["My timezone is UTC+5:30."],
        question="What is my timezone?",
        expected_substrings=["utc+5:30"],
    ),
    Scenario(
        name="profile_device_cross_turn",
        bucket="profile_recall",
        setup_turns=["I use a MacBook Pro."],
        question="What laptop am I using?",
        expected_substrings=["macbook pro"],
    ),
    Scenario(
        name="continuity_clear_cache",
        bucket="troubleshooting_continuity",
        setup_turns=["I already tried clearing cache and resetting my password."],
        question="What have we already tried?",
        expected_substrings=["clearing cache", "resetting my password"],
    ),
    Scenario(
        name="continuity_dns",
        bucket="troubleshooting_continuity",
        setup_turns=["I already tried flushing DNS and restarting the router."],
        question="What have we already tried?",
        expected_substrings=["flushing dns", "restarting the router"],
    ),
    Scenario(
        name="procedure_login_loop",
        bucket="procedure_retrieval",
        setup_turns=[],
        question="How should we troubleshoot a login loop after SSO changes?",
        expected_substrings=["clear cookies", "incognito", "re-authenticate"],
    ),
    Scenario(
        name="procedure_webhook_failures",
        bucket="procedure_retrieval",
        setup_turns=[],
        question="What should we do next for webhook delivery failures after a billing deployment?",
        expected_substrings=["deployment timestamp", "signature", "replay"],
    ),
]


def run_benchmark() -> dict[str, object]:
    baseline_results = [_run_scenario(scenario, enable_memory=False) for scenario in SCENARIOS]
    memory_results = [_run_scenario(scenario, enable_memory=True) for scenario in SCENARIOS]
    summary = {
        "baseline": _summarize(baseline_results),
        "memory_enabled": _summarize(memory_results),
        "baseline_results": baseline_results,
        "memory_results": memory_results,
    }
    return summary


def _run_scenario(scenario: Scenario, *, enable_memory: bool) -> ScenarioResult:
    bot = SupportChatbot(llm=DeterministicSupportLLM(), enable_memory=enable_memory, session_id=scenario.name)
    for turn in scenario.setup_turns:
        bot.reply(turn)
    response = bot.reply(scenario.question)
    answer = response.text.lower()
    passed = all(expected.lower() in answer for expected in scenario.expected_substrings)
    return ScenarioResult(
        scenario=scenario,
        mode="memory_enabled" if enable_memory else "baseline",
        answer=response.text,
        passed=passed,
    )


def _summarize(results: list[ScenarioResult]) -> dict[str, object]:
    by_bucket: dict[str, dict[str, int | float]] = {}
    for bucket in sorted({result.scenario.bucket for result in results}):
        bucket_results = [result for result in results if result.scenario.bucket == bucket]
        passed = sum(1 for result in bucket_results if result.passed)
        total = len(bucket_results)
        by_bucket[bucket] = {
            "passed": passed,
            "total": total,
            "accuracy": round((passed / total) * 100, 1),
        }

    total_passed = sum(1 for result in results if result.passed)
    total = len(results)
    return {
        "passed": total_passed,
        "total": total,
        "accuracy": round((total_passed / total) * 100, 1),
        "by_bucket": by_bucket,
    }


def main() -> None:
    summary = run_benchmark()
    for mode in ("baseline", "memory_enabled"):
        report = summary[mode]
        print(f"\n{mode}")
        print(f"  overall: {report['passed']}/{report['total']} ({report['accuracy']}%)")
        for bucket, bucket_report in report["by_bucket"].items():
            print(
                f"  {bucket}: {bucket_report['passed']}/{bucket_report['total']} "
                f"({bucket_report['accuracy']}%)"
            )


if __name__ == "__main__":
    main()
