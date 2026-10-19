
from uuid import uuid4
from dotenv import load_dotenv
from regulatory_agent.cli import create_multi_agent
from regulatory_agent.multi_agent import EvaluationRetryPolicy
from time import perf_counter

load_dotenv()


# human_review_policy = [on_reviewer_request, always, never]
# on_reviewer_request: Evaluation 결정이 human_review일 때만 중단
# always: 정상 생성된 모든 답변에 사람 승인 요구
#never: HITL 비활성화
agent = create_multi_agent(
    provider="groq",
    model="openai/gpt-oss-120b",
    # pass/revise/research_again의 실제 라우팅을 확인하려면 always가 아니라
    # on_reviewer_request를 사용해야 합니다.
    human_review_policy="on_reviewer_request",
    # Phase 3B 테스트에서 적용되는 한도를 출력과 함께 명시적으로 확인합니다.
    retry_policy=EvaluationRetryPolicy(
        max_total_automatic_retries=3,
        max_answer_revisions=2,
        max_research_retries=1,
        max_research_queries_per_plan=3,
        max_merged_synthesis_sources=20,
    ),
    checkpointer=None,  # 기본 InMemorySaver 사용
)

start = perf_counter()

result = agent.ask(
    "What must informed consent contain?",
    thread_id=f"review-case-{uuid4()}",
)

answer_elapsed = perf_counter() - start

print("종료 사유:", result.termination_reason)
print("답변 생성 시간:", round(answer_elapsed, 2), "초")
print("답변:", result.final_answer)
print("인용:", result.citations)
print("인용 검증:", result.validation.model_dump() if result.validation else None)
print("평가 오류:", result.evaluation_error)
print(
    "평가 이력:",
    [entry.model_dump() for entry in result.evaluation_history],
)

# 실행 결과와 라우팅 정책의 일관성을 즉시 검증합니다.
print("evaluation_error가 None인가:", result.evaluation_error is None)
print("평가 기록 수:", len(result.evaluation_history))
print(
    "재시도 횟수:",
    {
        "total": result.evaluation_retry_count,
        "revision": result.revision_count,
        "research": result.research_retry_count,
    },
)

# 현재 구현에서는 시작된 자동 재시도의 합이 total counter와 같아야 합니다.
assert result.evaluation_retry_count == (
    result.revision_count + result.research_retry_count
)
assert [entry.sequence for entry in result.evaluation_history] == list(
    range(1, len(result.evaluation_history) + 1)
)

if result.evaluation_history:
    print(
        "평가 결정 흐름:",
        " -> ".join(entry.decision for entry in result.evaluation_history),
    )
    print(
        "평가 처리 흐름:",
        " -> ".join(entry.outcome for entry in result.evaluation_history),
    )

    last_evaluation = result.evaluation_history[-1]
    if last_evaluation.decision == "pass":
        assert last_evaluation.outcome == "success"
        assert result.termination_reason == "answered"
        print("라우팅 확인: pass -> answered")

if result.research_retry_plan is not None:
    print("재검색 계획:", result.research_retry_plan.model_dump_json(indent=2))
    print("재검색 실행 횟수:", result.research_retry_count)
    print(
        "재검색 진단:",
        [item.model_dump() for item in result.research_retry_diagnostics],
    )
    assert result.research_retry_count >= 1
    assert any(
        entry.decision == "research_again"
        and entry.outcome == "research_retry_started"
        for entry in result.evaluation_history
    )

    # 한 개의 검증된 plan이 정확히 한 번의 research attempt가 되었는지 확인합니다.
    assert len(result.research_retry_plans) == result.research_retry_count
    assert len(result.research_retry_diagnostics) == result.research_retry_count

    for diagnostic in result.research_retry_diagnostics:
        planned = [
            (query.target, query.query)
            for query in diagnostic.plan.queries
        ]
        executed = [
            (execution.target, execution.query)
            for execution in diagnostic.executions
        ]
        assert executed == planned, (planned, executed)
        assert diagnostic.evidence_after <= 20

        print(
            f"재검색 attempt {diagnostic.attempt}:",
            {
                "outcome": diagnostic.outcome,
                "executions": [
                    {
                        "target": item.target,
                        "query": item.query,
                        "success": item.success,
                        "evidence_count": item.evidence_count,
                        "error": item.error,
                    }
                    for item in diagnostic.executions
                ],
                "evidence_before": diagnostic.evidence_before,
                "evidence_after": diagnostic.evidence_after,
                "duplicates_removed": diagnostic.duplicates_removed,
                "excluded_by_budget": diagnostic.excluded_by_budget,
            },
        )

    # 병합된 synthesis source ID는 S1부터 중복 없이 다시 만들어집니다.
    assert [source.source_id for source in result.sources] == [
        f"S{index}" for index in range(1, len(result.sources) + 1)
    ]

    if result.termination_reason == "answered":
        assert result.validation is not None and result.validation.valid
        assert result.evaluation is not None
        assert result.evaluation.decision == "pass"
        print("Phase 3B 확인: 재검색 → 병합 → 재생성 → 인용 검증 → pass")

if any(
    entry.decision == "revise" and entry.outcome == "revision_started"
    for entry in result.evaluation_history
):
    assert result.revision_count >= 1
    assert result.evaluation_retry_count >= 1
    print(
        "라우팅 확인: revise -> 수정 루프 진입",
        f"(revision_count={result.revision_count})",
    )

if result.termination_reason == "evaluation_failed":
    assert result.evaluation_error
    assert result.evaluation_history[-1].outcome == "evaluation_failed"
    print("평가 실패 원인:", result.evaluation_error)

if result.termination_reason == "research_retry_planning_failed":
    assert result.research_retry_error
    print("재검색 계획 실패 원인:", result.research_retry_error)

if result.termination_reason in {
    "research_retry_partial_failure",
    "research_retry_failed",
    "research_retry_regeneration_failed",
}:
    assert result.research_retry_error
    print("재검색 실행 실패 원인:", result.research_retry_error)


if result.evaluation is not None:
    print("평가 점수:", result.evaluation.total_score)
    print("평가 의견:", result.evaluation.brief_evaluation)

if result.termination_reason == "awaiting_human_review":
    request = result.human_review_request

    print("\n--- 사람의 검토가 필요합니다 ---")
    print(request.final_answer)
    print(request.citations)
    print(request.reason)

    resume_start = perf_counter()

    result = agent.resume_human_review(
        thread_id=result.thread_id,
        approved=True,
        feedback="적용 가능성을 확인했으며 답변을 승인합니다.",
    )

    resume_elapsed = perf_counter() - resume_start

    print("검토 후 종료 사유:", result.termination_reason)
    print("승인 처리 시간:", round(resume_elapsed, 2), "초")
    print(
        "전체 시스템 처리 시간:",
        round(answer_elapsed + resume_elapsed, 2),
        "초",
    )

if result.research_retry_plan is None:
    print(
        "\n이번 실행에서는 evaluator가 research_again을 선택하지 않았습니다. "
        "따라서 일반 E2E 경로만 검증되었습니다. Phase 3B를 확정적으로 검증하려면 "
        "tests.test_multi_agent의 deterministic 테스트를 실행하세요."
    )
