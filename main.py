"""Local entry point for the source-specialized regulatory multi-agent graph."""

from regulatory_agent.cli import (
    create_multi_agent,
    create_question_router,
    create_research_coordinator,
    main,
)

__all__ = [
    "create_multi_agent",
    "create_question_router",
    "create_research_coordinator",
    "main",
]


if __name__ == "__main__":
    main()
