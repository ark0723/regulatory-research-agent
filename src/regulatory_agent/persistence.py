"""Persistence composition and lifecycle for application and graph storage."""

from __future__ import annotations


from dotenv import load_dotenv

load_dotenv()

import argparse
from contextlib import contextmanager
import os
from typing import Iterator, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection

from regulatory_agent.cli import create_multi_agent
from regulatory_agent.conversation_service import (
    ConversationService,
    InMemoryConversationRepository,
)
from regulatory_agent.multi_agent import (
    EvaluationRetryPolicy,
    create_checkpoint_serializer,
)
from regulatory_agent.postgres_repository import (
    CHECKPOINT_SCHEMA,
    APPLICATION_SCHEMA,
    PersistenceInitializationError,
    PostgresConversationRepository,
    create_connection_pool,
)
from regulatory_agent.routing import RouteTarget
from regulatory_agent.vector_db import database_url


StorageMode = Literal["memory", "postgres"]


def postgres_connection_string() -> str:
    """Return an explicit psycopg URI without logging its password."""

    configured = os.getenv("REGULATORY_APP_DATABASE_URL")
    if configured:
        return configured
    url = database_url().set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


def _configure_checkpoint_connection(conn: Connection) -> None:
    conn.execute(f"SET search_path TO {CHECKPOINT_SCHEMA}, public")


def setup_postgres_persistence(conninfo: str | None = None) -> None:
    """Apply application and official LangGraph checkpoint migrations."""

    selected = conninfo or postgres_connection_string()
    app_pool = create_connection_pool(selected, min_size=1, max_size=2)
    checkpoint_pool = None
    try:
        repository = PostgresConversationRepository(app_pool)
        repository.setup()
        with app_pool.connection() as conn:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {CHECKPOINT_SCHEMA}")
        checkpoint_pool = create_connection_pool(
            selected,
            min_size=1,
            max_size=2,
            configure=_configure_checkpoint_connection,
        )
        checkpointer = PostgresSaver(
            checkpoint_pool,
            serde=create_checkpoint_serializer(),
        )
        checkpointer.setup()
    except PersistenceInitializationError:
        raise
    except Exception as error:
        raise PersistenceInitializationError(
            f"persistence setup failed: {error}"
        ) from error
    finally:
        if checkpoint_pool is not None:
            checkpoint_pool.close()
        app_pool.close()


@contextmanager
def open_conversation_service(
    *,
    storage_mode: StorageMode | None = None,
    provider: str | None = None,
    model: str | None = None,
    human_review_policy: Literal[
        "never", "on_reviewer_request", "always"
    ] = "on_reviewer_request",
    retry_policy: EvaluationRetryPolicy | None = None,
    available_research_targets: set[RouteTarget] | None = None,
    conninfo: str | None = None,
    initialize: bool = False,
    worker_id: str | None = None,
    review_claim_ttl_seconds: int = 300,
) -> Iterator[ConversationService]:
    """Open one service and keep its pools/checkpointer alive for its lifetime."""

    mode = storage_mode or os.getenv("REGULATORY_STORAGE_MODE", "memory")
    if mode == "memory":
        checkpointer = InMemorySaver(serde=create_checkpoint_serializer())
        agent = create_multi_agent(
            provider=provider,
            model=model,
            human_review_policy=human_review_policy,
            retry_policy=retry_policy,
            available_research_targets=available_research_targets,
            checkpointer=checkpointer,
        )
        yield ConversationService(
            agent=agent,
            repository=InMemoryConversationRepository(),
            worker_id=worker_id,
            review_claim_ttl_seconds=review_claim_ttl_seconds,
        )
        return
    if mode != "postgres":
        raise ValueError("storage_mode must be 'memory' or 'postgres'")

    selected = conninfo or postgres_connection_string()
    if initialize:
        setup_postgres_persistence(selected)
    app_pool = create_connection_pool(selected)
    checkpoint_pool = None
    try:
        repository = PostgresConversationRepository(app_pool)
        try:
            repository.list_conversations()
        except Exception as error:
            raise PersistenceInitializationError(
                "application schema is unavailable; run persistence setup"
            ) from error
        checkpoint_pool = create_connection_pool(
            selected,
            configure=_configure_checkpoint_connection,
        )
        checkpointer = PostgresSaver(
            checkpoint_pool,
            serde=create_checkpoint_serializer(),
        )
        try:
            list(checkpointer.list(None, limit=1))
        except Exception as error:
            raise PersistenceInitializationError(
                "checkpoint schema is unavailable; run persistence setup"
            ) from error
        agent = create_multi_agent(
            provider=provider,
            model=model,
            human_review_policy=human_review_policy,
            retry_policy=retry_policy,
            available_research_targets=available_research_targets,
            checkpointer=checkpointer,
        )
        yield ConversationService(
            agent=agent,
            repository=repository,
            worker_id=worker_id,
            review_claim_ttl_seconds=review_claim_ttl_seconds,
        )
    finally:
        if checkpoint_pool is not None:
            checkpoint_pool.close()
        app_pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage persistence schemas.")
    parser.add_argument("command", choices=("setup",))
    args = parser.parse_args()
    if args.command == "setup":
        setup_postgres_persistence()
        print(
            f"Initialized {APPLICATION_SCHEMA} and {CHECKPOINT_SCHEMA} schemas."
        )


if __name__ == "__main__":
    main()
