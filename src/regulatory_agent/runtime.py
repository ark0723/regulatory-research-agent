"""Runtime factories shared by CLI and composition examples."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from ollama import Client

from regulatory_agent.vector_db import (
    create_vector_store,
    database_url,
    make_embeddings,
    read_model_profile,
)


def create_chat_model(
    *,
    provider: str | None = None,
    model: str | None = None,
) -> BaseChatModel:
    """Create the configured deterministic chat model."""

    selected_provider = (
        provider or os.getenv("REGULATORY_CHAT_PROVIDER") or "ollama"
    ).lower()
    selected_model = model or os.getenv("REGULATORY_CHAT_MODEL")
    if not selected_model:
        raise ValueError(
            "No chat model configured. Set REGULATORY_CHAT_MODEL or pass --model."
        )

    if selected_provider == "ollama":
        return ChatOllama(
            model=selected_model,
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=0,
        )
    if selected_provider == "groq":
        return ChatGroq(model=selected_model, temperature=0)
    if selected_provider == "openai":
        return ChatOpenAI(model=selected_model, temperature=0)
    raise ValueError(
        "REGULATORY_CHAT_PROVIDER must be one of: ollama, groq, openai"
    )


def create_existing_vector_store() -> PGVector:
    """Connect to the existing PGVector collection without reindexing it."""

    base_url = os.getenv("OLLAMA_BASE_URL")
    client = Client(host=base_url)
    profile = read_model_profile(client)
    embeddings = make_embeddings(profile=profile, base_url=base_url)
    return create_vector_store(
        connection=database_url(),
        embeddings=embeddings,
        embedding_length=profile.embedding_length,
    )
