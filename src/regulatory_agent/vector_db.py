"""Non-truncating Ollama/pgvector retrieval pipeline for regulatory chunks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import DistanceStrategy
from ollama import Client, ResponseError
from sqlalchemy.engine import URL

from regulatory_agent.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    build_chunked_documents,
)
from regulatory_agent.paths import DOCS_DIR


EMBEDDING_MODEL = "bge-m3"
COLLECTION_NAME = "regulatory_chunks"
EMBEDDING_AUDIT_PATH = DOCS_DIR / "experiments/embedding_input_audit.json"
DEFAULT_SEARCH_K = 5
DISTANCE_STRATEGY = DistanceStrategy.COSINE


@dataclass(frozen=True)
class ModelProfile:
    model: str
    architecture: str
    context_length: int
    embedding_length: int
    ollama_server_version: str = ""
    ollama_client_version: str = ""
    langchain_ollama_version: str = ""
    runtime_context_length: int | None = None
    model_digest: str = ""


@dataclass(frozen=True)
class EmbeddingAuditRow:
    record_id: str
    document_type: str
    content_type: str
    section_id: str
    character_count: int
    content_sha256: str
    accepted_without_truncation: bool
    token_count: int | None
    max_accepted_prefix_chars: int | None
    max_accepted_prefix_tokens: int | None
    first_rejected_prefix_chars: int | None
    error_status_code: int | None
    error: str


class EmbeddingInputLimitError(ValueError):
    """Raised before indexing when any chunk exceeds the embedding limit."""


class NonTruncatingOllamaEmbeddings(OllamaEmbeddings):
    """Ollama embeddings that reject over-limit documents and queries."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not self._client:
            raise RuntimeError("Ollama sync client is not initialized")
        if self.num_ctx is None:
            raise RuntimeError("num_ctx must be configured explicitly")
        return self._client.embed(
            self.model,
            texts,
            truncate=False,
            dimensions=self.dimensions,
            options=embedding_options(self.num_ctx),
            keep_alive=self.keep_alive,
        )["embeddings"]


def build_index_documents() -> list[Document]:
    """Build the one supported retrieval corpus with fixed chunk settings."""
    documents = build_chunked_documents(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    chunk_ids = [str(document.metadata["chunk_id"]) for document in documents]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk_id values detected")
    return documents


def database_url(environment: Mapping[str, str] | None = None) -> URL:
    """Build the PostgreSQL URL without exposing the password in logs."""
    env = os.environ if environment is None else environment
    required = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ValueError(
            f"Missing PostgreSQL environment variables: {', '.join(missing)}"
        )
    return URL.create(
        drivername="postgresql+psycopg",
        username=env["POSTGRES_USER"],
        password=env["POSTGRES_PASSWORD"],
        host=env.get("POSTGRES_HOST", "localhost"),
        port=int(env.get("POSTGRES_PORT", "5432")),
        database=env["POSTGRES_DB"],
    )


def read_ollama_server_version(base_url: str | None = None) -> str:
    url = f"{(base_url or 'http://127.0.0.1:11434').rstrip('/')}/api/version"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    return str(payload["version"])


def read_model_profile(
    client: Client,
    model: str = EMBEDDING_MODEL,
    *,
    server_version: str = "",
) -> ModelProfile:
    """Read context and vector dimensions from the installed Ollama model."""
    response = client.show(model)
    model_info = dict(response.modelinfo or {})
    architecture = str(model_info.get("general.architecture", ""))
    if not architecture:
        raise ValueError(f"Ollama did not report an architecture for {model!r}")
    try:
        context_length = int(model_info[f"{architecture}.context_length"])
        embedding_length = int(model_info[f"{architecture}.embedding_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Ollama did not report context/embedding dimensions for {model!r}"
        ) from exc
    return ModelProfile(
        model=model,
        architecture=architecture,
        context_length=context_length,
        embedding_length=embedding_length,
        ollama_server_version=server_version,
        ollama_client_version=importlib.metadata.version("ollama"),
        langchain_ollama_version=importlib.metadata.version("langchain-ollama"),
    )


def add_loaded_runtime_details(
    profile: ModelProfile, client: Client
) -> ModelProfile:
    """Record the context actually allocated to the loaded model."""
    for running in client.ps().models:
        running_name = str(running.name or running.model or "")
        if running_name.split(":", 1)[0] != profile.model.split(":", 1)[0]:
            continue
        return replace(
            profile,
            runtime_context_length=int(running.context_length),
            model_digest=str(running.digest or ""),
        )
    return profile


def embedding_options(context_length: int) -> dict[str, int]:
    """Return the model options shared by audit, indexing, and queries."""
    return {"num_ctx": context_length}


def _embed_for_audit(
    client: Client,
    *,
    model: str,
    text: str,
    context_length: int,
) -> Any:
    return client.embed(
        model=model,
        input=text,
        truncate=False,
        options=embedding_options(context_length),
    )


def audit_embedding_inputs(
    documents: Iterable[Document],
    *,
    client: Client,
    profile: ModelProfile,
) -> list[EmbeddingAuditRow]:
    """Verify every chunk through Ollama without ever enabling truncation."""
    rows: list[EmbeddingAuditRow] = []
    for document in documents:
        metadata = document.metadata
        try:
            response = _embed_for_audit(
                client,
                model=profile.model,
                text=document.page_content,
                context_length=profile.context_length,
            )
            accepted = True
            token_count = int(response.prompt_eval_count)
            max_prefix_chars = None
            max_prefix_tokens = None
            first_rejected_prefix_chars = None
            error_status_code = None
            error = ""
        except ResponseError as exc:
            accepted = False
            token_count = None
            max_prefix_chars = None
            max_prefix_tokens = None
            first_rejected_prefix_chars = None
            error_status_code = exc.status_code
            error = str(exc)
            if "exceeds the context length" in error:
                max_prefix_chars, max_prefix_tokens = _find_max_accepted_prefix(
                    client,
                    model=profile.model,
                    text=document.page_content,
                    context_length=profile.context_length,
                )
                first_rejected_prefix_chars = max_prefix_chars + 1

        rows.append(
            EmbeddingAuditRow(
                record_id=str(metadata["record_id"]),
                document_type=str(metadata["document_type"]),
                content_type=str(metadata["content_type"]),
                section_id=str(metadata["section_id"]),
                character_count=len(document.page_content),
                content_sha256=hashlib.sha256(
                    document.page_content.encode("utf-8")
                ).hexdigest(),
                accepted_without_truncation=accepted,
                token_count=token_count,
                max_accepted_prefix_chars=max_prefix_chars,
                max_accepted_prefix_tokens=max_prefix_tokens,
                first_rejected_prefix_chars=first_rejected_prefix_chars,
                error_status_code=error_status_code,
                error=error,
            )
        )
    return rows


def _find_max_accepted_prefix(
    client: Client,
    *,
    model: str,
    text: str,
    context_length: int,
) -> tuple[int, int | None]:
    """Find a rejected input's exact boundary without enabling truncation."""
    low = 0
    high = len(text) - 1
    token_count: int | None = None
    while low < high:
        midpoint = (low + high + 1) // 2
        try:
            response = _embed_for_audit(
                client,
                model=model,
                text=text[:midpoint],
                context_length=context_length,
            )
            low = midpoint
            token_count = int(response.prompt_eval_count)
        except ResponseError:
            high = midpoint - 1
    if low and token_count is None:
        response = _embed_for_audit(
            client,
            model=model,
            text=text[:low],
            context_length=context_length,
        )
        token_count = int(response.prompt_eval_count)
    return low, token_count


def audit_failures(rows: Sequence[EmbeddingAuditRow]) -> list[EmbeddingAuditRow]:
    return [row for row in rows if not row.accepted_without_truncation]


def require_complete_embeddings(rows: Sequence[EmbeddingAuditRow]) -> None:
    """Stop indexing before a vector store is opened when any chunk fails."""
    failures = audit_failures(rows)
    if not failures:
        return
    details = "; ".join(f"{row.record_id}: {row.error}" for row in failures)
    raise EmbeddingInputLimitError(
        f"{len(failures)} embedding input(s) failed; nothing was indexed. {details}"
    )


def write_embedding_audit(
    rows: Sequence[EmbeddingAuditRow],
    *,
    profile: ModelProfile,
    output_path: Path = EMBEDDING_AUDIT_PATH,
) -> None:
    """Write a reproducible report without duplicating source text."""
    failures = audit_failures(rows)
    payload = {
        "model": asdict(profile),
        "pipeline": {
            "collection_name": COLLECTION_NAME,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "distance_strategy": DISTANCE_STRATEGY.value,
            "default_search_k": DEFAULT_SEARCH_K,
            "embedding_request": {
                "truncate": False,
                "options": embedding_options(profile.context_length),
                "applies_to": ["audit", "index_documents", "embed_query"],
            },
        },
        "summary": {
            "document_count": len(rows),
            "accepted_without_truncation": len(rows) - len(failures),
            "failed": len(failures),
        },
        "documents": [asdict(row) for row in rows],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_embeddings(
    *,
    profile: ModelProfile,
    base_url: str | None = None,
) -> NonTruncatingOllamaEmbeddings:
    """Create the strict client shared by indexing and query embedding."""
    return NonTruncatingOllamaEmbeddings(
        model=profile.model,
        base_url=base_url,
        num_ctx=profile.context_length,
    )


def create_vector_store(
    *,
    connection: URL | str,
    embeddings: NonTruncatingOllamaEmbeddings,
    embedding_length: int,
    vector_store_class: type[PGVector] = PGVector,
) -> PGVector:
    """Open the single regulatory chunk collection."""
    connection_string = (
        connection.render_as_string(hide_password=False)
        if isinstance(connection, URL)
        else connection
    )
    return vector_store_class(
        embeddings=embeddings,
        connection=connection_string,
        collection_name=COLLECTION_NAME,
        collection_metadata={
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        },
        embedding_length=embedding_length,
        distance_strategy=DISTANCE_STRATEGY,
        use_jsonb=True,
    )


def vector_document_id(document: Document) -> str:
    """Namespace the stable chunk ID to avoid collisions with old collections."""
    return f"{COLLECTION_NAME}:{document.metadata['chunk_id']}"


def index_documents(store: PGVector, documents: Sequence[Document]) -> list[str]:
    """Upsert chunks with deterministic IDs; reruns update instead of duplicate."""
    ids = [vector_document_id(document) for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate vector document IDs detected")
    store.add_documents(documents=list(documents), ids=ids)
    return ids


def retrieve_with_scores(
    store: PGVector,
    query: str,
    *,
    k: int = DEFAULT_SEARCH_K,
    metadata_filter: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """Run cosine similarity search and retain raw pgvector scores."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    return store.similarity_search_with_score(
        query=query,
        k=k,
        filter=metadata_filter,
    )


def format_retrieval_results(
    results: Sequence[tuple[Document, float]],
    *,
    max_content_chars: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Render content, score, and citation metadata for CLI inspection."""
    blocks: list[str] = []
    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata
        page_range = (
            str(metadata.get("page_start"))
            if metadata.get("page_start") == metadata.get("page_end")
            else f"{metadata.get('page_start')}-{metadata.get('page_end')}"
        )
        content = document.page_content
        if max_content_chars > 0 and len(content) > max_content_chars:
            content = content[:max_content_chars].rstrip() + "…"
        blocks.append(
            "\n".join(
                [
                    f"[{rank}] score={score:.6f}",
                    f"citation: {metadata.get('citation', '')}",
                    f"record_id: {metadata.get('record_id', '')}",
                    f"parent_record_id: {metadata.get('parent_record_id', '')}",
                    f"section_id: {metadata.get('section_id', '')}",
                    f"content_type: {metadata.get('content_type', '')}",
                    f"pages: {page_range}",
                    f"source_url: {metadata.get('source_url', '')}",
                    "content:",
                    content,
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "No results."


def _runtime() -> tuple[str | None, Client, ModelProfile]:
    base_url = os.getenv("OLLAMA_BASE_URL")
    client = Client(host=base_url)
    profile = read_model_profile(
        client,
        server_version=read_ollama_server_version(base_url),
    )
    return base_url, client, profile


def _run_audit(
    *,
    output_path: Path,
) -> tuple[list[Document], list[EmbeddingAuditRow], ModelProfile, str | None]:
    documents = build_index_documents()
    base_url, client, profile = _runtime()
    rows = audit_embedding_inputs(documents, client=client, profile=profile)
    profile = add_loaded_runtime_details(profile, client)
    write_embedding_audit(rows, profile=profile, output_path=output_path)
    return documents, rows, profile, base_url


def _print_audit_summary(
    rows: Sequence[EmbeddingAuditRow], output_path: Path
) -> None:
    failures = audit_failures(rows)
    print(
        f"Embedding audit: {len(rows)} chunks, {len(failures)} failed; "
        f"report={output_path}"
    )
    for row in failures:
        print(
            f"- {row.record_id} ({row.character_count} chars; "
            f"status={row.error_status_code}; error={row.error})"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit, index, and query regulatory chunks in pgvector."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit", help="Validate all chunks without writing to PostgreSQL."
    )
    audit_parser.add_argument(
        "--output", type=Path, default=EMBEDDING_AUDIT_PATH
    )

    index_parser = subparsers.add_parser(
        "index", help="Audit and upsert all chunks into pgvector."
    )
    index_parser.add_argument(
        "--audit-output", type=Path, default=EMBEDDING_AUDIT_PATH
    )

    query_parser = subparsers.add_parser(
        "query", help="Inspect retrieved chunks, scores, and citations."
    )
    query_parser.add_argument("query")
    query_parser.add_argument("-k", type=int, default=DEFAULT_SEARCH_K)
    query_parser.add_argument(
        "--max-content-chars", type=int, default=DEFAULT_CHUNK_SIZE
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    load_dotenv()

    if args.command in {"audit", "index"}:
        output_path = args.output if args.command == "audit" else args.audit_output
        documents, rows, profile, base_url = _run_audit(
            output_path=output_path
        )
        _print_audit_summary(rows, output_path)
        if args.command == "audit":
            return
        require_complete_embeddings(rows)
        embeddings = make_embeddings(profile=profile, base_url=base_url)
        store = create_vector_store(
            connection=database_url(),
            embeddings=embeddings,
            embedding_length=profile.embedding_length,
        )
        ids = index_documents(store, documents)
        print(
            f"Indexed {len(ids)} chunks into {COLLECTION_NAME!r} "
            "with deterministic upsert IDs."
        )
        return

    base_url, _, profile = _runtime()
    embeddings = make_embeddings(profile=profile, base_url=base_url)
    store = create_vector_store(
        connection=database_url(),
        embeddings=embeddings,
        embedding_length=profile.embedding_length,
    )
    results = retrieve_with_scores(store, args.query, k=args.k)
    print(
        format_retrieval_results(
            results,
            max_content_chars=args.max_content_chars,
        )
    )


if __name__ == "__main__":
    main()
