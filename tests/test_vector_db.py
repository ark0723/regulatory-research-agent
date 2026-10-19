"""Focused tests for non-truncating indexing and retrieval."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document
from ollama import ResponseError

from regulatory_agent.vector_db import (
    COLLECTION_NAME,
    DISTANCE_STRATEGY,
    EmbeddingAuditRow,
    EmbeddingInputLimitError,
    ModelProfile,
    audit_embedding_inputs,
    build_index_documents,
    create_vector_store,
    database_url,
    embedding_options,
    format_retrieval_results,
    index_documents,
    make_embeddings,
    require_complete_embeddings,
    retrieve_with_scores,
    vector_document_id,
    write_embedding_audit,
)


PROFILE = ModelProfile(
    model="bge-m3",
    architecture="bert",
    context_length=8192,
    embedding_length=1024,
)


def document(record_id: str, text: str = "content") -> Document:
    return Document(
        page_content=text,
        metadata={
            "record_id": record_id,
            "parent_record_id": "parent-1",
            "chunk_id": record_id,
            "document_type": "regulation",
            "content_type": "section",
            "section_id": "50.23",
            "citation": "21 CFR 50.23",
            "page_start": 10,
            "page_end": 11,
            "source_url": "https://example.test/50.23",
        },
    )


class FakeAuditClient:
    def __init__(self):
        self.calls = []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["input"] == "too long":
            raise ResponseError("the input length exceeds the context length", 400)
        return SimpleNamespace(prompt_eval_count=12)


class FakeEmbeddingClient:
    def __init__(self):
        self.calls = []

    def embed(self, model, texts, **kwargs):
        self.calls.append((model, texts, kwargs))
        return {"embeddings": [[0.1, 0.2] for _ in texts]}


class FakeStore:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.add_calls = []
        self.search_calls = []
        self.search_results = []
        self.__class__.instances.append(self)

    def add_documents(self, *, documents, ids):
        self.add_calls.append((documents, ids))
        return ids

    def similarity_search_with_score(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_results


class VectorDbTests(unittest.TestCase):
    def setUp(self):
        FakeStore.instances.clear()

    def test_fixed_chunk_pipeline_builds_330_unique_documents(self):
        documents = build_index_documents()
        self.assertEqual(330, len(documents))
        ids = [document.metadata["chunk_id"] for document in documents]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(
            max(len(document.page_content) for document in documents), 1500
        )

    def test_database_url_requires_expected_environment(self):
        url = database_url(
            {
                "POSTGRES_USER": "user",
                "POSTGRES_PASSWORD": "secret",
                "POSTGRES_DB": "regulatory",
            }
        )
        self.assertEqual("localhost", url.host)
        self.assertEqual(5432, url.port)
        with self.assertRaisesRegex(ValueError, "POSTGRES_PASSWORD"):
            database_url({"POSTGRES_USER": "user", "POSTGRES_DB": "db"})

    def test_audit_and_production_embeddings_never_truncate(self):
        audit_client = FakeAuditClient()
        rows = audit_embedding_inputs(
            [document("short", "short")],
            client=audit_client,
            profile=PROFILE,
        )
        self.assertTrue(rows[0].accepted_without_truncation)
        self.assertTrue(
            all(call["truncate"] is False for call in audit_client.calls)
        )
        self.assertTrue(
            all(
                call["options"] == embedding_options(8192)
                for call in audit_client.calls
            )
        )

        embeddings = make_embeddings(profile=PROFILE)
        embedding_client = FakeEmbeddingClient()
        embeddings._client = embedding_client
        embeddings.embed_documents(["document"])
        embeddings.embed_query("query")
        for model, texts, kwargs in embedding_client.calls:
            self.assertEqual("bge-m3", model)
            self.assertEqual(1, len(texts))
            self.assertFalse(kwargs["truncate"])
            self.assertEqual(embedding_options(8192), kwargs["options"])

    def test_failed_audit_blocks_indexing(self):
        client = FakeAuditClient()
        rows = audit_embedding_inputs(
            [document("long", "too long")], client=client, profile=PROFILE
        )
        with self.assertRaisesRegex(
            EmbeddingInputLimitError, "nothing was indexed"
        ):
            require_complete_embeddings(rows)
        self.assertTrue(all(call["truncate"] is False for call in client.calls))

    def test_audit_report_describes_single_collection(self):
        row = EmbeddingAuditRow(
            record_id="record-1",
            document_type="regulation",
            content_type="section",
            section_id="50.23",
            character_count=100,
            content_sha256="a" * 64,
            accepted_without_truncation=True,
            token_count=25,
            max_accepted_prefix_chars=None,
            max_accepted_prefix_tokens=None,
            first_rejected_prefix_chars=None,
            error_status_code=None,
            error="",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "audit.json"
            write_embedding_audit([row], profile=PROFILE, output_path=path)
            content = path.read_text(encoding="utf-8")
        self.assertIn('"collection_name": "regulatory_chunks"', content)
        self.assertIn('"document_count": 1', content)
        self.assertIn('"truncate": false', content)
        self.assertNotIn("baseline", content.lower())
        self.assertNotIn("page_content", content)

    def test_single_store_uses_expected_collection_and_cosine_distance(self):
        embeddings = object()
        store = create_vector_store(
            connection="postgresql+psycopg://example",
            embeddings=embeddings,
            embedding_length=1024,
            vector_store_class=FakeStore,
        )
        self.assertIs(store, FakeStore.instances[0])
        self.assertEqual(COLLECTION_NAME, store.kwargs["collection_name"])
        self.assertIs(embeddings, store.kwargs["embeddings"])
        self.assertEqual(DISTANCE_STRATEGY, store.kwargs["distance_strategy"])
        self.assertEqual(1024, store.kwargs["embedding_length"])

    def test_indexing_uses_deterministic_namespaced_upsert_ids(self):
        store = FakeStore()
        documents = [document("chunk-1"), document("chunk-2")]
        first_ids = index_documents(store, documents)
        second_ids = index_documents(store, documents)
        expected = [
            "regulatory_chunks:chunk-1",
            "regulatory_chunks:chunk-2",
        ]
        self.assertEqual(expected, first_ids)
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(expected, store.add_calls[0][1])
        self.assertEqual(expected, store.add_calls[1][1])
        self.assertEqual(expected[0], vector_document_id(documents[0]))

    def test_retrieval_passes_k_filter_and_formats_citations(self):
        store = FakeStore()
        result_document = document("chunk-1", "retrieved regulatory text")
        store.search_results = [(result_document, 0.1234567)]
        results = retrieve_with_scores(
            store,
            "informed consent",
            k=5,
            metadata_filter={"document_type": "regulation"},
        )
        self.assertEqual(
            {
                "query": "informed consent",
                "k": 5,
                "filter": {"document_type": "regulation"},
            },
            store.search_calls[0],
        )
        rendered = format_retrieval_results(results)
        self.assertIn("score=0.123457", rendered)
        self.assertIn("citation: 21 CFR 50.23", rendered)
        self.assertIn("record_id: chunk-1", rendered)
        self.assertIn("retrieved regulatory text", rendered)

    def test_retrieval_rejects_invalid_query_and_k(self):
        store = FakeStore()
        with self.assertRaisesRegex(ValueError, "query"):
            retrieve_with_scores(store, "   ")
        with self.assertRaisesRegex(ValueError, "k"):
            retrieve_with_scores(store, "query", k=0)


if __name__ == "__main__":
    unittest.main()
