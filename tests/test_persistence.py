"""Deterministic tests for Phase 4B persistence composition."""

import os
import unittest
from unittest.mock import patch

from regulatory_agent.persistence import open_conversation_service
from regulatory_agent.postgres_repository import PersistenceInitializationError


class PersistenceConfigurationTests(unittest.TestCase):
    def test_postgres_initialization_failure_never_falls_back_to_memory(self):
        failure = PersistenceInitializationError("database unavailable")
        with patch(
            "regulatory_agent.persistence.create_connection_pool",
            side_effect=failure,
        ):
            with self.assertRaisesRegex(
                PersistenceInitializationError, "database unavailable"
            ):
                with open_conversation_service(
                    storage_mode="postgres",
                    conninfo="postgresql://unused/unused",
                ):
                    self.fail("PostgreSQL failure must not yield a service")

    def test_unknown_storage_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "storage_mode"):
            with open_conversation_service(
                storage_mode="sqlite",  # type: ignore[arg-type]
            ):
                self.fail("invalid mode must not yield a service")

    def test_postgres_mode_does_not_use_storage_mode_fallback(self):
        with patch.dict(os.environ, {"REGULATORY_STORAGE_MODE": "postgres"}):
            with patch(
                "regulatory_agent.persistence.create_connection_pool",
                side_effect=PersistenceInitializationError("explicit failure"),
            ):
                with self.assertRaisesRegex(
                    PersistenceInitializationError, "explicit failure"
                ):
                    with open_conversation_service(
                        conninfo="postgresql://unused/unused"
                    ):
                        self.fail("PostgreSQL mode must fail closed")


if __name__ == "__main__":
    unittest.main()
