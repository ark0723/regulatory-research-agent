"""Tests for shared runtime factories."""

import os
import unittest
from unittest.mock import patch

from regulatory_agent.runtime import create_chat_model


class RuntimeTests(unittest.TestCase):
    def test_chat_model_requires_explicit_model_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "No chat model configured"):
                create_chat_model(provider="ollama", model=None)


if __name__ == "__main__":
    unittest.main()
