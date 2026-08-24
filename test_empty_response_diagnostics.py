import unittest
from unittest.mock import patch, MagicMock

import run_sycon_live as rsl


def _fake_response(content, finish_reason, reasoning_content=None):
    message = MagicMock()
    message.content = content
    message.reasoning_content = reasoning_content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    return response


class TestEmptyResponseDiagnostics(unittest.TestCase):
    def test_truncated_reasoning_populates_diag_and_logs_warning(self):
        diag = []
        fake = _fake_response(content=None, finish_reason="length", reasoning_content="thinking " * 200)
        with patch("run_sycon_live.completion", return_value=fake):
            with self.assertLogs("sycon", level="WARNING") as cm:
                result = rsl.call(
                    "fake/model", [{"role": "user", "content": "hi"}], None,
                    temperature=1.0, top_p=1.0, max_tokens=2048, tag="generation", diag=diag,
                )
        self.assertEqual(result, "")
        self.assertEqual(len(diag), 1)
        self.assertEqual(diag[0]["finish_reason"], "length")
        self.assertIsNone(diag[0]["error"])
        self.assertLessEqual(len(diag[0]["reasoning_preview"]), 500)
        self.assertTrue(any("length" in msg for msg in cm.output))

    def test_real_content_appends_none_to_diag(self):
        diag = []
        fake = _fake_response(content="a real answer", finish_reason="stop")
        with patch("run_sycon_live.completion", return_value=fake):
            result = rsl.call(
                "fake/model", [{"role": "user", "content": "hi"}], None,
                temperature=1.0, top_p=1.0, max_tokens=2048, tag="generation", diag=diag,
            )
        self.assertEqual(result, "a real answer")
        self.assertEqual(diag, [None])

    def test_diag_none_by_default_does_not_error(self):
        fake = _fake_response(content="fine", finish_reason="stop")
        with patch("run_sycon_live.completion", return_value=fake):
            result = rsl.call(
                "fake/model", [{"role": "user", "content": "hi"}], None,
                temperature=1.0, top_p=1.0, max_tokens=2048, tag="generation",
            )
        self.assertEqual(result, "fine")


if __name__ == "__main__":
    unittest.main()
