import importlib.util
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "hooks" / "scripts" / "pii-redact.py"
SPEC = importlib.util.spec_from_file_location("pii_redact", SCRIPT_PATH)
pii_redact = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pii_redact)


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data


class PiiRedactTests(unittest.TestCase):
    def test_extracts_json_array_even_with_leading_brackets_in_prose(self):
        response_text = (
            "Detected [EMAIL] and [PHONE_NUMBER] entities:\n"
            "```json\n"
            '[{"text": "john@acme.com", "type": "EMAIL"}, {"text": "555-123-4567", "type": "PHONE_NUMBER"}]\n'
            "```"
        )
        with mock.patch.object(
            pii_redact.urllib.request,
            "urlopen",
            return_value=FakeHTTPResponse({"content": [{"text": response_text}]}),
        ):
            items = pii_redact.call_haiku("contact john@acme.com or 555-123-4567", "test-key")

        self.assertEqual(
            items,
            [
                {"text": "john@acme.com", "type": "EMAIL"},
                {"text": "555-123-4567", "type": "PHONE_NUMBER"},
            ],
        )

    def test_redact_single_pass_avoids_corrupting_inserted_tags(self):
        # 'DR' is a substring of '[ADDRESS]'; single-pass replacement must not rewrite [ADDRESS] -> [A[NAME]RESS]
        text = "123 Main St (attn DR)"
        pii_items = [
            {"text": "123 Main St", "type": "ADDRESS"},
            {"text": "DR", "type": "NAME"},
        ]
        self.assertEqual(pii_redact.redact(text, pii_items), "[ADDRESS] (attn [NAME])")

    def test_redact_ignores_empty_or_malformed_items(self):
        pii_items = [
            {"text": "", "type": "NAME"},
            {"text": None, "type": "EMAIL"},
            "invalid",
            {"text": "jane@acme.com", "type": "EMAIL"},
        ]
        self.assertEqual(
            pii_redact.redact("hello jane@acme.com", pii_items),
            "hello [EMAIL]",
        )

    def test_call_haiku_scans_all_chunks_past_max_content_length(self):
        prompts_seen = []

        def fake_urlopen(req, timeout=25):
            body = json.loads(req.data.decode("utf-8"))
            prompt = body["messages"][0]["content"]
            prompts_seen.append(prompt)
            if "tail-secret@acme.com" in prompt:
                return FakeHTTPResponse(
                    {"content": [{"text": '[{"text": "tail-secret@acme.com", "type": "EMAIL"}]'}]}
                )
            return FakeHTTPResponse({"content": [{"text": "[]"}]})

        large_text = ("x" * (pii_redact.MAX_CONTENT_LENGTH + 50)) + " tail-secret@acme.com"
        with mock.patch.object(pii_redact.urllib.request, "urlopen", side_effect=fake_urlopen):
            items = pii_redact.call_haiku(large_text, "test-key")

        self.assertGreaterEqual(len(prompts_seen), 2)
        self.assertEqual(items, [{"text": "tail-secret@acme.com", "type": "EMAIL"}])

    def test_main_fails_open_on_timeout_and_empty_content(self):
        cases = (
            (lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("timed out")), True),
            (lambda *a, **kw: FakeHTTPResponse({"content": []}), False),
        )
        for fake_open, expect_exit in cases:
            stdout = io.StringIO()
            stdin = io.StringIO(json.dumps({"tool_input": {"q": "hi"}}))
            with (
                mock.patch.object(
                    pii_redact.urllib.request, "urlopen", side_effect=fake_open
                ),
                mock.patch.dict(
                    os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False
                ),
                mock.patch.object(sys, "stdin", stdin),
                mock.patch.object(sys, "stdout", stdout),
                mock.patch.object(sys, "argv", ["pii-redact.py", "--hook", "pre"]),
            ):
                if expect_exit:
                    with self.assertRaises(SystemExit) as cm:
                        pii_redact.main()
                    self.assertEqual(cm.exception.code, 0)
                else:
                    pii_redact.main()

            out = stdout.getvalue().strip()
            self.assertTrue(out)
            parsed = json.loads(out)
            if expect_exit:
                self.assertIn("systemMessage", parsed)
            else:
                self.assertEqual(parsed, {})

    def test_pre_and_post_tool_use_nested_redaction(self):
        with mock.patch.object(
            pii_redact.urllib.request,
            "urlopen",
            return_value=FakeHTTPResponse(
                {"content": [{"text": '[{"text": "john@acme.com", "type": "EMAIL"}]'}]}
            ),
        ):
            pre_out = json.loads(
                pii_redact.handle_pre_tool_use(
                    {"tool_input": {"recipients": ["john@acme.com"], "meta": {"ok": 1}}},
                    "test-key",
                )
            )
            post_out = json.loads(
                pii_redact.handle_post_tool_use(
                    {"tool_result": {"body": "Contact john@acme.com"}},
                    "test-key",
                )
            )

        self.assertEqual(
            pre_out["hookSpecificOutput"]["updatedInput"],
            {"recipients": ["[EMAIL]"], "meta": {"ok": 1}},
        )
        self.assertIn("[EMAIL]", post_out["systemMessage"])
        self.assertNotIn("john@acme.com", post_out["systemMessage"])


if __name__ == "__main__":
    unittest.main()
