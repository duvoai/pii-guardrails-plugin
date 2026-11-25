#!/usr/bin/env python3
"""
PII detection and redaction for Claude Code MCP tool calls.

Uses Claude Haiku to detect PII in tool inputs/outputs and redacts them
with typed tags like [EMAIL], [SSN], [PHONE_NUMBER], etc.

Called by Claude Code hooks (PreToolUse / PostToolUse).
Zero external dependencies — uses only Python stdlib.
"""

import json
import os
import sys
import urllib.request
import urllib.error
import argparse
import re

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"

DETECTION_PROMPT = """\
Analyze the following text for Personally Identifiable Information (PII).

Detect these PII types:
- EMAIL: Email addresses
- PHONE_NUMBER: Phone numbers (any format)
- SSN: Social Security Numbers
- CREDIT_CARD: Credit/debit card numbers
- NAME: Full person names (not company/product names)
- ADDRESS: Physical/mailing addresses
- DATE_OF_BIRTH: Dates of birth
- IP_ADDRESS: IP addresses (v4 and v6)
- PASSPORT: Passport numbers
- DRIVERS_LICENSE: Driver's license numbers
- BANK_ACCOUNT: Bank account/routing numbers

For each PII found, return its exact text as it appears and its type.

Return ONLY a JSON array. No other text. Examples:
- Found PII: [{"text": "john@example.com", "type": "EMAIL"}, {"text": "555-123-4567", "type": "PHONE_NUMBER"}]
- No PII: []

Text to analyze:
<content>
{content}
</content>"""

MAX_CONTENT_LENGTH = 50000  # Max characters per API request chunk
CHUNK_OVERLAP = 256


def get_api_key():
    """Get Anthropic API key from environment."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    return key


def _extract_pii_items(text):
    """Extract and validate a JSON array of {text, type} dicts from model output."""
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("[", idx)
        if start == -1:
            return []
        try:
            parsed, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(parsed, list):
            valid = [
                {"text": item["text"], "type": item["type"]}
                for item in parsed
                if isinstance(item, dict)
                and isinstance(item.get("text"), str)
                and item["text"]
                and isinstance(item.get("type"), str)
                and item["type"]
            ]
            if valid or not parsed:
                return valid
        idx = start + 1


def _chunk_content(content, max_len=MAX_CONTENT_LENGTH, overlap=CHUNK_OVERLAP):
    """Split large content into bounded chunks, preferring line/whitespace breaks."""
    if len(content) <= max_len:
        return [content]
    chunks = []
    start = 0
    n = len(content)
    while start < n:
        end = min(start + max_len, n)
        if end < n:
            split_at = content.rfind("\n", start + max_len // 2, end)
            if split_at == -1:
                split_at = content.rfind(" ", start + max_len // 2, end)
            if split_at > start:
                end = split_at + 1
        chunks.append(content[start:end])
        if end >= n:
            break
        next_start = end - min(overlap, max_len // 4)
        start = max(start + 1, next_start)
    return chunks


def _call_haiku_chunk(chunk, api_key):
    """Call Claude Haiku once on a bounded content chunk."""
    prompt = DETECTION_PROMPT.replace("{content}", chunk)

    payload = json.dumps({
        "model": MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    resp = urllib.request.urlopen(req, timeout=25)
    body = json.loads(resp.read().decode("utf-8"))

    content_blocks = body.get("content") or []
    text_parts = [
        block.get("text", "")
        for block in content_blocks
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    text = "\n".join(text_parts).strip()
    if not text:
        return []

    return _extract_pii_items(text)


def call_haiku(content, api_key):
    """Call Claude Haiku to detect PII in content. Returns list of {text, type}."""
    if not content or not content.strip():
        return []

    seen = set()
    items = []
    for chunk in _chunk_content(content):
        if not chunk.strip():
            continue
        for item in _call_haiku_chunk(chunk, api_key):
            key = (item["text"], item["type"])
            if key not in seen:
                seen.add(key)
                items.append(item)
    return items


def redact(text, pii_items):
    """Replace each PII occurrence with its typed tag in a single non-cascading pass."""
    replacements = {}
    for item in pii_items:
        if not isinstance(item, dict):
            continue
        pii_text = item.get("text")
        pii_type = item.get("type")
        if isinstance(pii_text, str) and pii_text and isinstance(pii_type, str) and pii_type:
            replacements.setdefault(pii_text, f"[{pii_type}]")
    if not replacements:
        return text

    # Longest match first, single-pass regex so inserted [TAG] markers are never re-matched
    ordered = sorted(replacements, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(tok) for tok in ordered))
    return pattern.sub(lambda m: replacements[m.group(0)], text)


def flatten_text(obj):
    """Recursively extract all string values from a JSON object."""
    parts = []
    if isinstance(obj, str):
        parts.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            parts.extend(flatten_text(v))
    elif isinstance(obj, list):
        for v in obj:
            parts.extend(flatten_text(v))
    return parts


def redact_obj(obj, pii_items):
    """Recursively redact PII from all string values in a JSON object."""
    if not pii_items:
        return obj
    if isinstance(obj, str):
        return redact(obj, pii_items)
    elif isinstance(obj, dict):
        return {k: redact_obj(v, pii_items) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [redact_obj(v, pii_items) for v in obj]
    return obj


def handle_pre_tool_use(hook_input, api_key):
    """Detect and redact PII from tool input before execution."""
    tool_input = hook_input.get("tool_input", {})

    # Flatten all string content for PII scanning
    all_text = "\n".join(flatten_text(tool_input))
    if not all_text.strip():
        # No text content to scan
        return json.dumps({})

    pii_items = call_haiku(all_text, api_key)
    if not pii_items:
        # No PII found — allow as-is
        return json.dumps({})

    # Redact PII from tool input
    redacted_input = redact_obj(tool_input, pii_items)
    types_found = sorted(set(item["type"] for item in pii_items))

    return json.dumps({
        "hookSpecificOutput": {
            "permissionDecision": "allow",
            "updatedInput": redacted_input,
        },
        "systemMessage": f"PII redacted from tool input. Types found: {', '.join(types_found)}",
    })


def handle_post_tool_use(hook_input, api_key):
    """Detect and redact PII from tool output after execution."""
    tool_result = hook_input.get("tool_result", "")

    # tool_result can be a string or structured object
    all_text = "\n".join(flatten_text(tool_result))
    if not all_text.strip():
        return json.dumps({})

    pii_items = call_haiku(all_text, api_key)
    if not pii_items:
        return json.dumps({})

    redacted_result = redact_obj(tool_result, pii_items)
    types_found = sorted(set(item["type"] for item in pii_items))

    # For PostToolUse, provide redacted content via systemMessage
    if isinstance(redacted_result, str):
        redacted_display = redacted_result
    else:
        redacted_display = json.dumps(redacted_result, indent=2)

    return json.dumps({
        "systemMessage": (
            f"PII detected in tool output ({', '.join(types_found)}). "
            f"Use this redacted version instead:\n\n{redacted_display}"
        ),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hook", required=True, choices=["pre", "post"])
    args = parser.parse_args()

    # Read hook input from stdin
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError, ValueError, UnicodeError):
        # Can't parse input — fail open
        sys.exit(0)

    if not isinstance(hook_input, dict):
        sys.exit(0)

    # Check for API key
    api_key = get_api_key()
    if not api_key:
        # No API key — fail open with a warning
        print(json.dumps({
            "systemMessage": "guardrails-pii: ANTHROPIC_API_KEY not set. PII redaction skipped.",
        }))
        sys.exit(0)

    try:
        if args.hook == "pre":
            result = handle_pre_tool_use(hook_input, api_key)
        else:
            result = handle_post_tool_use(hook_input, api_key)

        print(result)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        OSError,
        UnicodeError,
    ) as e:
        # API call or response parsing failed — fail open
        print(
            json.dumps(
                {
                    "systemMessage": (
                        f"guardrails-pii: PII detection unavailable ({type(e).__name__})."
                        " Tool call proceeded without redaction."
                    ),
                }
            )
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
