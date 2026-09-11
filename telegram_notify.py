#!/usr/bin/env python3
"""Small Telegram notification client shared by forecast monitors."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Mapping


TELEGRAM_API_ROOT = "https://api.telegram.org"
MAX_MESSAGE_CHARACTERS = 4000


def configured(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return bool(env.get("TELEGRAM_BOT_TOKEN", "").strip() and env.get("TELEGRAM_CHAT_ID", "").strip())


def send_message(
    title: str,
    body: str,
    *,
    url: str | None = None,
    environ: Mapping[str, str] | None = None,
    timeout: float = 15.0,
) -> dict[str, object]:
    env = os.environ if environ is None else environ
    token = env.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = env.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")

    sections = [title.strip(), body.strip()]
    if url:
        sections.append(url.strip())
    text = "\n\n".join(section for section in sections if section)[:MAX_MESSAGE_CHARACTERS]
    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
            "disable_notification": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{TELEGRAM_API_ROOT}/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except Exception as exc:
        raise RuntimeError(str(exc).replace(token, "[redacted]")) from None
    if not isinstance(result, dict) or not result.get("ok"):
        description = result.get("description", "unknown Telegram error") if isinstance(result, dict) else "invalid response"
        raise RuntimeError(f"Telegram notification failed: {description}")
    message = result.get("result") or {}
    return {"status": "accepted_by_telegram", "message_id": message.get("message_id"), "date": message.get("date")}
