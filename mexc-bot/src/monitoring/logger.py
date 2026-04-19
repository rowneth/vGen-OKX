"""Structured logging utilities with secret-safe redaction."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

_LOCAL_TZ = ZoneInfo("Asia/Colombo")

SENSITIVE_PATTERNS = [
	re.compile(r"(MEXC_API_KEY\s*=\s*)([^\s]+)", re.IGNORECASE),
	re.compile(r"(MEXC_API_SECRET\s*=\s*)([^\s]+)", re.IGNORECASE),
	re.compile(r"(ApiKey\s*[:=]\s*)([^\s,]+)", re.IGNORECASE),
	re.compile(r"(Signature\s*[:=]\s*)([^\s,]+)", re.IGNORECASE),
]


class SecretRedactingFormatter(logging.Formatter):
	"""Formatter that redacts sensitive credential-like values."""

	def format(self, record: logging.LogRecord) -> str:
		base = super().format(record)
		return redact_sensitive_text(base)


class JsonFormatter(logging.Formatter):
	"""Formatter that emits one JSON object per log line."""

	def format(self, record: logging.LogRecord) -> str:
		payload: Dict[str, Any] = {
			"timestamp": datetime.now(tz=_LOCAL_TZ).isoformat(),
			"level": record.levelname,
			"logger": record.name,
			"message": redact_sensitive_text(record.getMessage()),
		}
		if record.exc_info:
			payload["exception"] = self.formatException(record.exc_info)
		return json.dumps(payload, ensure_ascii=True)


def configure_logging(
	log_level: str,
	log_file_path: Optional[Path] = None,
	json_console: bool = False,
) -> None:
	"""Configure root logger for console and optional file output.

	Args:
		log_level: Logging level string.
		log_file_path: Optional file destination.
		json_console: Whether console output should be JSON.
	"""
	root = logging.getLogger()
	root.handlers.clear()
	root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

	console_handler = logging.StreamHandler()
	if json_console:
		console_handler.setFormatter(JsonFormatter())
	else:
		console_handler.setFormatter(
			SecretRedactingFormatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
		)
	root.addHandler(console_handler)

	if log_file_path is not None:
		log_file_path.parent.mkdir(parents=True, exist_ok=True)
		file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
		file_handler.setFormatter(JsonFormatter())
		root.addHandler(file_handler)


def redact_sensitive_text(text: str) -> str:
	"""Redact sensitive values from log text.

	Args:
		text: Input text.

	Returns:
		Redacted text.
	"""
	redacted = text
	for pattern in SENSITIVE_PATTERNS:
		redacted = pattern.sub(r"\1***REDACTED***", redacted)
	return redacted
