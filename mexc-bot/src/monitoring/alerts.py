"""Alerting utilities for fills, warnings, and runtime errors."""

from __future__ import annotations

import asyncio
import logging
import smtplib
from dataclasses import dataclass
from email.mime.text import MIMEText
from typing import Optional

import aiohttp

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlertMessage:
	"""Represents one alert payload."""

	level: str
	title: str
	body: str


class AlertSink:
	"""Base sink interface for alert delivery."""

	async def send(self, message: AlertMessage) -> None:
		"""Send one alert message."""
		raise NotImplementedError


class ConsoleAlertSink(AlertSink):
	"""Writes alerts to local process logs."""

	async def send(self, message: AlertMessage) -> None:
		LOGGER.warning("ALERT level=%s title=%s body=%s", message.level, message.title, message.body)


class TelegramAlertSink(AlertSink):
	"""Send alerts via Telegram bot API."""

	def __init__(self, bot_token: str, chat_id: str) -> None:
		"""Initialize sink.

		Args:
			bot_token: Telegram bot token.
			chat_id: Target chat ID.
		"""
		self._bot_token = bot_token
		self._chat_id = chat_id

	async def send(self, message: AlertMessage) -> None:
		url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
		text = f"[{message.level}] {message.title}\n{message.body}"
		payload = {
			"chat_id": self._chat_id,
			"text": text,
		}
		async with aiohttp.ClientSession() as session:
			async with session.post(url, json=payload, timeout=10) as response:
				response.raise_for_status()


class EmailAlertSink(AlertSink):
	"""Send alerts using SMTP."""

	def __init__(
		self,
		smtp_host: str,
		smtp_port: int,
		username: str,
		password: str,
		from_email: str,
		to_email: str,
		use_tls: bool = True,
	) -> None:
		"""Initialize SMTP alert sink.

		Args:
			smtp_host: SMTP server host.
			smtp_port: SMTP server port.
			username: SMTP username.
			password: SMTP password.
			from_email: Sender email.
			to_email: Recipient email.
			use_tls: Whether to start TLS.
		"""
		self._smtp_host = smtp_host
		self._smtp_port = smtp_port
		self._username = username
		self._password = password
		self._from_email = from_email
		self._to_email = to_email
		self._use_tls = use_tls

	async def send(self, message: AlertMessage) -> None:
		await asyncio.to_thread(self._send_sync, message)

	def _send_sync(self, message: AlertMessage) -> None:
		mime = MIMEText(message.body)
		mime["Subject"] = f"[{message.level}] {message.title}"
		mime["From"] = self._from_email
		mime["To"] = self._to_email

		with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=10) as smtp:
			if self._use_tls:
				smtp.starttls()
			smtp.login(self._username, self._password)
			smtp.sendmail(self._from_email, [self._to_email], mime.as_string())


class AlertPublisher:
	"""Fan-out publisher for multiple alert sinks."""

	def __init__(self, sinks: Optional[list[AlertSink]] = None) -> None:
		"""Initialize publisher.

		Args:
			sinks: Optional sink list.
		"""
		self._sinks = sinks or [ConsoleAlertSink()]

	async def publish(self, level: str, title: str, body: str) -> None:
		"""Publish one alert to all sinks.

		Args:
			level: Alert severity level.
			title: Alert title.
			body: Alert details.
		"""
		message = AlertMessage(level=level, title=title, body=body)
		for sink in self._sinks:
			try:
				await sink.send(message)
			except Exception as exc:  # noqa: BLE001
				LOGGER.error("Alert sink failed: %s", exc)
