"""Minimal aiohttp dashboard for runtime state visibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

from aiohttp import web


@dataclass
class DashboardState:
	"""Mutable dashboard state snapshot."""

	started_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
	last_message_at: str = ""
	last_signal: str = ""
	last_error: str = ""
	equity: float = 0.0
	open_orders: int = 0
	open_positions: int = 0
	extra: Dict[str, Any] = field(default_factory=dict)


class DashboardServer:
	"""Small web dashboard with JSON health and HTML summary endpoints."""

	def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
		"""Initialize dashboard server.

		Args:
			host: Bind host.
			port: Bind port.
		"""
		self._host = host
		self._port = port
		self.state = DashboardState()
		self._app = web.Application()
		self._app.add_routes([
			web.get("/", self._index),
			web.get("/health", self._health),
			web.get("/state", self._state),
		])
		self._runner: web.AppRunner | None = None
		self._site: web.TCPSite | None = None

	async def start(self) -> None:
		"""Start dashboard HTTP server."""
		self._runner = web.AppRunner(self._app)
		await self._runner.setup()
		self._site = web.TCPSite(self._runner, host=self._host, port=self._port)
		await self._site.start()

	async def stop(self) -> None:
		"""Stop dashboard HTTP server."""
		if self._runner is not None:
			await self._runner.cleanup()

	def update(self, **kwargs: Any) -> None:
		"""Update mutable state fields for dashboard rendering."""
		for key, value in kwargs.items():
			if hasattr(self.state, key):
				setattr(self.state, key, value)
			else:
				self.state.extra[key] = value

	async def _health(self, _: web.Request) -> web.Response:
		return web.json_response({"status": "ok", "timestamp": datetime.now(tz=timezone.utc).isoformat()})

	async def _state(self, _: web.Request) -> web.Response:
		return web.json_response(
			{
				"started_at": self.state.started_at.isoformat(),
				"last_message_at": self.state.last_message_at,
				"last_signal": self.state.last_signal,
				"last_error": self.state.last_error,
				"equity": self.state.equity,
				"open_orders": self.state.open_orders,
				"open_positions": self.state.open_positions,
				"extra": self.state.extra,
			}
		)

	async def _index(self, _: web.Request) -> web.Response:
		html = f"""
		<html>
		<head>
			<title>MEXC Paper Dashboard</title>
			<style>
				body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; }}
				table {{ border-collapse: collapse; width: 700px; }}
				th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
				th {{ background: #f4f4f4; width: 220px; }}
			</style>
		</head>
		<body>
			<h1>MEXC Paper Trading Dashboard</h1>
			<table>
				<tr><th>Started At (UTC)</th><td>{self.state.started_at.isoformat()}</td></tr>
				<tr><th>Last Feed Message</th><td>{self.state.last_message_at}</td></tr>
				<tr><th>Last Signal</th><td>{self.state.last_signal}</td></tr>
				<tr><th>Last Error</th><td>{self.state.last_error}</td></tr>
				<tr><th>Equity</th><td>{self.state.equity:.2f}</td></tr>
				<tr><th>Open Orders</th><td>{self.state.open_orders}</td></tr>
				<tr><th>Open Positions</th><td>{self.state.open_positions}</td></tr>
			</table>
			<p>JSON endpoints: /health, /state</p>
		</body>
		</html>
		"""
		return web.Response(text=html, content_type="text/html")
