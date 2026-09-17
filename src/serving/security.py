"""HTTP boundary controls; configure production hosts/origins explicitly.

CSP is deliberately omitted: Swagger/ReDoc load external scripts and styles.
HSTS is opt-in for deployments terminated by a trusted HTTPS reverse proxy.
No forwarded headers are interpreted here. Uvicorn/proxy Server headers belong
in deployment configuration, not application-level cosmetic filtering.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


@dataclass(frozen=True)
class SecurityConfig:
    hosts: tuple[str, ...]
    origins: tuple[str, ...]
    hsts: bool = False

    @classmethod
    def from_env(cls) -> SecurityConfig:
        hosts = tuple(h.strip() for h in os.getenv(
            "ALLOWED_HOSTS", "localhost,127.0.0.1,testserver"
        ).split(",") if h.strip())
        if not hosts or any(
            "*" in h or any(c in h for c in "/:@\\ \t\r\n") for h in hosts
        ):
            raise ValueError("ALLOWED_HOSTS must contain explicit hostnames without ports")
        origins = tuple(o.strip() for o in os.getenv(
            "CORS_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
        ).split(",") if o.strip())
        for origin in origins:
            try:
                url = urlsplit(origin)
                valid = (
                    url.scheme in ("http", "https") and bool(url.hostname)
                    and not url.username and not url.password
                    and not url.path and not url.query and not url.fragment
                    and "*" not in origin and not any(c.isspace() for c in origin)
                )
                _ = url.port  # reject invalid/out-of-range ports
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("CORS_ALLOWED_ORIGINS must contain explicit HTTP(S) origins")
        flag = os.getenv("SECURITY_HSTS_ENABLED", "false").strip().lower()
        if flag not in ("true", "false"):
            raise ValueError("SECURITY_HSTS_ENABLED must be true or false")
        return cls(hosts, origins, flag == "true")

    def headers(self) -> dict[str, str]:
        headers = {
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        }
        if self.hsts:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return headers


class SecurityHeadersMiddleware:
    """Pure ASGI response-header wrapper; never reads bodies or identities."""

    def __init__(self, app: ASGIApp, headers: dict[str, str]) -> None:
        self.app = app
        self.headers = headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in self.headers.items():
                    headers[name] = value
            await send(message)

        await self.app(scope, receive, send_headers)
