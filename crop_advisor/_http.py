"""Minimal HTTP helper with TLS verification via certifi.

python.org macOS builds don't wire up system CA certs, so we verify against
certifi's bundle when present (falling back to the default context otherwise).
Verification is never disabled.
"""
import json
import ssl
import urllib.request

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    _SSL_CTX = ssl.create_default_context()

_UA = "crop-climate-advisor/0.1 (portfolio project; non-commercial research)"


def get_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
        return resp.read().decode("utf-8", "replace")


def get_json(url: str, timeout: int = 30) -> dict:
    return json.loads(get_text(url, timeout))
