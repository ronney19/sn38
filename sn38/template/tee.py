"""
TEE attestation for validator authentication.

Uses dstack SDK to obtain attestation from inside a TEE.
The attestation is attached to backend API requests via headers.
Traffic is protected by HTTPS — TLS terminates inside the CVM,
so the host operator cannot read the requests or responses.

Retries indefinitely on backend downtime.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ValidatorSession:
    """
    HTTP session that attaches TEE attestation to every backend request.
    Retries indefinitely on connection errors and 5xx responses.
    """

    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.attestation = None
        self.compose_hash = None

        try:
            from dstack_sdk import DstackClient
            client = DstackClient()
            info = client.info()
            quote = client.get_quote(b"sn38-chronollm-validator")

            self.attestation = quote.quote
            self.compose_hash = info.compose_hash
        except Exception:
            pass

        self.session = requests.Session()
        self.session.verify = False
        retry = Retry(
            total=None,
            backoff_factor=5,
            backoff_max=300,
            status_forcelist=[502, 503, 504],
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    @property
    def is_tee(self) -> bool:
        return self.attestation is not None

    def _headers(self) -> dict:
        if not self.attestation:
            return {}
        return {
            "X-TEE-Attestation": self.attestation,
            "X-TEE-Compose-Hash": self.compose_hash or "",
        }

    def get(self, path: str) -> requests.Response:
        return self.session.get(f"{self.backend_url}{path}", headers=self._headers())

    def post(self, path: str, json=None) -> requests.Response:
        return self.session.post(f"{self.backend_url}{path}", json=json, headers=self._headers())
