"""
TEE attestation for validator authentication.

Uses dstack SDK to obtain full attestation from inside a TEE.
The attestation is attached to backend API requests via a header.
The backend forwards it to dstack-verifier for full verification.

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

        try:
            from dstack_sdk import DstackClient
            client = DstackClient()
            result = client.attest(b"sn38-chronollm-validator")
            self.attestation = result.attestation
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
        }

    def get(self, path: str) -> requests.Response:
        return self.session.get(f"{self.backend_url}{path}", headers=self._headers())

    def post(self, path: str, json=None) -> requests.Response:
        return self.session.post(f"{self.backend_url}{path}", json=json, headers=self._headers())
