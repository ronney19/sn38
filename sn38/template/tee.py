"""
TEE attestation and request signing for validator authentication.

Uses dstack SDK to:
- Obtain full attestation from inside a TEE
- Sign each request (method + path + body hash + timestamp + nonce)

The backend verifies the signature to ensure requests come from a real TEE
and are not replayed.

Retries indefinitely on backend downtime.
"""

import hashlib
import json
import time
import uuid

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ValidatorSession:

    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.attestation = None

        try:
            from dstack_sdk import DstackClient
            self._client = DstackClient()
            result = self._client.attest(b"sn38-chronollm-validator")
            self.attestation = result.attestation
        except Exception:
            self._client = None

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

    def _sign_request(self, method: str, path: str, body: bytes = b""):
        """Sign request data and return auth headers."""
        if not self._client:
            return {}

        timestamp = str(int(time.time()))
        nonce = str(uuid.uuid4())
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{method}:{path}:{body_hash}:{timestamp}:{nonce}"

        result = self._client.sign("secp256k1", message.encode())

        return {
            "X-TEE-Attestation": self.attestation,
            "X-TEE-Signature": result.signature,
            "X-TEE-Public-Key": result.public_key,
            "X-TEE-Signature-Chain": json.dumps(result.signature_chain),
            "X-TEE-Timestamp": timestamp,
            "X-TEE-Nonce": nonce,
        }

    def get(self, path: str) -> requests.Response:
        headers = self._sign_request("GET", path)
        return self.session.get(f"{self.backend_url}{path}", headers=headers)

    def post(self, path: str, json_data=None) -> requests.Response:
        body = json.dumps(json_data, separators=(",", ":"), sort_keys=True).encode() if json_data else b""
        headers = self._sign_request("POST", path, body)
        return self.session.post(f"{self.backend_url}{path}", content=body, headers={**headers, "Content-Type": "application/json"})
