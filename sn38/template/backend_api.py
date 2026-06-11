"""Backend API client for the validator."""

import bittensor as bt
from .tee import ValidatorSession


class BackendAPI:
    def __init__(self, backend_url: str, hotkey: str = "unknown"):
        self.session = ValidatorSession(backend_url, hotkey=hotkey)
        bt.logging.info(f"Backend URL: {backend_url}")
        bt.logging.info(f"TEE status: {self.session.is_tee}")

    def get_config(self):
        return self.session.get("/config").json()

    def get_eval_round(self):
        return self.session.get("/rounds/current").json()["eval_round"]

    def get_benchmark(self, cutoff_year):
        resp = self.session.get(f"/benchmark/{cutoff_year}")
        if resp.status_code != 200:
            bt.logging.error(f"Backend /benchmark/{cutoff_year} returned {resp.status_code}")
            return None
        return resp.json()

    def get_submissions(self, round_num):
        resp = self.session.get(f"/submissions/{round_num}")
        if resp.status_code != 200:
            bt.logging.error(f"Backend /submissions/{round_num} returned {resp.status_code}")
            return {}, {}
        data = resp.json()
        models = {int(uid): sub["models"] for uid, sub in data.get("submissions", {}).items()}
        timestamps = {int(uid): sub["snapshot_at"] for uid, sub in data.get("submissions", {}).items()}
        return models, timestamps

    def get_quality_questions(self):
        resp = self.session.get("/quality/questions")
        if resp.status_code != 200:
            bt.logging.error(f"Backend /quality/questions returned {resp.status_code}")
            return []
        return resp.json().get("questions", [])
