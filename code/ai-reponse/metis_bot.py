import requests
import time


class MetisBotClient:
    def __init__(self, api_key: str, base_url: str = "https://api.metisai.ir/api/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def create_session(self, bot_id: str) -> str:
        url = f"{self.base_url}/chat/session"
        resp = requests.post(url, headers=self.headers, json={"botId": bot_id})
        resp.raise_for_status()
        return resp.json()["id"]

    def send_message(
        self,
        bot_id: str,
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> dict:
        session_id = self.create_session(bot_id)

        url = f"{self.base_url}/chat/session/{session_id}/message"
        payload = {
            "message": {
                "content": prompt,
                "type": "USER",
            }
        }

        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(url, headers=self.headers, json=payload, timeout=60)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException:
                if attempt == max_retries:
                    raise
                time.sleep(retry_delay)
                retry_delay *= 2
