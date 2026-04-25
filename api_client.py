import io
import time
import requests


class CloudAPIError(RuntimeError):
    pass


class ComfyCloudClient:
    def __init__(self, api_key, base_url="https://cloud.comfy.org"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

    def _request(self, method, path, **kwargs):
        url = f"{self.base_url}{path}"
        timeout = kwargs.pop("timeout", 60)
        resp = self.session.request(method, url, timeout=timeout, **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise CloudAPIError(f"{method} {path} failed ({resp.status_code}): {detail}")
        return resp

    # --- Workflow submission ---

    def submit_prompt(self, workflow_dict):
        resp = self._request("POST", "/api/prompt", json={"prompt": workflow_dict})
        data = resp.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise CloudAPIError(f"No prompt_id in response: {data}")
        return prompt_id

    def get_job_status(self, prompt_id):
        return self._request("GET", f"/api/job/{prompt_id}/status").json()

    # The OpenAPI spec lists one set of status names but the live API returns
    # others (e.g. "success" / "queued_waiting" / "executing"). Accept both.
    TERMINAL_STATES = {"completed", "success", "error", "failed", "cancelled"}
    SUCCESS_STATES = {"completed", "success"}

    def poll_until_done(self, prompt_id, interval=3.0, timeout=600, on_status=None):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            status = self.get_job_status(prompt_id)
            state = status.get("status")
            if on_status and state != last:
                on_status(state, status)
                last = state
            if state in self.TERMINAL_STATES:
                return status
            time.sleep(interval)
        raise CloudAPIError(f"Timed out after {timeout}s waiting for prompt {prompt_id}")

    def get_history(self, prompt_id):
        return self._request("GET", f"/api/history_v2/{prompt_id}").json()

    # --- File operations ---

    def download_view(self, filename, file_type="output"):
        resp = self._request(
            "GET", "/api/view",
            params={"filename": filename, "type": file_type},
            allow_redirects=True,
            timeout=120,
        )
        return resp.content

    def upload_image(self, pil_image, filename="upload.png"):
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        files = {"image": (filename, buf, "image/png")}
        data = {"type": "input"}
        resp = self._request("POST", "/api/upload/image", files=files, data=data, timeout=120)
        out = resp.json()
        return out.get("name", filename)

    # --- Model browsing ---

    def list_model_folders(self):
        return self._request("GET", "/api/experiment/models").json()

    def list_models(self, folder):
        return self._request("GET", f"/api/experiment/models/{folder}").json()
