import json
import os
import sys
import urllib.error
from http.server import BaseHTTPRequestHandler
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import analyze_with_anthropic  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", os.getenv("ALLOWED_ORIGIN", "*"))
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            job_description = str(payload.get("job_description", "")).strip()
            resume = str(payload.get("resume", "")).strip()

            if not job_description:
                self.send_json(400, {"error": "Job description is required."})
                return
            if not resume:
                self.send_json(400, {"error": "Resume text is required."})
                return

            self.send_json(200, analyze_with_anthropic(job_description, resume))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            self.send_json(exc.code, {"error": f"Anthropic API error: {details}"})
        except (json.JSONDecodeError, ValueError) as exc:
            self.send_json(400, {"error": f"Invalid request or AI response: {exc}"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})
