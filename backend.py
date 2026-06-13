import json
import os
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def load_env():
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def clamp_score(value, default=0):
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return default


def extract_keywords(text, limit):
    words = re.findall(r"[A-Za-z][A-Za-z+#.-]{2,}", text.lower())
    stop_words = {
        "and", "the", "for", "with", "you", "are", "our", "this", "that", "will",
        "have", "from", "your", "job", "role", "work", "team", "able", "using",
        "about", "into", "their", "they", "them", "experience", "requirements",
    }
    counts = {}
    for word in words:
        if word in stop_words:
            continue
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word.title() for word, _ in ranked[:limit]]


def mock_analysis(job_description, resume):
    jd_keywords = extract_keywords(job_description, 9)
    resume_lc = resume.lower()
    matched = [kw for kw in jd_keywords if kw.lower() in resume_lc][:5]
    missing = [kw for kw in jd_keywords if kw not in matched][:4]
    fit = clamp_score(58 + min(32, len(matched) * 7) - min(18, len(missing) * 3), 72)

    return {
        "fit_score": fit,
        "grade": "Strong Match" if fit >= 80 else "Good Match" if fit >= 65 else "Partial Match",
        "verdict": "Your resume has a solid foundation and needs sharper keyword alignment.",
        "skills_score": clamp_score(fit + 4),
        "experience_score": clamp_score(fit - 2),
        "keywords_score": clamp_score(55 + len(matched) * 8),
        "matched_keywords": matched or jd_keywords[:3],
        "missing_keywords": missing,
        "resume": (
            "Tailored Resume\n\n"
            "Professional Summary\n"
            "Results-focused candidate with experience aligned to the target role. "
            "Emphasizes relevant achievements, measurable outcomes, and role-specific skills.\n\n"
            "Core Strengths\n"
            f"- {', '.join((matched or jd_keywords[:5])[:5])}\n"
            "- Cross-functional collaboration\n"
            "- Clear communication and execution\n\n"
            "Experience\n"
            "Rewrite each role with impact bullets that connect your existing accomplishments "
            "to the responsibilities and outcomes in the job description."
        ),
        "cover_letter": (
            "Dear Hiring Manager,\n\n"
            "I am excited to apply for this role. My background maps well to the needs in the "
            "job description, especially the combination of execution, communication, and "
            "continuous improvement.\n\n"
            "In my previous work, I have built a track record of turning ambiguous goals into "
            "clear deliverables and measurable results. I would bring that same focus to your team.\n\n"
            "Thank you for your time and consideration. I would welcome the chance to discuss "
            "how my experience can support your goals."
        ),
        "interview_questions": [
            {"category": "Behavioral", "question": "Tell me about a time you adapted quickly to a new requirement."},
            {"category": "Behavioral", "question": "Describe a project where you improved a process or outcome."},
            {"category": "Role-specific", "question": "Which parts of this job description match your strongest experience?"},
            {"category": "Role-specific", "question": "How would you prioritize your first 30 days in this role?"},
            {"category": "Technical", "question": "What tools or methods would you use to solve the core problems in this role?"},
            {"category": "Technical", "question": "How do you measure whether your work is successful?"},
            {"category": "Culture fit", "question": "What type of team environment helps you do your best work?"},
            {"category": "Culture fit", "question": "How do you handle feedback when deadlines are tight?"},
        ],
    }


def build_prompt(job_description, resume):
    return f"""You are an expert career coach. Given a job description and resume, respond ONLY with a valid JSON object. Do not include markdown fences or a preamble.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume}

JSON schema:
{{
  "fit_score": integer 0-100,
  "grade": "Excellent Match"|"Strong Match"|"Good Match"|"Partial Match"|"Weak Match",
  "verdict": "one concise sentence about overall fit, max 18 words",
  "skills_score": integer 0-100,
  "experience_score": integer 0-100,
  "keywords_score": integer 0-100,
  "matched_keywords": ["up to 5 keywords from JD that appear in resume"],
  "missing_keywords": ["up to 4 important keywords from JD missing from resume"],
  "resume": "full rewritten resume as plain text, optimized for this role",
  "cover_letter": "professional cover letter as plain text, 3-4 paragraphs",
  "interview_questions": [
    {{"category":"Behavioral"|"Technical"|"Role-specific"|"Culture fit","question":"string"}}
  ]
}}"""


def parse_model_json(raw_text):
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    return json.loads(cleaned)


def analyze_with_anthropic(job_description, resume):
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key or os.getenv("MOCK_AI", "").lower() in {"1", "true", "yes"}:
        return mock_analysis(job_description, resume)

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    payload = {
        "model": model,
        "max_tokens": 3000,
        "messages": [{"role": "user", "content": build_prompt(job_description, resume)}],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))

    raw = "".join(block.get("text", "") for block in data.get("content", []))
    return parse_model_json(raw)


class ApplyFastHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path not in {"/", "/index.html"}:
            self.send_error(404, "Not found")
            return

        html = (BASE_DIR / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def do_POST(self):
        if self.path != "/api/analyze":
            self.send_error(404, "Not found")
            return

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


def main():
    load_env()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), ApplyFastHandler)
    print(f"ApplyFast backend running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
