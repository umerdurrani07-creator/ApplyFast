import json
import os
import re
import urllib.error
import urllib.request
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MAX_JD_CHARS = 12000
DEFAULT_MAX_CONTEXT_CHARS = 18000
DEFAULT_CHUNK_CHARS = 1800
DEFAULT_CHUNK_OVERLAP = 250
DEFAULT_MAX_CHUNKS = 8


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


def decode_text_file(file_bytes):
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def extract_pdf_text(file_bytes):
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF support requires installing dependencies from requirements.txt.") from exc

    reader = PdfReader(BytesIO(file_bytes))
    text = "\n".join((page.extract_text() or "").strip() for page in reader.pages)
    return text.strip()


def extract_docx_text(file_bytes):
    try:
        from docx import Document
    except ImportError as exc:
        raise ValueError("DOCX support requires installing dependencies from requirements.txt.") from exc

    document = Document(BytesIO(file_bytes))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    return text.strip()


def extract_resume_text(file_bytes, filename="", content_type=""):
    name = (filename or "").lower()
    kind = (content_type or "").lower()

    if name.endswith(".pdf") or "pdf" in kind:
        text = extract_pdf_text(file_bytes)
    elif name.endswith(".docx") or "wordprocessingml.document" in kind:
        text = extract_docx_text(file_bytes)
    elif name.endswith(".txt") or kind.startswith("text/"):
        text = decode_text_file(file_bytes).strip()
    elif name.endswith(".doc"):
        raise ValueError("Legacy .doc files are not supported. Please upload PDF, DOCX, or TXT.")
    else:
        text = decode_text_file(file_bytes).strip()

    if not text:
        raise ValueError("Could not extract text from the uploaded resume. Try a text-based PDF, DOCX, or TXT file.")
    return text


def parse_multipart_form(content_type, body):
    headers = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    message = BytesParser(policy=policy.default).parsebytes(headers + body)
    fields = {}
    files = {}

    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files[name] = {
                "filename": filename,
                "content_type": part.get_content_type(),
                "content": payload,
            }
        else:
            fields[name] = decode_text_file(payload).strip()

    return fields, files


def get_analysis_inputs(headers, rfile):
    content_type = headers.get("Content-Type", "")
    length = int(headers.get("Content-Length", "0"))
    body = rfile.read(length)

    if content_type.startswith("multipart/form-data"):
        fields, files = parse_multipart_form(content_type, body)
        job_description = fields.get("job_description", "").strip()
        uploaded = files.get("resume_file") or files.get("resume")
        if not uploaded:
            raise ValueError("Resume file is required.")
        resume = extract_resume_text(uploaded["content"], uploaded["filename"], uploaded["content_type"])
        return job_description, resume

    payload = json.loads(body.decode("utf-8"))
    return str(payload.get("job_description", "")).strip(), str(payload.get("resume", "")).strip()


def clamp_score(value, default=0):
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return default


def env_int(name, default, minimum=None, maximum=None):
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def normalize_space(text):
    return re.sub(r"\s+", " ", text or "").strip()


def limit_text(text, max_chars):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() + "\n\n[Truncated for token safety.]"


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


def retrieval_terms(job_description):
    terms = [term.lower() for term in extract_keywords(job_description, 80)]
    phrases = re.findall(r"\b[A-Z][A-Za-z+#.-]*(?:\s+[A-Z][A-Za-z+#.-]*){1,3}\b", job_description or "")
    for phrase in phrases:
        phrase = phrase.lower()
        if phrase not in terms and len(phrase) <= 60:
            terms.append(phrase)
    return terms


def chunk_text(text, chunk_chars=None, overlap=None):
    chunk_chars = chunk_chars or env_int("RAG_CHUNK_CHARS", DEFAULT_CHUNK_CHARS, 600, 5000)
    overlap = overlap if overlap is not None else env_int("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP, 0, 1000)
    text = normalize_space(text)
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_chars, text_len)
        if end < text_len:
            soft_end = text.rfind(". ", start, end)
            if soft_end > start + chunk_chars // 2:
                end = soft_end + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - overlap, start + 1)
    return chunks


def score_chunk(chunk, terms):
    chunk_lc = chunk.lower()
    score = 0
    for term in terms:
        if not term:
            continue
        hits = chunk_lc.count(term)
        if hits:
            score += hits * (4 if " " in term else 1)

    # Resume sections with durable signal should survive even if wording differs from the JD.
    for section in ("experience", "projects", "skills", "education", "certifications", "summary"):
        if section in chunk_lc:
            score += 2
    return score


def retrieve_resume_context(job_description, resume_text):
    max_context_chars = env_int("RAG_MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS, 6000, 60000)
    max_chunks = env_int("RAG_MAX_CHUNKS", DEFAULT_MAX_CHUNKS, 3, 20)
    chunks = chunk_text(resume_text)
    if not chunks:
        raise ValueError("Could not prepare resume text for analysis.")

    terms = retrieval_terms(job_description)
    ranked = sorted(
        enumerate(chunks),
        key=lambda item: (score_chunk(item[1], terms), -item[0]),
        reverse=True,
    )

    selected = sorted(ranked[:max_chunks], key=lambda item: item[0])
    context_parts = []
    used = 0
    for index, chunk in selected:
        header = f"[Resume excerpt {index + 1}/{len(chunks)}]\n"
        remaining = max_context_chars - used - len(header)
        if remaining <= 0:
            break
        safe_chunk = limit_text(chunk, remaining)
        context_parts.append(header + safe_chunk)
        used += len(header) + len(safe_chunk)

    if not context_parts:
        context_parts.append(limit_text(chunks[0], max_context_chars))

    return "\n\n".join(context_parts), {
        "resume_chars": len(resume_text),
        "chunk_count": len(chunks),
        "selected_chunks": len(context_parts),
        "context_chars": sum(len(part) for part in context_parts),
    }


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


def build_prompt(job_description, resume_context, retrieval_meta):
    return f"""You are an expert career coach. Given a job description and resume, respond ONLY with a valid JSON object. Do not include markdown fences or a preamble.

JOB DESCRIPTION:
{job_description}

RESUME CONTEXT:
The resume may have been shortened with retrieval to control token usage. Use only the supplied excerpts. Do not invent missing employment history, dates, tools, employers, degrees, or certifications.

Retrieval metadata:
- Original resume characters: {retrieval_meta["resume_chars"]}
- Resume chunks created: {retrieval_meta["chunk_count"]}
- Chunks supplied: {retrieval_meta["selected_chunks"]}
- Supplied context characters: {retrieval_meta["context_chars"]}

{resume_context}

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


def env_flag(name):
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def analyze_with_openrouter(job_description, resume):
    job_description = limit_text(job_description, env_int("MAX_JOB_DESCRIPTION_CHARS", DEFAULT_MAX_JD_CHARS, 2000, 50000))
    resume_context, retrieval_meta = retrieve_resume_context(job_description, resume)
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if env_flag("MOCK_AI"):
        result = mock_analysis(job_description, resume_context)
        result["_retrieval"] = retrieval_meta
        return result
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured. Add it to your Vercel Environment Variables and redeploy.")

    model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
    payload = {
        "model": model,
        "max_tokens": env_int("OPENROUTER_MAX_TOKENS", 3000, 1000, 8000),
        "messages": [{"role": "user", "content": build_prompt(job_description, resume_context, retrieval_meta)}],
        "reasoning": {"enabled": False},
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "ApplyFast"),
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))

    message = data.get("choices", [{}])[0].get("message", {})
    raw = message.get("content") or ""
    result = parse_model_json(raw)
    result["_retrieval"] = retrieval_meta
    return result


def openrouter_health():
    return {
        "ok": True,
        "openrouter_key_configured": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
        "mock_ai": env_flag("MOCK_AI"),
        "model": os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
    }


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
        if self.path == "/api/analyze":
            self.send_json(200, openrouter_health())
            return

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
            job_description, resume = get_analysis_inputs(self.headers, self.rfile)

            if not job_description:
                self.send_json(400, {"error": "Job description is required."})
                return
            if not resume:
                self.send_json(400, {"error": "Resume text is required."})
                return

            self.send_json(200, analyze_with_openrouter(job_description, resume))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            self.send_json(exc.code, {"error": f"OpenRouter API error: {details}"})
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
