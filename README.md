# ApplyFast

ApplyFast is a small AI-powered job application assistant. It accepts a job description and an uploaded resume, then generates:

- Resume fit score
- Matched and missing keywords
- Tailored resume rewrite
- Cover letter
- Interview practice questions

The backend uses OpenRouter for AI generation and includes PDF, DOCX, and TXT resume parsing.

## Project Structure

```txt
.
├── api/
│   └── analyze.py      # Vercel serverless API handler
├── backend.py          # Local backend server and shared analysis logic
├── index.html          # Frontend UI
├── requirements.txt    # Python dependencies
└── vercel.json         # Vercel function config
```

## Local Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=google/gemma-4-31b-it:free
OPENROUTER_SITE_URL=http://localhost:8000
OPENROUTER_APP_NAME=ApplyFast
MOCK_AI=false
```

Run the local server:

```bash
python backend.py
```

Open:

```txt
http://127.0.0.1:8000
```

## API Health Check

The analysis endpoint supports a safe GET health check:

```txt
GET /api/analyze
```

Example response:

```json
{
  "ok": true,
  "openrouter_key_configured": true,
  "mock_ai": false,
  "model": "google/gemma-4-31b-it:free"
}
```

If `openrouter_key_configured` is `false`, the backend cannot call OpenRouter.

## Vercel Deployment

Set these environment variables in Vercel Project Settings:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=google/gemma-4-31b-it:free
OPENROUTER_SITE_URL=https://your-vercel-domain.vercel.app
OPENROUTER_APP_NAME=ApplyFast
MOCK_AI=false
```

Then redeploy the project.

Important: Vercel does not read your local `.env` file. Environment variables must be configured in the Vercel dashboard.

## Performance Notes

Long responses can take a minute or more because the app asks the model to produce a score, resume rewrite, cover letter, and interview questions in one request.

To reduce latency, lower the token limit:

```env
OPENROUTER_MAX_TOKENS=1800
```

You can also use a faster OpenRouter model.

## Mock Mode

Set this only when you want deterministic fallback output without calling OpenRouter:

```env
MOCK_AI=true
```

For production, use:

```env
MOCK_AI=false
```

## Security

Never commit `.env` or expose API keys in chat, screenshots, logs, or frontend code. If a key is exposed, revoke it and create a new one.
