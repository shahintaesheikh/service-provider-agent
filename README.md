# Service Provider Agent

An agent that helps Santa Barbara homeowners find the right home service provider through a conversational chat interface.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env with your keys (OPENAI_API_KEY, TYPESAFE_API_KEY, GOOGLE_API_KEY)

# 3. Start the server
python web/voice_server.py

# 4. Open your browser to http://localhost:8000
```

## Try It

Type a home problem in the chat:

> "There's water leaking from my kitchen sink at 123 State Street in Santa Barbara"

The agent will:
- Classify the issue (Plumbing)
- Find local providers in Santa Barbara
- Show you the best match with rating and contact info

## Architecture

```
User chat → LangGraph pipeline
  └── ChatOpenAI extracts entities
  └── Jev classifies category + subcategory
  └── Google Places API finds providers
  └── HITL gate checks for human review
  └── Lead result displayed in UI
```

## Project Structure

```
agent/              LangGraph pipeline nodes
web/                Web frontend (iPhone-style chat UI)
tools/              Utility modules
  provider_search.py   Google Places API integration
  jev_utils.py         Jev TypeSafe decision functions
  hitl_utils.py        HITL risk scoring + trainer log
```