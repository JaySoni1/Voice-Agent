# ARIA — Voice-Based AI Agent with Memory & Tools

A full-stack voice AI agent that manages your To-Do list and remembers
important context .

```
voice-agent/
├── backend/          ← Python FastAPI server
│   ├── main.py       ← Agent loop, tools, API endpoints
│   ├── requirements.txt
│   ├── .env.example
│   └── venv/
└── frontend/
    └── index.html    ← Single-file UI (voice + chat + sidebar)
```

## Quick Start

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

Open `frontend/index.html` in any modern browser 

---

## Architecture

```
Browser (index.html)
│  Web Speech API (mic → text, text → speech)
│  Fetch API
▼
FastAPI (main.py :8000)
│  POST /chat  →  Anthropic Claude claude-sonnet-4-20250514
│                  + Tool calling (agentic loop)
│                    ├── add_todo / update_todo / delete_todo / list_todos
│                    └── save_memory / recall_memories
│  GET  /todos
│  GET  /memories
└── DELETE /reset
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send a message to the agent |
| GET | `/todos` | List all todos |
| GET | `/memories` | List all saved memories |
| DELETE | `/reset` | Clear all data |
| GET | `/health` | Health check |

## Features

| Feature | Detail |
|---------|--------|
| 🎤 Voice input | Web Speech API, interim transcripts |
| 🔊 TTS output | Web Speech Synthesis, toggleable |
| ✅ To-Do CRUD | add / update / delete / list via Claude tools |
| 🧠 Memory | Agent proactively saves & recalls user facts |
| 💬 Multi-turn | Full conversation history per session |
| 🔄 Agent loop | Keeps calling tools until task complete |

## Example Phrases

- *"Add a high-priority task: submit report by Friday"*
- *"Mark task abc123 as done"*
- *"What pending tasks do I have?"*
- *"Remember that my dog's name is Max"*
- *"What do you know about me?"*
- *"Delete all completed tasks"*
