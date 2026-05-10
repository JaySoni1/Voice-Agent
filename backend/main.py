"""
Voice-Based AI Agent with Memory & Tools
Backend: FastAPI + Groq + Tool Calling
"""

import json
import uuid
import os
import re
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice AI Agent API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Models to try in order — first one that works wins
MODELS = [
    "llama3-groq-70b-8192-tool-use-preview",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
]

# ─────────────────────────────────────────────
# In-memory stores
# ─────────────────────────────────────────────
todo_store: dict[str, dict] = {}
memory_store: list[dict]    = []
conversation_history: list[dict] = []

# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    reply: str
    tool_calls: list = []
    memories_used: list = []

# ─────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_todo",
            "description": "Add a new task to the To-Do list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":       {"type": "string", "description": "Short task title"},
                    "description": {"type": "string", "description": "Optional details"},
                    "priority":    {"type": "string", "enum": ["low", "medium", "high"]},
                    "due_date":    {"type": "string", "description": "Due date YYYY-MM-DD (optional)"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_todo",
            "description": "Update an existing To-Do item by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id":          {"type": "string"},
                    "title":       {"type": "string"},
                    "description": {"type": "string"},
                    "priority":    {"type": "string", "enum": ["low", "medium", "high"]},
                    "due_date":    {"type": "string"},
                    "status":      {"type": "string", "enum": ["pending", "in_progress", "done"]},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_todo",
            "description": "Delete a To-Do item by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_todos",
            "description": "List all To-Do items, optionally filtered by status or priority.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status":   {"type": "string", "enum": ["pending", "in_progress", "done", "all"]},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "all"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save an important fact or event about the user to long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content":  {"type": "string"},
                    "category": {"type": "string", "description": "preference / event / goal / personal"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memories",
            "description": "Recall stored memories about the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":    {"type": "string"},
                    "category": {"type": "string"},
                },
            },
        },
    },
]

# ─────────────────────────────────────────────
# Tool executor functions
# ─────────────────────────────────────────────
def add_todo(title: str, description: str = "", priority: str = "medium", due_date: str = "") -> dict:
    task_id = str(uuid.uuid4())[:8]
    task = {
        "id": task_id, "title": title, "description": description,
        "priority": priority, "due_date": due_date,
        "status": "pending", "created_at": datetime.now().isoformat(),
    }
    todo_store[task_id] = task
    return {"success": True, "message": f"Task '{title}' added with ID {task_id}.", "task": task}

def update_todo(id: str, **kwargs) -> dict:
    if id not in todo_store:
        return {"success": False, "message": f"Task ID {id} not found."}
    for k, v in kwargs.items():
        if v is not None and k in todo_store[id]:
            todo_store[id][k] = v
    todo_store[id]["updated_at"] = datetime.now().isoformat()
    return {"success": True, "message": f"Task {id} updated.", "task": todo_store[id]}

def delete_todo(id: str) -> dict:
    if id not in todo_store:
        return {"success": False, "message": f"Task ID {id} not found."}
    task = todo_store.pop(id)
    return {"success": True, "message": f"Task '{task['title']}' deleted."}

def list_todos(status: str = "all", priority: str = "all") -> dict:
    tasks = list(todo_store.values())
    if status   != "all": tasks = [t for t in tasks if t["status"]   == status]
    if priority != "all": tasks = [t for t in tasks if t["priority"] == priority]
    return {"success": True, "count": len(tasks), "tasks": tasks}

def save_memory(content: str, category: str = "general") -> dict:
    entry = {
        "id": str(uuid.uuid4())[:8], "content": content,
        "category": category, "created_at": datetime.now().isoformat(),
    }
    memory_store.append(entry)
    return {"success": True, "message": "Memory saved.", "entry": entry}

def recall_memories(query: str = "", category: str = "") -> dict:
    results = list(memory_store)
    if category: results = [m for m in results if m.get("category", "").lower() == category.lower()]
    if query:
        q = query.lower()
        results = [m for m in results if q in m["content"].lower()]
    return {"success": True, "count": len(results), "memories": results}

TOOL_MAP = {
    "add_todo": add_todo, "update_todo": update_todo,
    "delete_todo": delete_todo, "list_todos": list_todos,
    "save_memory": save_memory, "recall_memories": recall_memories,
}

# ─────────────────────────────────────────────
# Safe JSON parser
# ─────────────────────────────────────────────
def safe_parse_args(raw: str) -> dict:
    if not raw or raw.strip() == "":
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    try:
        fixed = re.sub(r',\s*}', '}', raw)
        fixed = re.sub(r',\s*]', ']', fixed)
        return json.loads(fixed)
    except Exception:
        pass
    logger.warning(f"Could not parse tool args: {raw}")
    return {}

# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are ARIA — a voice-based AI assistant with memory and task management.

## Personality
- Warm, concise, helpful. Speak naturally — responses will be converted to speech.
- Keep replies brief (1-3 sentences for simple tasks).
- Confirm actions: "Done! I've added X to your list."

## Tools available
1. add_todo        — When user says "I need to / remind me to / add a task"
2. update_todo     — Update status, priority, or details by task ID
3. delete_todo     — Remove a task by ID
4. list_todos      — Show tasks filtered by status or priority
5. save_memory     — Save personal facts, preferences, events user mentions
6. recall_memories — Retrieve past context before answering personal questions

## Important rules
- Always use valid JSON for tool arguments. Never wrap JSON in XML tags.
- Save personal info proactively (names, birthdays, preferences, goals).
- Confirm every tool action in your reply.
- Today: {datetime.now().strftime("%A, %B %d, %Y")}
"""

# ─────────────────────────────────────────────
# Call Groq with model fallback
# ─────────────────────────────────────────────
def call_groq(messages: list, use_tools: bool = True) -> tuple:
    """Try each model in order, return (response, model_used)."""
    last_error = None
    for model in MODELS:
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=1024,
                temperature=0.1,
            )
            if use_tools:
                kwargs["tools"] = TOOLS
                kwargs["tool_choice"] = "auto"

            response = client.chat.completions.create(**kwargs)
            logger.info(f"✅ Model used: {model}")
            return response, model
        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")
            last_error = e
            continue
    raise last_error

# ─────────────────────────────────────────────
# Core agent loop
# ─────────────────────────────────────────────
def run_agent(user_message: str) -> ChatResponse:
    conversation_history.append({"role": "user", "content": user_message})

    tool_calls_log: list = []
    memories_used:  list = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(conversation_history)

    max_iterations = 5
    iteration = 0
    msg = None

    while iteration < max_iterations:
        iteration += 1

        try:
            response, model_used = call_groq(messages, use_tools=True)
        except Exception as e:
            logger.error(f"All models failed: {e}")
            # Last resort: call without tools
            try:
                response, _ = call_groq(messages, use_tools=False)
                reply_text = response.choices[0].message.content or "Sorry, I had trouble with that."
                conversation_history.append({"role": "assistant", "content": reply_text})
                return ChatResponse(reply=reply_text, tool_calls=tool_calls_log, memories_used=memories_used)
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Groq error: {str(e2)}")

        msg           = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        logger.info(f"Iter {iteration} | finish: {finish_reason} | tools: {len(msg.tool_calls or [])}")

        # Append assistant message
        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # No tool calls → done
        if not msg.tool_calls:
            break

        # Execute tools
        for tc in msg.tool_calls:
            tool_name  = tc.function.name
            tool_input = safe_parse_args(tc.function.arguments)

            logger.info(f"Tool: {tool_name} | Args: {tool_input}")
            fn     = TOOL_MAP.get(tool_name)
            result = fn(**tool_input) if fn else {"error": f"Unknown tool: {tool_name}"}

            tool_calls_log.append({"tool": tool_name, "input": tool_input, "result": result})
            if tool_name == "recall_memories" and result.get("memories"):
                memories_used.extend(result["memories"])

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result),
            })

    reply_text = (msg.content or "") if msg else "Done!"
    conversation_history.append({"role": "assistant", "content": reply_text})

    if len(conversation_history) > 40:
        conversation_history[:] = conversation_history[-40:]

    return ChatResponse(reply=reply_text, tool_calls=tool_calls_log, memories_used=memories_used)


# ─────────────────────────────────────────────
# API endpoints
# ─────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        return run_agent(req.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/todos")
async def get_todos():
    return {"todos": list(todo_store.values())}

@app.get("/memories")
async def get_memories():
    return {"memories": memory_store}

@app.delete("/reset")
async def reset():
    todo_store.clear()
    memory_store.clear()
    conversation_history.clear()
    return {"message": "All data reset."}

@app.get("/health")
async def health():
    return {"status": "ok", "todos": len(todo_store), "memories": len(memory_store)}