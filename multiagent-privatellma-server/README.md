# How to use Claude Code + local LLM setup

## What is Claude Code?
Claude Code is a terminal tool (like a smart command-line assistant).
You run it in your terminal and it can read your files, write code,
run commands, and help you build things — all by chatting.

## Install Claude Code
```
npm install -g @anthropic-ai/claude-code
```
Needs Node.js 18+. Check: `node --version`

---

## Mode 1 — Real Claude (needs API key, costs money, very powerful)
```powershell
# Set your key once
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
claude
```
Use this for: complex refactoring, multi-file projects, hard bugs.

---

## Mode 2 — Local LLM via litellm proxy (free, limited)
Step 1: start Ollama
```
ollama serve
```

Step 2: start litellm proxy (separate terminal)
```
litellm --model ollama/phi4-mini --port 8080
```

Step 3: run Claude Code pointing to proxy (separate terminal)
```powershell
$env:ANTHROPIC_BASE_URL = "http://localhost:8080"
$env:ANTHROPIC_API_KEY  = "fake-key"
claude
```

Limitation: local models don't support Claude's tool-use format well,
so file editing and commands may not work. Use for Q&A only in this mode.

---

## Claude Code commands inside the chat
Once `claude` is running, type:

| What you type          | What happens                          |
|------------------------|---------------------------------------|
| `explain this file`    | Claude reads + explains current dir   |
| `/add file.py`         | Adds a file to Claude's context       |
| `fix the bug in app.py`| Claude edits the file directly        |
| `write tests for utils.py` | Claude creates a test file        |
| `what does this code do` | Explains selected code              |
| `/clear`               | Clears conversation history           |
| `/exit`                | Quit                                  |

---

## Daily workflow recommendation for your setup

```
Morning coding:
  → open VS Code
  → Ctrl+L → chat with phi4-mini via Continue (free, fast)
  → for autocomplete: just type, qwen2.5-coder suggests

Hard problem / stuck:
  → open terminal
  → set ANTHROPIC_API_KEY
  → run `claude`
  → use real Claude Code (costs ~$0.01-0.05 per session)

Quick terminal questions:
  → ollama run phi4-mini "your question here"
```

---

## Project file structure
```
multi_agent_llm/
├── requirements.txt      ← pip install -r requirements.txt
├── agents.py             ← main multi-agent system (run this)
├── quantization_demo.py  ← understand quantization with code
└── README.md             ← this file
```

## Quick start
```bash
# 1. install deps
pip install -r requirements.txt

# 2. make sure ollama is running with the right models
ollama pull phi4-mini
ollama pull qwen2.5-coder:3b
ollama serve

# 3. understand quantization (optional but recommended)
python quantization_demo.py

# 4. run the multi-agent chat
python agents.py
```
