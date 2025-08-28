This repository contains the code for Manning Publications' "Build an AI Agent From Scratch".

### Install uv (docs: https://docs.astral.sh/uv/getting-started/installation/)
- macOS/Linux (official script):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Homebrew (macOS):
```bash
brew install uv
```
- Verify installation:
```bash
uv --version
```

### Create a virtual environment (uv venv)
```bash
uv venv
source .venv/bin/activate
```

### Install dependencies with uv
```bash
uv pip install -r requirements.txt
```

### Install scratch_agents package (Required for Chapter 4+)
- For Chapter 4 and beyond, install the scratch_agents package in editable mode:
```bash
uv pip install -e .
```

### Environment variables
- Copy the example env file and set your API keys:
```bash
cp .env.example .env
```
- Open `.env` and provide the necessary keys (e.g., `OPENAI_API_KEY=...`).
