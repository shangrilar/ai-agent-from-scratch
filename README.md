# Build an AI Agent from Scratch

Companion code repository for Manning Publications' [*Build an AI Agent from Scratch*](https://www.manning.com/books/build-an-ai-agent-from-scratch).

## Structure

```
scratch_agents/          # Final package (complete through CH10)
  types.py              # Message, ToolCall, ToolResult, Event, ContentItem
  context.py            # ExecutionContext, AgentResult, PendingToolCall, ToolConfirmation
  llm.py                # LlmRequest, LlmResponse, LlmClient
  agent.py              # Agent (ReAct loop)
  rag.py                # Embeddings, chunking, vector search
  callbacks.py          # approval_callback, search_compressor
  planning.py           # Task, create_tasks, reflection
  skills.py             # SkillInfo, discover_skills, generate_skills_prompt
  transfer.py           # create_transfer_tool
  remote.py             # RemoteAgent (A2A)
  a2a_server.py         # MathAgentExecutor
  tools/                # Tool modules
  memory/               # Session, long-term memory, context optimization
  workflows/            # Sequential, Parallel, Loop
  eval/                 # GAIA benchmark, evaluation prompts

notebooks/              # Chapter notebooks
  ch02/                 # LLM API Basics
  ch03/                 # Tools and Function Calling
  ch04/                 # ReAct Agent (+ chapter snapshot code)
  ch05/                 # RAG and File Tools (+ chapter snapshot code)
  ch06/                 # Memory Systems (+ chapter snapshot code)
  ch07/                 # Planning and Reflection
  ch08/                 # Code Execution (+ chapter snapshot code)
  ch09/                 # Multi-Agent Systems (+ chapter snapshot code)
  ch10/                 # Evaluation
```

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Set up API keys in .env
cp .env.example .env
# Edit .env and add your API keys

# Launch Jupyter Lab
uv run jupyter lab
```

## API Keys

Create a `.env` file in the project root with the following keys:

```
OPENAI_API_KEY=sk-...          # Required for all chapters
ANTHROPIC_API_KEY=sk-ant-...   # Required for CH02 Anthropic examples
TAVILY_API_KEY=tvly-...        # Required for CH03 web search
HF_TOKEN=hf_...                # Required for CH02 GAIA benchmark
E2B_API_KEY=e2b_...            # Required for CH08 code execution
```

At minimum, you need `OPENAI_API_KEY` to follow along with the examples.

## Chapters

| Chapter | Topic | Key Modules |
|---------|-------|-------------|
| CH02 | LLM API Basics | eval/gaia.py |
| CH03 | Tools and Function Calling | tools/helpers.py, tools/calculator.py, tools/search.py |
| CH04 | ReAct Agent | types.py, context.py, llm.py, agent.py, tools/base.py |
| CH05 | RAG and File Tools | rag.py, callbacks.py, tools/file_tools.py |
| CH06 | Memory Systems | memory/session.py, memory/long_term.py, memory/context_optimizer.py |
| CH07 | Planning and Reflection | planning.py |
| CH08 | Code Execution | tools/code_execution.py, skills.py |
| CH09 | Multi-Agent Systems | workflows/, transfer.py, tools/agent_tool.py |
| CH10 | Evaluation | eval/prompts.py |

## Chapter Snapshot Files

Some notebook directories (ch04, ch05, ch06, ch08, ch09) contain `.py` snapshot files that represent the state of core modules *at that chapter*. This lets each chapter's notebook use the version of the code that matches what has been introduced so far, without exposing features from later chapters.
