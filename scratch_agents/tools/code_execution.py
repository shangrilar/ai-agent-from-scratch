"""Code execution tools using E2B sandbox."""

import json

from scratch_agents.tools.base import tool
from scratch_agents.context import ExecutionContext


@tool(
    name="execute_python",
    description="Execute Python code in a sandboxed environment. "
                "Use this to perform calculations, data processing, or any Python operations."
)
def execute_python(context: ExecutionContext, code: str) -> str:
    """Execute Python code in the E2B sandbox.

    Args:
        context: Execution context with code_env
        code: Python code to execute
    """
    if context.code_env is None:
        raise RuntimeError("No code execution environment available.")
    sandbox = context.code_env
    execution = sandbox.run_code(code)
    return json.dumps(execution.to_json(), indent=2, ensure_ascii=False)


@tool(name="bash_tool", description="Execute a bash command in a sandboxed environment.")
def bash_tool(context: ExecutionContext, command: str) -> str:
    """Execute a bash command in the E2B sandbox.

    Args:
        context: Execution context with code_env
        command: Bash command to execute
    """
    if context.code_env is None:
        raise RuntimeError("No code execution environment available.")
    sandbox = context.code_env

    try:
        result = sandbox.commands.run(command)
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"STDERR: {result.stderr}")
        return "\n".join(output_parts) if output_parts else "Command completed (no output)"
    except Exception as e:
        return f"Command error: {str(e)}"


@tool
def upload_file(context: ExecutionContext, local_path: str, sandbox_path: str = None) -> str:
    """Upload a local file to the E2B sandbox.

    Args:
        context: Execution context with code_env
        local_path: Path to the local file
        sandbox_path: Destination path in sandbox (defaults to /home/user/)
    """
    import os

    sandbox = context.code_env
    if sandbox is None:
        return "Error: No code execution environment available"

    if sandbox_path is None:
        sandbox_path = f"/home/user/{os.path.basename(local_path)}"

    try:
        with open(local_path, "rb") as f:
            sandbox.files.write(sandbox_path, f.read())
        return f"File uploaded to {sandbox_path}"
    except Exception as e:
        return f"Upload error: {str(e)}"
