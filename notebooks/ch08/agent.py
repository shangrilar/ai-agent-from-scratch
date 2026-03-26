"""CH08 snapshot: ReAct agent + code execution + skills
Changes from CH06:
  - __init__: code_execution, skills_path added
  - run(): code_env setup/cleanup (try-finally)
  - _setup_tools(): execute_python, sandbox_executable tool collection
  - _prepare_llm_request(): sandbox/skills prompt added
  - _setup_code_env(), _register_sandbox_tools(), _get_sandbox_tools_prompt() new
  - sub_agents, transfer related not yet added
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type

from pydantic import BaseModel

from scratch_agent.llm import LlmClient, LlmRequest, LlmResponse
from scratch_agent.types import Event, Message, ToolCall, ToolResult
from scratch_agent.tools.base import BaseTool, FunctionTool, tool
from scratch_agent.tools.helpers import format_tool_definition
from scratch_agent.context import (
    AgentResult,
    ExecutionContext,
    PendingToolCall,
    ToolConfirmation,
)

if TYPE_CHECKING:
    from scratch_agent.memory.long_term import TaskMemoryManager
    from scratch_agent.memory.session import BaseSessionManager

logger = logging.getLogger(__name__)


class Agent:
    """Tool-calling agent with ReAct loop, callbacks, sessions, memory,
    code execution, and skills."""

    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
        description: str = "",
        output_type: Optional[Type[BaseModel]] = None,
        # CH05 callbacks
        before_tool_callbacks: list[Callable] | None = None,
        after_tool_callbacks: list[Callable] | None = None,
        # CH06 session & memory
        session_manager: Optional["BaseSessionManager"] = None,
        memory_manager: Optional["TaskMemoryManager"] = None,
        before_llm_callbacks: list[Callable] | None = None,
        # NEW: CH08 code execution
        code_execution: str | None = None,  # "e2b"
        skills_path: str | None = None,
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.description = description
        self.output_type = output_type
        self.output_tool_name: str | None = None
        self.before_tool_callbacks = before_tool_callbacks or []
        self.after_tool_callbacks = after_tool_callbacks or []
        self.before_llm_callbacks = before_llm_callbacks or []
        self.session_manager = session_manager
        self.memory_manager = memory_manager
        self.code_execution = code_execution
        self.skills_path = skills_path
        self._sandbox_tools: List[FunctionTool] = []
        self.tools = self._setup_tools(tools or [])

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #

    async def run(
        self,
        user_input: str | None = None,
        context: ExecutionContext | None = None,
        session_id: str | None = None,
        tool_confirmations: list[ToolConfirmation] | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Execute the agent."""

        session = None
        if session_id and self.session_manager:
            session = await self.session_manager.get_or_create(session_id)

        if context is None:
            context = ExecutionContext(
                session=session,
                session_manager=self.session_manager,
            )
            # Restore previous events from session
            if session and session.events:
                context.events = list(session.events)

        if tool_confirmations:
            await self._process_confirmations(context, tool_confirmations)

        if user_input:
            user_event = Event(
                execution_id=context.execution_id,
                author="user",
                content=[Message(role="user", content=user_input)],
            )
            context.add_event(user_event)

        # NEW: Set up code execution environment
        if self.code_execution == "e2b" and context.code_env is None:
            await self._setup_code_env(context)

        try:
            while not context.final_result and context.current_step < self.max_steps:
                result = await self.step(context, verbose=verbose)

                if result and result.status == "pending":
                    if session and self.session_manager:
                        session.events = list(context.events)
                        await self.session_manager.save(session)
                    return result

                if context.events:
                    last_event = context.events[-1]
                    if self._is_final_response(last_event):
                        context.final_result = self._extract_final_result(last_event)

            if self.memory_manager:
                try:
                    await self.memory_manager.save(context)
                except Exception as e:
                    logger.warning(f"Failed to save memory: {e}")

            if session and self.session_manager:
                session.events = list(context.events)
                await self.session_manager.save(session)

            return AgentResult(output=context.final_result, context=context)
        finally:
            # NEW: Clean up sandbox
            if context.code_env is not None:
                context.code_env.kill()

    async def step(
        self,
        context: ExecutionContext,
        verbose: bool = False,
    ) -> AgentResult | None:
        """Perform one think-act cycle."""
        llm_request = self._prepare_llm_request(context)

        for callback in self.before_llm_callbacks:
            cb_result = callback(context, llm_request)
            if hasattr(cb_result, "__await__"):
                cb_result = await cb_result
            if isinstance(cb_result, LlmResponse):
                llm_response = cb_result
                break
        else:
            llm_response = await self.think(llm_request)

        if verbose:
            self._log_response(llm_response)

        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=llm_response.content,
        )
        context.add_event(response_event)

        tool_calls = [c for c in llm_response.content if isinstance(c, ToolCall)]
        if tool_calls:
            result = await self.act(context, tool_calls)
            if result and result.status == "pending":
                return result

        context.increment_step()
        return None

    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        """Call the LLM to decide the next action."""
        return await self.model.generate(llm_request)

    async def act(
        self,
        context: ExecutionContext,
        tool_calls: List[ToolCall],
    ) -> AgentResult | None:
        """Execute the tools requested by the LLM."""
        tools_dict = {t.name: t for t in self.tools}
        results = []
        pending = []

        for tool_call in tool_calls:
            if tool_call.name not in tools_dict:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[f"Tool '{tool_call.name}' not found"],
                ))
                continue

            tool_obj = tools_dict[tool_call.name]

            if tool_obj.requires_confirmation:
                msg = tool_obj.get_confirmation_message(tool_call.arguments)
                pending.append(PendingToolCall(
                    tool_call=tool_call,
                    confirmation_message=msg,
                ))
                continue

            skip = False
            for cb in self.before_tool_callbacks:
                cb_result = cb(context, tool_call)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    results.append(ToolResult(
                        tool_call_id=tool_call.tool_call_id,
                        name=tool_call.name,
                        status="success",
                        content=[cb_result],
                    ))
                    skip = True
                    break

            if skip:
                continue

            try:
                output = await tool_obj(context, **tool_call.arguments)
                tool_result = ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="success",
                    content=[output],
                )
            except Exception as e:
                tool_result = ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[str(e)],
                )

            for cb in self.after_tool_callbacks:
                cb_result = cb(context, tool_result)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    tool_result = cb_result

            results.append(tool_result)

        if pending:
            context.state["pending_tool_calls"] = [
                p.model_dump() for p in pending
            ]
            if results:
                tool_event = Event(
                    execution_id=context.execution_id,
                    author=self.name,
                    content=results,
                )
                context.add_event(tool_event)
            return AgentResult(
                output=None,
                context=context,
                status="pending",
                pending_tool_calls=pending,
            )

        if results:
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=results,
            )
            context.add_event(tool_event)

        return None

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #

    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        """Build an LlmRequest from the current context."""
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)

        instructions = []
        if self.instructions:
            instructions.append(self.instructions)

        # NEW: Add sandbox tools prompt
        sandbox_prompt = self._get_sandbox_tools_prompt()
        if sandbox_prompt:
            instructions.append(sandbox_prompt)

        # NEW: Add skills prompt
        if self.skills_path:
            try:
                from scratch_agent.skills import discover_skills, generate_skills_prompt
                skills = discover_skills(self.skills_path)
                skills_prompt = generate_skills_prompt(skills)
                if skills_prompt:
                    instructions.append(skills_prompt)
            except Exception:
                pass

        if self.output_tool_name:
            tool_choice = "required"
        elif self.tools:
            tool_choice = "auto"
        else:
            tool_choice = None

        return LlmRequest(
            instructions=instructions,
            contents=flat_contents,
            tools=self.tools,
            tool_choice=tool_choice,
        )

    def _is_final_response(self, event: Event) -> bool:
        if self.output_tool_name:
            for item in event.content:
                if (isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"):
                    return True
            return False
        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> Any:
        if self.output_tool_name:
            for item in event.content:
                if (isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success" and item.content):
                    return item.content[0]
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None

    def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        tools = list(tools)

        if self.output_type is not None:
            output_schema = self.output_type.model_json_schema()
            output_schema.pop("title", None)
            output_schema.pop("$defs", None)

            tool_definition = format_tool_definition(
                "final_answer",
                "Return the final structured answer matching the required schema.",
                {
                    "type": "object",
                    "properties": {"output": output_schema},
                    "required": ["output"],
                },
            )

            captured_type = self.output_type

            def _parse_output(output) -> str:
                if isinstance(output, dict):
                    return captured_type.model_validate(output)
                return output

            final_answer_tool = FunctionTool(
                func=_parse_output,
                name="final_answer",
                description="Return the final structured answer matching the required schema.",
                tool_definition=tool_definition,
            )
            tools.append(final_answer_tool)
            self.output_tool_name = "final_answer"

        # NEW: Collect sandbox-executable tools
        for t in tools:
            if isinstance(t, FunctionTool) and t.sandbox_executable:
                if self.code_execution != "e2b":
                    raise ValueError(
                        f"Tool '{t.name}' is marked as sandbox_executable "
                        "but code_execution is not enabled."
                    )
                self._sandbox_tools.append(t)

        # NEW: Add code execution tool
        if self.code_execution == "e2b":
            from scratch_agent.tools.code_execution import execute_python
            tools.append(execute_python)

        if self.memory_manager:
            from scratch_agent.tools.memory_tool import MemoryTool
            tools.append(MemoryTool(self.memory_manager))

        return tools

    # NEW: CH08 methods
    async def _setup_code_env(self, context: ExecutionContext):
        """Set up E2B sandbox environment."""
        try:
            from e2b_code_interpreter import Sandbox
            sandbox = Sandbox.create(timeout=300)
            context.code_env = sandbox
            self._register_sandbox_tools(sandbox)

            if self.skills_path:
                from scratch_agent.skills import discover_skills
                skills = discover_skills(self.skills_path)
                for skill_info in skills:
                    sandbox.files.write(
                        f"/home/user/skills/{skill_info.name}/{skill_info.path.name}",
                        skill_info.path.read_text(),
                    )
        except Exception as e:
            logger.warning(f"Failed to set up code execution environment: {e}")

    def _register_sandbox_tools(self, sandbox) -> None:
        """Register sandbox-executable tools in the sandbox."""
        tool_sources = []
        for t in self._sandbox_tools:
            source = t.get_source_code()
            tool_sources.append(source)
        combined_source = "\n\n".join(tool_sources)
        result = sandbox.run_code(combined_source)
        if result.error:
            raise RuntimeError(f"Failed to register sandbox tools: {result.error}")

    def _get_sandbox_tools_prompt(self) -> str:
        """Generate prompt describing sandbox-executable tools."""
        if not self._sandbox_tools:
            return ""
        tool_definitions = [t.tool_definition for t in self._sandbox_tools]
        tools_json = json.dumps(tool_definitions, indent=2, ensure_ascii=False)
        return (
            "\n\n## Sandbox-Executable Tools\n"
            "The following functions are pre-registered in the sandbox "
            "and can be called directly in your Python code:\n"
            f"{tools_json}"
        )

    async def _process_confirmations(
        self, context: ExecutionContext, confirmations: list[ToolConfirmation],
    ):
        raw_pending = context.state.pop("pending_tool_calls", [])
        pending = [PendingToolCall.model_validate(d) for d in raw_pending]
        tools_dict = {t.name: t for t in self.tools}
        results = []
        for pending_call in pending:
            tc = pending_call.tool_call
            confirmation = next(
                (c for c in confirmations if c.tool_call_id == tc.tool_call_id), None)
            if confirmation and confirmation.approved:
                args = confirmation.modified_arguments or tc.arguments
                tool_obj = tools_dict.get(tc.name)
                if tool_obj:
                    try:
                        output = await tool_obj(context, **args)
                        results.append(ToolResult(
                            tool_call_id=tc.tool_call_id, name=tc.name,
                            status="success", content=[output]))
                    except Exception as e:
                        results.append(ToolResult(
                            tool_call_id=tc.tool_call_id, name=tc.name,
                            status="error", content=[str(e)]))
            else:
                results.append(ToolResult(
                    tool_call_id=tc.tool_call_id, name=tc.name,
                    status="error", content=["User denied the tool execution."]))
        if results:
            tool_event = Event(
                execution_id=context.execution_id, author=self.name, content=results)
            context.add_event(tool_event)

    def _log_response(self, response: LlmResponse):
        for item in response.content:
            if isinstance(item, Message):
                logger.info(f"[{self.name}] {item.content}")
            elif isinstance(item, ToolCall):
                logger.info(f"[{self.name}] Tool call: {item.name}({item.arguments})")
