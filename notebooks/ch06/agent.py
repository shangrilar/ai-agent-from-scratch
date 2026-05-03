"""CH06 snapshot: ReAct agent + session + memory + Human-in-the-loop
Changes from CH05:
  - __init__: session_manager, memory_manager, before_llm_callbacks added
  - run(): session load/save, confirmation handling, memory save logic
  - step(): before_llm_callbacks execution
  - act(): requires_confirmation check, pending handling
  - _setup_tools(): memory tool added
  - _process_confirmations() new method
  - code_execution, skills, sub_agents not yet added
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type

from pydantic import BaseModel

from scratch_agents.llm import LlmClient, LlmRequest, LlmResponse
from scratch_agents.types import Event, Message, ToolCall, ToolResult
from scratch_agents.tools.base import BaseTool, FunctionTool, tool
from scratch_agents.tools.helpers import format_tool_definition
from scratch_agents.context import (
    AgentResult,
    ExecutionContext,
    PendingToolCall,
    ToolConfirmation,
)

if TYPE_CHECKING:
    from scratch_agents.memory.long_term import TaskMemoryManager
    from scratch_agents.memory.session import BaseSessionManager

logger = logging.getLogger(__name__)


class Agent:
    """Tool-calling agent with ReAct loop, callbacks, sessions, and memory."""

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
        # NEW: CH06
        session_manager: Optional["BaseSessionManager"] = None,
        memory_manager: Optional["TaskMemoryManager"] = None,
        before_llm_callbacks: list[Callable] | None = None,
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

        # NEW: Load or create session
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

        # NEW: Handle tool confirmations (human-in-the-loop resume)
        if tool_confirmations:
            await self._process_confirmations(context, tool_confirmations)

        if user_input:
            user_event = Event(
                execution_id=context.execution_id,
                author="user",
                content=[Message(role="user", content=user_input)],
            )
            context.add_event(user_event)

        while not context.final_result and context.current_step < self.max_steps:
            result = await self.step(context, verbose=verbose)

            # NEW: Check for pending tool calls (human-in-the-loop)
            if result and result.status == "pending":
                if session and self.session_manager:
                    session.events = list(context.events)
                    await self.session_manager.save(session)
                return result

            if context.events:
                last_event = context.events[-1]
                if self._is_final_response(last_event):
                    context.final_result = self._extract_final_result(last_event)

        # NEW: Save memory
        if self.memory_manager:
            try:
                await self.memory_manager.save(context)
            except Exception as e:
                logger.warning(f"Failed to save memory: {e}")

        # NEW: Save session
        if session and self.session_manager:
            session.events = list(context.events)
            await self.session_manager.save(session)

        return AgentResult(output=context.final_result, context=context)

    async def step(
        self,
        context: ExecutionContext,
        verbose: bool = False,
    ) -> AgentResult | None:
        """Perform one think-act cycle."""
        llm_request = self._prepare_llm_request(context)

        # NEW: Run before-LLM callbacks
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

            # NEW: Check if tool requires confirmation
            if tool_obj.requires_confirmation:
                msg = tool_obj.get_confirmation_message(tool_call.arguments)
                pending.append(PendingToolCall(
                    tool_call=tool_call,
                    confirmation_message=msg,
                ))
                continue

            # before_tool callbacks (CH05)
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

            # after_tool callbacks (CH05)
            for cb in self.after_tool_callbacks:
                cb_result = cb(context, tool_result)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    tool_result = cb_result

            results.append(tool_result)

        # NEW: If there are pending confirmations, pause execution
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
        """Check if this event contains a final response."""
        if self.output_tool_name:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                ):
                    return True
            return False

        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> Any:
        """Extract the final result from an event."""
        if self.output_tool_name:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                    and item.content
                ):
                    return item.content[0]

        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None

    def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """Prepare the tools list, including dynamic tools."""
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

        # NEW: Add memory tool
        if self.memory_manager:
            from scratch_agents.tools.memory_tool import MemoryTool
            tools.append(MemoryTool(self.memory_manager))

        return tools

    # NEW: CH06
    async def _process_confirmations(
        self,
        context: ExecutionContext,
        confirmations: list[ToolConfirmation],
    ):
        """Process tool confirmations from human-in-the-loop."""
        raw_pending = context.state.pop("pending_tool_calls", [])
        pending = [PendingToolCall.model_validate(d) for d in raw_pending]

        tools_dict = {t.name: t for t in self.tools}
        results = []

        for pending_call in pending:
            tc = pending_call.tool_call
            confirmation = next(
                (c for c in confirmations if c.tool_call_id == tc.tool_call_id),
                None,
            )

            if confirmation and confirmation.approved:
                args = confirmation.modified_arguments or tc.arguments
                tool_obj = tools_dict.get(tc.name)
                if tool_obj:
                    try:
                        output = await tool_obj(context, **args)
                        results.append(ToolResult(
                            tool_call_id=tc.tool_call_id,
                            name=tc.name,
                            status="success",
                            content=[output],
                        ))
                    except Exception as e:
                        results.append(ToolResult(
                            tool_call_id=tc.tool_call_id,
                            name=tc.name,
                            status="error",
                            content=[str(e)],
                        ))
            else:
                results.append(ToolResult(
                    tool_call_id=tc.tool_call_id,
                    name=tc.name,
                    status="error",
                    content=["User denied the tool execution."],
                ))

        if results:
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=results,
            )
            context.add_event(tool_event)

    def _log_response(self, response: LlmResponse):
        """Log LLM response for verbose mode."""
        for item in response.content:
            if isinstance(item, Message):
                logger.info(f"[{self.name}] {item.content}")
            elif isinstance(item, ToolCall):
                logger.info(f"[{self.name}] Tool call: {item.name}({item.arguments})")
