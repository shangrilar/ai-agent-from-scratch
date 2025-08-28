from typing import List, Dict, Any, Optional
from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..types.contents import Message, ToolCall
from ..types.events import Event
from .execution_context_ch4 import ExecutionContext
from ..tools.base_tool import BaseTool
from ..types.contents import ToolResult
from typing import Type
from pydantic import BaseModel
from ..tools.decorator import tool

class ToolCallingAgent:
    def __init__(self, name: str, model: BaseLlm, tools: List[BaseTool], instructions: str, max_steps: int = 10, output_type: Optional[Type[BaseModel]] = None):
        self.name = name
        self.model = model
        self.max_steps = max_steps
        self.instructions = instructions
        self.output_type: Optional[Type[BaseModel]] = output_type
        self.output_tool: Optional[str] = None
        self.tools = self._setup_tools(tools)
        
    def _setup_tools(self, tools: List[BaseTool]):
        if self.output_type is not None:
            @tool(name="final_answer", description="Return the final structured answer matching the required schema.")
            def final_answer(output: self.output_type) -> self.output_type:
                return output
            tools.append(final_answer)
            self.output_tool = final_answer.name
        return {t.name: t for t in tools}
        
    async def think(self, context: ExecutionContext, llm_request: LlmRequest):
        llm_response = await self.model.generate(llm_request)
        return llm_response
    
    async def act(self, context: ExecutionContext, tool_calls: List[ToolCall]):
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_input = tool_call.arguments
            print(f"  → Calling {tool_name} with {tool_input}")
            try:
                result_output = await self.tools[tool_name](**tool_input)
                tool_results.append(
                    ToolResult(
                        tool_call_id=tool_call.tool_call_id,
                        name=tool_call.name,
                        status="success",
                        content=[result_output],
                    )
                )
            except Exception as e:
                tool_results.append(
                    ToolResult(
                        tool_call_id=tool_call.tool_call_id,
                        name=tool_call.name,
                        status="error",
                        content=[str(e)],
                    )
                )
        return tool_results
    
    async def step(self, context: ExecutionContext):
        print(f"[Step {context.current_step + 1}]")
        llm_request = self._prepare_llm_request(context)
        llm_response = await self.think(context, llm_request)
        if llm_response.error_message:
            raise RuntimeError(f"LLM error: {llm_response.error_message}")
        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            required_output_tool=self.output_tool or None,
            **llm_response.model_dump(),
        )
        context.add_event(response_event)
        
        if tool_calls := response_event.get_tool_calls():
            tool_results = await self.act(context, tool_calls)
            tool_results_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                required_output_tool=self.output_tool or None,
                content=tool_results,
            )
            context.add_event(tool_results_event)
            
        context.increment_step()
        
    async def run(self, user_input: str):
        context = ExecutionContext(
            user_input=user_input,
        )
        user_input_event = Event(
            execution_id=context.execution_id,
            author="user",
            content=[
                Message(
                    role="user",
                    content=user_input,
                )
            ],
        )
        context.add_event(user_input_event)
        
        while not context.final_result and context.current_step < self.max_steps:
            await self.step(context)
            
            last_event = context.events[-1]
            if last_event.is_final_response():
                context.final_result = self._extract_final_result(last_event)
                
        return context.final_result
            
    def _prepare_llm_request(self, context: ExecutionContext):
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)
            
        if self.output_tool:
            tool_choice = "required"
        elif self.tools:
            tool_choice = "auto"
        else:
            tool_choice = None
            
        return LlmRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools_dict=self.tools,
            tool_choice=tool_choice
        )
    
    def _extract_final_result(self, event: Event):
        if event.required_output_tool:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.status == "success"
                    and item.name == event.required_output_tool
                    and item.content
                ):
                    return item.content[0]
        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content