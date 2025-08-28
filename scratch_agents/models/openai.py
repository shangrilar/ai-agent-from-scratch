from openai import AsyncOpenAI
from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse
from ..types.contents import Message, ToolCall, ToolResult
import json
from pydantic import Field

class OpenAILlm(BaseLlm):
    """OpenAI LLM implementation"""
    
    llm_config: dict = Field(default_factory=dict)
    
    def __init__(self, model, **kwargs):
        super().__init__(model=model)
        self.llm_config = kwargs
        self._client = None
    
    @property
    def openai_client(self):
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client
    
    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Generate a response using OpenAI API"""
        try:
            # Build messages for OpenAI API
            messages, model_params = self._build_llm_input(request, self.llm_config)
            
            # Convert tools_dict to tools array for OpenAI
            tools = None
            if request.tools_dict:
                tools = [tool.tool_definition for tool in request.tools_dict.values()]
            # Call OpenAI API
            call_kwargs = {}
            if request.tool_choice is not None:
                call_kwargs["tool_choice"] = request.tool_choice
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                **call_kwargs,
                **model_params
            )
           
            # Extract message from response
            choice = response.choices[0]
            content_items = []
           
            # Handle text content
            if choice.message.content:
               content_items.append(Message(role="assistant", content=choice.message.content))
           
            # Handle tool calls
            if choice.message.tool_calls:
               for tool_call in choice.message.tool_calls:
                   content_items.append(ToolCall(
                       tool_call_id=tool_call.id,
                       name=tool_call.function.name,
                       arguments=json.loads(tool_call.function.arguments)
                   ))
           
            # Extract usage metadata
            usage_metadata = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
           
            return LlmResponse(
                content=content_items,
                usage_metadata=usage_metadata
            )
        except Exception as e:
            return LlmResponse(
                error_message=str(e)
            )
    
    def _build_llm_input(self, request: LlmRequest, model_config: dict):
        """Build messages and parameters for OpenAI API"""
        messages = []
        
        # Add instructions as system messages
        for instruction in request.instructions:
            messages.append({"role": "system", "content": instruction})
        
        # Add conversation history
        # Group assistant messages and their tool calls together
        pending_assistant_content = None
        pending_tool_calls = []
        
        def flush_assistant_message():
            """Flush any pending assistant message with its tool calls"""
            if pending_assistant_content is not None or pending_tool_calls:
                msg = {"role": "assistant"}
                if pending_assistant_content is not None:
                    msg["content"] = pending_assistant_content
                else:
                    msg["content"] = None
                if pending_tool_calls:
                    msg["tool_calls"] = pending_tool_calls
                messages.append(msg)
                return True
            return False
        
        for item in request.contents:
            if isinstance(item, Message):
                if item.role == "assistant":
                    # Accumulate assistant content
                    pending_assistant_content = item.content
                else:
                    # Non-assistant message, flush any pending assistant message
                    flush_assistant_message()
                    pending_assistant_content = None
                    pending_tool_calls = []
                    messages.append({"role": item.role, "content": item.content})
                    
            elif isinstance(item, ToolCall):
                # Accumulate tool calls with the assistant message
                pending_tool_calls.append({
                    "id": item.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": json.dumps(item.arguments)
                    }
                })
                
            elif isinstance(item, ToolResult):
                # Tool result means we need to flush any pending assistant message
                flush_assistant_message()
                pending_assistant_content = None
                pending_tool_calls = []
                    
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.tool_call_id,
                    "content": str(item.content[0]) if item.content else ""
                })
        
        # Flush any remaining assistant message
        flush_assistant_message()
        
        # Extract model parameters
        model_params = {**self.llm_config}
        
        return messages, model_params