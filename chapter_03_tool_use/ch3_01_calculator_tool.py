import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Listing 3.1
calculator_tool_definition = { 
    "type": "function",
    "function": {
        "name": "calculator", 
        "description": "Perform basic arithmetic operations between two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {
                    "type": "string",
                    "description": "Arithmetic operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "first_number": {
                    "type": "number",
                    "description": "First number for the calculation"
                },
                "second_number": {
                    "type": "number",
                    "description": "Second number for the calculation"
                }
            },
            "required": ["operator", "first_number", "second_number"],
        }
    }
}

# Listing 3.2
def calculator(operator: str, first_number: float, second_number: float) -> float:
   if operator == 'add':
       return first_number + second_number
   elif operator == 'subtract':
       return first_number - second_number
   elif operator == 'multiply':
       return first_number * second_number
   elif operator == 'divide':
       if second_number == 0:
           raise ValueError("Cannot divide by zero")
       return first_number / second_number
   else:
       raise ValueError(f"Unsupported operator: {operator}")

if __name__ == "__main__":
    # Listing 3.3
    tools = [calculator_tool_definition]

    response_without_tool = client.chat.completions.create(
            model='gpt-5-mini',
            messages=[{"role": "user", "content": "What is the capital of South Korea?"}],
            tools=tools
    )
    print(response_without_tool.choices[0].message.content) # The capital of South Korea is Seoul.
    print(response_without_tool.choices[0].message.tool_calls) # None

    response_with_tool = client.chat.completions.create(
            model='gpt-5-mini',
            messages=[{"role": "user", "content": "What is 1234 x 5678?"}],
            tools=tools
    )
    print(response_with_tool.choices[0].message.content) # None
    print(response_with_tool.choices[0].message.tool_calls) 
    # [ChatCompletionMessageFunctionToolCall(id='call_viaOEiQJ5VEB9YvKl95qlDjM', function=Function(arguments='{"operator":"multiply","first_number":1234,"second_number":5678}', name='calculator'), type='function')]

    # Listing 3.4
    ai_message = response_with_tool.choices[0].message

    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "calculator":
                result = calculator(**function_args)
                print("calculator result:", result)
                
    # Listing 3.5
    messages = []
    messages.append({"role": "user", "content": "What is 1234 x 5678?"})

    response_with_tool = client.chat.completions.create(
    model='gpt-5-mini',
    messages=messages,
    tools=tools
    )

    ai_message = response_with_tool.choices[0].message

    messages.append({  
    "role": "assistant",  
    "content": ai_message.content,  
    "tool_calls": ai_message.tool_calls  
    })

    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            function_name = tool_call.function.name  
            function_args = json.loads(tool_call.function.arguments)  
            
            if function_name == "calculator":
                result = calculator(**function_args)  
                
                messages.append({  
                    "role": "tool",  
                    "tool_call_id": tool_call.id,  
                    "content": str(result)  
                })

    final_response = client.chat.completions.create(
        model='gpt-5-mini',
        messages=messages
    )
    print("Messages:", messages)
    print("Final Answer:", final_response.choices[0].message.content)
    
