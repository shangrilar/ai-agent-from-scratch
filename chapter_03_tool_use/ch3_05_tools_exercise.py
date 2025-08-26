import json
from openai import OpenAI
from dotenv import load_dotenv
from ch3_01_calculator_tool import calculator
from ch3_02_tavily_search_tool import search_web
from ch3_03_wikipedia_tool import search_wikipedia, get_wikipedia_page
from ch3_04_tool_definition import function_to_tool_definition

load_dotenv()

client = OpenAI()

system_prompt = "You are a helpful assistant. calculator has only 4 operations: add, subtract, multiply, divide"

tools = [calculator, search_web, search_wikipedia, get_wikipedia_page]
tool_box = {tool.__name__: tool for tool in tools}
tool_definitions = [function_to_tool_definition(tool) for tool in tools]

# Listing 3.20
def tool_execution(tool_box, tool_call):
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    tool_result = tool_box[function_name](**function_args)
    return tool_result

# Listing 3.21
def run_step(system_prompt, question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            tools=tool_definitions
        )
        
        assistant_message = response.choices[0].message
        
        if assistant_message.tool_calls:
            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                tool_result = tool_execution(tool_box, tool_call)
                messages.append({
                    "role": "tool", 
                    "content": str(tool_result), 
                    "tool_call_id": tool_call.id
                })
        else:
            return assistant_message.content

# Listing 3.22
def step_1_search_kipchoge():
    question = """I need to find Eliud Kipchoge's record-making marathon pace. 
    Please search for information about his world record marathon time and 
    calculate his pace per kilometer.

    FINAL ANSWER should be in the format: "X.XX minutes per km"."""
    
    result = run_step(system_prompt, question)
    return result

kipchoge_result = step_1_search_kipchoge()
print(f"Step 1 Complete - Kipchoge pace: {kipchoge_result}")

# Listing 3.23
def step_2_search_moon_distance():
    question = """I need to find the minimum perigee value (closest approach 
    distance) between Earth and Moon from the Wikipedia page for the Moon. 
    Please search for this information.

    FINAL ANSWER should be in the format: "X km"."""
    
    result = run_step(system_prompt, question)
    return result

moon_result = step_2_search_moon_distance()
print(f"Step 2 Complete - Moon distance: {moon_result}")

# Listing 3.24
def step_3_calculate(kipchoge_pace, moon_distance):
    question = f"""Given the following information:
- Kipchoge's pace: {kipchoge_pace}
- Moon distance: {moon_distance}

    Please calculate how many hours it would take Kipchoge to run this distance 
    at his record pace. Make sure to handle unit conversions properly.

    FINAL ANSWER should be in the format: "X hours"."""
    
    result = run_step(system_prompt, question)
    return result

time_result = step_3_calculate(kipchoge_result, moon_result)
print(f"Step 3 Complete - Time needed: {time_result}")

# Listing 3.25
def step_4_final_answer(total_hours):
    question = f"""Given that the total time is {total_hours}, I need to round 
    this to the nearest 1000 hours and express the answer in thousand hours.

    The original question asks for the result rounded to the nearest 1000 hours.

    FINAL ANSWER should be just the number (in thousand hours)."""
    
    result = run_step(system_prompt, question)
    return result

final_result = step_4_final_answer(time_result)
print(f"Step 4 Complete - Final answer: {final_result}")
