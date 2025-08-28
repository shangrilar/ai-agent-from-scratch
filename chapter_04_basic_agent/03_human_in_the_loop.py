import asyncio
from scratch_agents.tools import search_web, calculator, search_wikipedia, get_wikipedia_page
from scratch_agents.models.openai import OpenAILlm
from scratch_agents.agents.tool_calling_agent_ch4_callback import ToolCallingAgent
from dotenv import load_dotenv

load_dotenv()

gaia_system_prompt = """
You are a general AI assistant. 
I will ask you a question. 
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
"""

kipchoge_problem = """
If Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon at its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest integer.
"""

def basic_approval_callback(context, tool_call):
    tool_name = tool_call.name
    print(f"\n🔧 Tool Execution Request")
    print(f"Tool: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    
    response = input("Execute this tool? (y/n): ").lower().strip()
    
    if response == 'y':
        print("✅ Approved. Executing...\n")
        return None
    else:
        print("❌ Denied. Skipping execution.\n")
        return f"User denied execution of {tool_name}"
    

def session_aware_approval_callback(context, tool_call):
    tool_name = tool_call.name
    # Check if tool is already marked as safe in this session
    safe_tools = context.state.get('safe_tools', [])
    if tool_name in safe_tools:
        print(f"✓ Auto-executing {tool_name} (marked as safe)")
        return None
    
    response = input("Execute this tool? (y to run once, ya to allow for session, n to skip): ").lower().strip()
    
    if response == 'y':
        print("✅ Approved. Executing...\n")
        return None
    elif response == 'ya':
        if 'safe_tools' not in context.state:
            context.state['safe_tools'] = []
        context.state['safe_tools'].append(tool_name)
        print(f"✅ {tool_name} marked as safe for this session. Executing...\n")
        return None
    else:
        print("❌ Denied. Skipping execution.\n")
        return f"User denied execution of {tool_name}"


async def main():
    tools = [search_web, calculator, search_wikipedia, get_wikipedia_page]
    model = OpenAILlm(model="gpt-5-mini")
    agent = ToolCallingAgent(
        name="callback_agent",
        model=model, 
        tools=tools, 
        instructions=gaia_system_prompt,
        before_tool_callbacks=[basic_approval_callback]
    )
    result = await agent.run(kipchoge_problem)
    print(result)
    
if __name__ == "__main__":
    asyncio.run(main())
