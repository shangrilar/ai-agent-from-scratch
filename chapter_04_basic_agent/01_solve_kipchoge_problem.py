import asyncio
from dotenv import load_dotenv
load_dotenv()

from scratch_agents.tools import calculator, search_web, search_wikipedia, get_wikipedia_page
from scratch_agents.models.openai import OpenAILlm
from scratch_agents.agents.tool_calling_agent_ch4_base import ToolCallingAgent

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

async def main():
    tools = [search_web, calculator, search_wikipedia, get_wikipedia_page]
    model = OpenAILlm(model="gpt-5")
    agent = ToolCallingAgent(model=model, tools=tools, instruction=gaia_system_prompt, max_steps=20)
    result, context = await agent.run(kipchoge_problem, return_context=True)
    print(result)
    
if __name__ == "__main__":
    asyncio.run(main())
