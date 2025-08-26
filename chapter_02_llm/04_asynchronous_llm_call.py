import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI()

# Listing 2.11
async def get_answer(prompt):  #A
    response = await client.chat.completions.create(  #B
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

async def main(): 
    prompts = [
        "Hello!",
        "What's 2 + 2?",
        "Tell me a short joke about cats."
    ]
    
    tasks = [get_answer(p) for p in prompts] 
    
    results = await asyncio.gather(*tasks)
    
    for r in results:  
        print(r)

asyncio.run(main())
