from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

client = AsyncOpenAI()

import asyncio
import time 

class PotatoSolution(BaseModel):
    thought_process: str
    final_answer: str
    
SYS_PROMPT = """You are a general AI assistant. 
I will ask you a question. Report your thoughts in "thought_process" and finish your answer in "final_answer".
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
"""

FAMILY_REUNION_PROBLEM = """
My family reunion is this week, and I was assigned the mashed potatoes to bring. The attendees include my married mother and father, my twin brother and his family, my aunt and her family, my grandma and her brother, her brother's daughter, and his daughter's family. All the adults but me have been married, and no one is divorced or remarried, but my grandpa and my grandma's sister-in-law passed away last year. All living spouses are attending. My brother has two children that are still kids, my aunt has one six-year-old, and my grandma's brother's daughter has three kids under 12. I figure each adult will eat about 1.5 potatoes of mashed potatoes and each kid will eat about 1/2 a potato of mashed potatoes, except my second cousins don't eat carbs. How many potatoes do I need in total? Just give the number.
"""

EXPECTED_ANSWER = "18"

async def get_llm_answer(
    client: AsyncOpenAI,
    model_name: str,
    prompt: str,
    result_format: type[BaseModel] | None = None
) -> tuple[str, str]:
    try:
        api_call_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt}
            ]
        }

        if model_name not in ["gpt-5", "gpt-5-mini"]:
            api_call_params["temperature"] = 0.5

        response = await client.beta.chat.completions.parse(
            **api_call_params,
            response_format=result_format
        )
        parsed_object = response.choices[0].message.parsed
        return parsed_object.thought_process, parsed_object.final_answer

    except Exception as e:
        print(f"Error during LLM API call for model {model_name}: {e}")
        return "", ""
    
    
async def run_problem_test(
    local_client: AsyncOpenAI,
    model_name: str,
    prompt_name: str,
    prompt_content: str,
    num_tests: int,
    expected_answer: str
) -> tuple[int, float]:
    """
    Asynchronously runs the math problem test N times for the specified prompt
    and returns the number of successful answers and total execution time.
    """
    print(f"\n--- Testing '{prompt_name}' prompt strategy ({num_tests} repetitions) ---")

    tasks = [
        get_llm_answer(local_client, model_name, prompt_content, result_format=PotatoSolution)
        for _ in range(num_tests)
    ]

    llm_responses = await asyncio.gather(*tasks)

    correct_answers = 0
    for i, (_, final_answer) in enumerate(llm_responses):
        if final_answer == expected_answer:
            correct_answers += 1

    print(f"'{prompt_name}' test completed: {correct_answers}/{num_tests} correct (Success rate: {correct_answers/num_tests*100:.2f}%)")
    return correct_answers

async def test_model_with_all_strategies(model_name: str, number_of_runs: int):
    """
    Test a single model with all prompt strategies in parallel.
    Returns results for the model.
    """
    print(f"\n======================================================================")
    print(f"Testing Model: {model_name}")
    print(f"======================================================================")
    
    # Define all prompts
    prompts = {
        "Baseline (Zero-shot)": FAMILY_REUNION_PROBLEM,
        
        "Few-shot": f"""
Here's an example of how to solve a similar family calculation problem: 
<example> 
Question: "I'm hosting a birthday party. Attendees include me, my parents, my sister and her husband, and my uncle with his two teenage children. Each adult will eat 2 slices of pizza and each child will eat 1 slice. How many pizza slices do I need?" 
Answer: 14 
</example> 
Now solve this problem:

{FAMILY_REUNION_PROBLEM}
""",
        
        "Role-based": f"""
You are a family event planning specialist with expertise in calculating food quantities for family gatherings. You excel at parsing complex family relationships and determining accurate serving quantities based on different demographics and dietary preferences. 
Using your expertise, please solve this problem:


{FAMILY_REUNION_PROBLEM}
""",
        
        "Chain-of-Thought (Guided)": f"""
{FAMILY_REUNION_PROBLEM}

Let's solve this step by step:
1. First, identify all family members attending:
- List each person and their relationship to you
- Account for spouses of married individuals
- Note any deceased family members who won't be attending
2. Categorize attendees by age group:
- Count total adults
- Count total children
- Note any special dietary restrictions
3. Apply consumption rules:
- Calculate potatoes needed for adults 
- Calculate potatoes needed for children 
- Adjust for any dietary restrictions 
4. Sum the total number of potatoes needed Please work through each step carefully. 

""",
        
        "Simple Chain-of-Thought": f"""
{FAMILY_REUNION_PROBLEM}

Think step by step and give the answer.
"""
    }
    
    # Run all strategies in parallel
    tasks = [
        run_problem_test(client, model_name, prompt_name, prompt_content, number_of_runs, EXPECTED_ANSWER)
        for prompt_name, prompt_content in prompts.items()
    ]
    
    results = await asyncio.gather(*tasks)
    return results

async def main():
    """
    Tests the family reunion problem using various prompt engineering techniques
    across multiple LLM models.
    """
    models_to_test = ["gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini"]
    number_of_runs = 10
    
    print(f"Starting family reunion problem test ({number_of_runs} runs per prompt per model)")
    print(f"Problem: Calculate how many bags of potatoes needed for family reunion")
    print(f"Expected answer: '{EXPECTED_ANSWER}'")
    
    overall_start = time.time()
    
    # Test each model sequentially (could also parallelize this)
    for model_name in models_to_test:
        await test_model_with_all_strategies(model_name, number_of_runs)
    
    overall_end = time.time()
    print(f"\n======================================================================")
    print(f"All tests completed in {overall_end - overall_start:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())