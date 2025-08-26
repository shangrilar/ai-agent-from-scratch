# Listing 2.1
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Listing 2.2
response = client.chat.completions.create(
    model="gpt-5-mini", 
    messages=[ 
        {"role": "developer", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": [{ "type": "text", "text": "Who's there?" }]} 
    ] 
) 
print(response.choices[0].message.content)

# Listing 2.3
response = client.chat.completions.create(
    model="o4-mini", 
    messages=[ 
        {"role": "developer", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": "Who's there?"} 
    ] 
) 
print(response.choices[0].message.content)
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Reasoning tokens: {response.usage.completion_tokens_details.reasoning_tokens}")

# Listing 2.4
response = client.chat.completions.create( 
    model="gpt-4o-mini", 
    messages=[ 
        {"role": "developer", "content": "You are a helpful assistant."}, 
        {"role": "user", "content": "Who's there?"} 
    ], 
    stream=True, 
    temperature=0.1, 
    max_completion_tokens=200, 
    logprobs=True 
)
for chunk in response: 
    print(chunk.choices[0].delta.content, end="", flush=True)

# Listing 2.5
response = client.responses.create( 
    model="gpt-5-mini", 
    input="Where is the capital of South Korea?", 
    instructions="You are a helpful assistant." 
) 
print(response.output_text)