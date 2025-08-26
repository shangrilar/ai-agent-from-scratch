from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Listing 2.6
response_1 = client.chat.completions.create( 
    model='gpt-5-mini', 
    messages=[{"role": "user", "content": "My name is Jungjun"}], 
) 
print(response_1.choices[0].message.content) 

response_2 = client.chat.completions.create( 
    model='gpt-5-mini', 
    messages=[{"role": "user", "content": "What is my name?"}], 
) 
print(response_2.choices[0].message.content) 

# Listing 2.7
messages = [] 
messages.append({"role": "user", "content": "My name is Jungjun"}) 

response_3 = client.chat.completions.create( 
    model='gpt-5-mini', 
    messages=messages, 
) 
print(response_3.choices[0].message.content) 

messages.append({"role": "assistant", "content": response_3.choices[0].message.content}) 
messages.append({"role": "user", "content": "What is my name?"}) 

response_4 = client.chat.completions.create( 
    model='gpt-5-mini', 
    messages=messages, 
) 
print(response_4.choices[0].message.content)

# Listing 2.8
response = client.responses.create( 
    model="gpt-5-mini",
    input="My name is Jungjun",
)
print(response.output_text)

second_response = client.responses.create(
    model="gpt-5-mini",
    previous_response_id=response.id, 
    input=[{"role": "user", "content": "What is my name?"}],
)
print(second_response.output_text)

