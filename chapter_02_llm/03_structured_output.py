from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# Listing 2.10
class User(BaseModel):
    name: str
    email: str

response = client.beta.chat.completions.parse(
    model='gpt-5-mini',
    messages=[{"role": "user", "content": """My name is John Smith, 
my phone number is (555) 123-4567, 
and my email is john.smith@example.com"""}],
    response_format=User,
)

print(type(response.choices[0].message.parsed))
print(response.choices[0].message.parsed)
