import asyncio
from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from scratch_agents.agents.tool_calling_agent_ch4_structured_output import ToolCallingAgent
from scratch_agents.models.openai import OpenAILlm

load_dotenv()


async def main():
    # Initialize LLM (ensure OPENAI_API_KEY is set in your environment)
    llm = OpenAILlm(model="gpt-5-mini")

    class SentimentAnalysis(BaseModel):
        sentiment: Literal["positive", "negative", "neutral"]
        confidence: float
        key_phrases: List[str]
        
    agent = ToolCallingAgent(
        name="sentiment_analyzer",
        model=llm,
        tools=[],  # Could include tools for data retrieval
        instructions="Analyze the sentiment of the provided text.",
        output_type=SentimentAnalysis
    )

    result = await agent.run("This product exceeded my expectations! Highly recommend.")
    # result is now a SentimentAnalysis instance with validated fields
    print(f"Sentiment: {result.sentiment} (confidence: {result.confidence})")

if __name__ == "__main__":
    asyncio.run(main())
