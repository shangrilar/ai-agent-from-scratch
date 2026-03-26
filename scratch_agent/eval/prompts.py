"""Evaluation prompts and schemas for CH10."""

# Overall Pass Rate calculation
# OPR = (# all-pass) / (total evaluations)

ANSWER_RELEVANCY_PROMPT = """Evaluate whether the agent's answer is relevant to the question asked.

Question: {question}
Agent's Answer: {answer}

Rate the relevancy on a scale of 1-5:
1 - Completely irrelevant
2 - Slightly relevant
3 - Somewhat relevant
4 - Mostly relevant
5 - Highly relevant

Provide your rating and a brief explanation."""


CITATION_RELIABILITY_PROMPT = """Evaluate whether the agent's citations and sources are reliable.

Question: {question}
Agent's Answer: {answer}
Sources Used: {sources}

Check:
1. Are the sources real and accessible?
2. Do the sources support the claims made?
3. Are the sources authoritative for the topic?

Rate reliability on a scale of 1-5 and explain."""


REQUIREMENT_COMPLIANCE_PROMPT = """Evaluate whether the agent's answer complies with the requirements.

Original Question: {question}
Requirements: {requirements}
Agent's Answer: {answer}

Check if the answer:
1. Addresses all parts of the question
2. Follows the specified format
3. Meets any constraints mentioned

Rate compliance on a scale of 1-5 and explain."""


EVALUATION_SYSTEM_PROMPT = """You are an evaluation judge for AI agent responses.
Provide fair, consistent, and detailed evaluations.
Always explain your reasoning."""
