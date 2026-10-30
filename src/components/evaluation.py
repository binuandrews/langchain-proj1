from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# =====================================================
# Tone Evaluation
# =====================================================

class ToneEvaluation(BaseModel):
    score: int = Field(description="Score between 1 and 5")
    reasoning: str = Field(description="Explanation of the score")


def get_tone_chain(llm):
    parser = PydanticOutputParser(pydantic_object=ToneEvaluation)

    prompt = PromptTemplate(
        template="""
You are a QA evaluator for customer support calls.

Evaluate the agent's tone and empathy in the following transcript.

Consider:
- Did the agent acknowledge the customer's issue?
- Was the tone polite and professional?
- Did the agent show empathy?

Transcript:
{transcript}

{format_instructions}
""",
        input_variables=["transcript"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    return prompt | llm | parser


# =====================================================
# Knowledge Evaluation
# =====================================================

class KnowledgeEvaluation(BaseModel):
    score: int = Field(description="Score between 1 and 5")
    reasoning: str = Field(description="Explanation of the score")


def get_knowledge_chain(llm):
    parser = PydanticOutputParser(pydantic_object=KnowledgeEvaluation)

    prompt = PromptTemplate(
        template="""
You are a QA evaluator for customer support calls.

Evaluate the agent's knowledge accuracy and clarity.

Consider:
- Did the agent provide correct and relevant information?
- Was the explanation clear and easy to understand?
- Did the agent avoid vague or misleading statements?

IMPORTANT:
- If the transcript does not contain enough information, give a moderate score (2 or 3) and explain why.

Transcript:
{transcript}

{format_instructions}
""",
        input_variables=["transcript"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    return prompt | llm | parser


# =====================================================
# Resolution Evaluation
# =====================================================

class ResolutionEvaluation(BaseModel):
    score: int = Field(description="Score between 1 and 5")
    reasoning: str = Field(description="Explanation of the score")


def get_resolution_chain(llm):
    parser = PydanticOutputParser(pydantic_object=ResolutionEvaluation)

    prompt = PromptTemplate(
        template="""
You are a QA evaluator for customer support calls.

Evaluate the resolution quality of the agent.

Consider:
- Did the agent fully resolve the customer's issue?
- Were next steps clearly communicated?
- Did the agent confirm resolution before ending?

Transcript:
{transcript}

{format_instructions}
""",
        input_variables=["transcript"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    return prompt | llm | parser