from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# =====================================================
# Final Report Schema
# =====================================================

class FinalReport(BaseModel):
    summary: str = Field(description="Overall evaluation summary")
    recommendations: list[str] = Field(description="List of actionable improvements")


# =====================================================
# Final Report Chain
# =====================================================

def get_final_report_chain(llm):
    """
    Creates final QA reporting chain.
    """

    parser = PydanticOutputParser(pydantic_object=FinalReport)

    prompt = PromptTemplate(
        template="""
You are a QA manager reviewing customer support calls.

Based on the evaluation results below, generate:

1. A concise summary of the agent's performance
2. A list of actionable recommendations for improvement

Evaluation Data:
{evaluation_output}

IMPORTANT:
- Be specific and practical
- Do not repeat scores
- Focus on improvement

{format_instructions}
""",
        input_variables=["evaluation_output"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    return prompt | llm | parser