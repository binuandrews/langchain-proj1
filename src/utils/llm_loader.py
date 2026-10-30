import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


def load_llm(config: dict):
    """
    Initialize LLM based on provider from environment.
    """
    provider = os.getenv("LLM_PROVIDER")

    if provider is None:
        raise ValueError("LLM_PROVIDER not set in environment")

    provider = provider.lower()

    if provider == "openai":
        return ChatOpenAI(
            model=config["llm"]["openai_model"],
            temperature=config["llm"]["temperature"]
        )

    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config["llm"]["gemini_model"],
            temperature=config["llm"]["temperature"]
        )

    else:
        raise ValueError(f"Invalid LLM_PROVIDER: {provider}")