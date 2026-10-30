from src.utils.config_loader import load_config
from src.utils.data_loader import load_transcripts
from src.utils.llm_loader import load_llm
from src.utils.helpers import save_dataframe

from src.components.classification import get_classification_chain
from src.components.router import apply_routing
from src.components.evaluation import (
    get_tone_chain,
    get_knowledge_chain,
    get_resolution_chain
)
from src.components.reporting import get_final_report_chain

from src.pipeline.pipeline import (
    run_classification,
    apply_evaluations,
    apply_final_reports
)

def main():
    # Load config + data
    config = load_config()
    df = load_transcripts()

    # Initialize LLM
    llm = load_llm(config)

    # Create chains
    classification_chain = get_classification_chain(llm, config)
    tone_chain = get_tone_chain(llm)
    knowledge_chain = get_knowledge_chain(llm)
    resolution_chain = get_resolution_chain(llm)
    final_chain = get_final_report_chain(llm)

    # Pipeline execution
    df = run_classification(df, classification_chain)
    df = apply_routing(df)
    df = apply_evaluations(df, tone_chain, knowledge_chain, resolution_chain)
    df = apply_final_reports(df, final_chain)

    # Save output
    save_dataframe(df)

    print("✅ Pipeline executed successfully!")


if __name__ == "__main__":
    main()