import pandas as pd
from tqdm import tqdm


# =====================================================
# Classification Step (Already added earlier)
# =====================================================

def run_classification(df: pd.DataFrame, classification_chain):
    results = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Classifying Calls"):
        try:
            output = classification_chain.invoke({
                "transcript": row["transcript"]
            })

            results.append({
                "call_id": row["call_id"],
                "predicted_call_type": output.call_type,
                "confidence": output.confidence
            })

        except Exception as e:
            print(f"❌ Error at row {i}: {e}")

            results.append({
                "call_id": row["call_id"],
                "predicted_call_type": None,
                "confidence": None
            })

    results_df = pd.DataFrame(results)
    df = df.merge(results_df, on="call_id")

    return df


# =====================================================
# Evaluation Step
# =====================================================

def run_evaluations(transcript, eval_plan, tone_chain, knowledge_chain, resolution_chain):
    """
    Runs required evaluators for a single transcript.
    """

    results = {}

    if "tone_empathy" in eval_plan:
        try:
            tone_result = tone_chain.invoke({"transcript": transcript})
            results["tone"] = tone_result.model_dump()
        except Exception as e:
            results["tone"] = {"error": str(e)}

    if "knowledge_accuracy" in eval_plan:
        try:
            knowledge_result = knowledge_chain.invoke({"transcript": transcript})
            results["knowledge"] = knowledge_result.model_dump()
        except Exception as e:
            results["knowledge"] = {"error": str(e)}

    if "resolution_quality" in eval_plan:
        try:
            resolution_result = resolution_chain.invoke({"transcript": transcript})
            results["resolution"] = resolution_result.model_dump()
        except Exception as e:
            results["resolution"] = {"error": str(e)}

    return results


def apply_evaluations(df, tone_chain, knowledge_chain, resolution_chain):
    """
    Applies evaluation logic to entire dataset.
    """

    evaluation_outputs = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Running Evaluations"):
        output = run_evaluations(
            transcript=row["transcript"],
            eval_plan=row["evaluation_plan"],
            tone_chain=tone_chain,
            knowledge_chain=knowledge_chain,
            resolution_chain=resolution_chain
        )

        evaluation_outputs.append({
            "call_id": row["call_id"],
            "evaluation_output": output
        })

    eval_df = pd.DataFrame(evaluation_outputs)

    df = df.merge(eval_df, on="call_id")

    return df


# =====================================================
# Final Reporting Step
# =====================================================

def generate_final_report(evaluation_output, final_chain):
    """
    Generates summary + recommendations for a single record.
    """

    try:
        result = final_chain.invoke({
            "evaluation_output": evaluation_output
        })

        return {
            "summary": result.summary,
            "recommendations": result.recommendations
        }

    except Exception as e:
        return {
            "summary": None,
            "recommendations": None,
            "error": str(e)
        }


def apply_final_reports(df, final_chain):
    """
    Applies final reporting to entire dataset.
    """

    final_outputs = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating Final Reports"):
        output = generate_final_report(
            evaluation_output=row["evaluation_output"],
            final_chain=final_chain
        )

        final_outputs.append({
            "call_id": row["call_id"],
            "summary": output.get("summary"),
            "recommendations": output.get("recommendations")
        })

    final_df = pd.DataFrame(final_outputs)

    df = df.merge(final_df, on="call_id")

    return df