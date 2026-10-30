# -----------------------------
# Routing Logic
# -----------------------------

def route_call(call_type: str):
    """
    Determines evaluation plan based on call type.
    """

    if call_type == "billing":
        return ["knowledge_accuracy", "resolution_quality"]

    elif call_type == "claims":
        return ["knowledge_accuracy", "resolution_quality"]

    elif call_type == "complaint":
        return ["tone_empathy", "resolution_quality"]

    elif call_type == "general_query":
        return ["knowledge_accuracy"]

    else:
        return ["knowledge_accuracy"]  # fallback


# -----------------------------
# Apply Routing to DataFrame
# -----------------------------

def apply_routing(df):
    """
    Adds evaluation_plan column to DataFrame.
    """
    df["evaluation_plan"] = df["predicted_call_type"].apply(route_call)
    return df