# explainer.py
import os
from groq import Groq
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_explanation_prompt(hex_id: str, features: dict, shap_values: dict, score: float) -> str:
    """
    Translate model features + SHAP values into a prompt.
    SHAP values tell the LLM *why* the model scored this hex highly.
    """
    top_drivers = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    driver_text = "\n".join(
        f"  - {feat}: SHAP={val:+.3f} ({'increases' if val > 0 else 'decreases'} likelihood)"
        for feat, val in top_drivers
    )

    return f"""You are an urban analytics assistant explaining why a location in Liverpool 
is likely or unlikely to attract a chain coffee shop.

Hex cell: {hex_id}
Chain likelihood score: {score:.2%}

Key features for this area:
  - Nearest café distance: {features.get('nearest_cafe_distance', 'N/A'):.0f}m
  - Nearby cafés (500m): {features.get('count_nearby_cafes', 'N/A')}
  - Chain ratio in hex: {features.get('hex_chain_ratio', 'N/A'):.0%}
  - Cafés in surrounding rings: {features.get('ring_cafe_count', 'N/A')}

Top model drivers (SHAP):
{driver_text}

In 2-3 sentences, explain in plain English why this area scores {score:.0%} likelihood 
for a chain coffee shop opening. Be specific to Liverpool where relevant. 
Do not mention SHAP or technical model details."""


@observe()  # Langfuse traces this entire function automatically
def explain_hex(hex_id: str, features: dict, shap_values: dict, score: float) -> str:
    prompt = build_explanation_prompt(hex_id, features, shap_values, score)

    # Tag the trace in Langfuse so you can filter by hex in the dashboard
    langfuse_context.update_current_trace(
        name="hex-explanation",
        tags=["coffee-predictor", hex_id],
        metadata={"score": score, "hex_id": hex_id},
    )

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # free, fast, good
        messages=[
            {"role": "system", "content": "You are a concise urban analytics assistant."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=200,
        temperature=0.3,  # low temp = more factual, less hallucination
    )

    explanation = response.choices[0].message.content
    return explanation