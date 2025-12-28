import os
from openai import OpenAI

# Render uses environment variables, not Streamlit secrets
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def explain_result(prediction, confidence):
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OpenAI API key not configured."

    prompt = f"""
A CT scan was analyzed for internal bleeding.

Prediction: {prediction}
Confidence: {confidence:.2f}%

Explain this result in simple, non-diagnostic clinical language.
Do not provide medical advice.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content



