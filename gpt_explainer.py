from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def explain_result(prediction, confidence):
    prompt = f"""
A CT scan was analyzed for internal bleeding.

Prediction: {prediction}
Confidence: {confidence:.2f}%

Explain this result in simple, non-diagnostic clinical language.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


