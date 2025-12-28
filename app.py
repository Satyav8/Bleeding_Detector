import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from vit_model import load_model
from heatmap import generate_heatmap
from gpt_explainer import explain_result

st.set_page_config(page_title="Internal Bleeding Detection", layout="centered")
st.title("🩸 Internal Bleeding Detection (CT Scan)")

model = load_model()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

uploaded = st.file_uploader("Upload CT Scan Image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded CT Scan", width=300)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()

    label = "Bleeding Detected" if pred == 1 else "No Bleeding Detected"
    confidence = probs[pred].item() * 100

    st.subheader(f"🧠 Prediction: {label}")
    st.write(f"Confidence: **{confidence:.2f}%**")

    heatmap = generate_heatmap(img_tensor)

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(heatmap, cmap="jet", alpha=0.5)
    ax.axis("off")
    st.pyplot(fig)

    with st.spinner("Generating explanation..."):
        explanation = explain_result(label, confidence)

    st.subheader("🧾 AI Explanation")
    st.write(explanation)
