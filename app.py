# main.py

import streamlit as st
import io
import json
from PIL import Image
import langchain_helper as lch

st.set_page_config(page_title="Gemini PDF/Image → JSON", layout="wide")

st.title("📄 AI Document Extractor - PDF/Image to JSON using Gemini + LangChain")

uploaded_file = st.file_uploader("📤 Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    # Extract content
    if uploaded_file.type == "application/pdf":
        raw_text = lch.extract_text_from_pdf(uploaded_file)
    else:
        image = Image.open(uploaded_file)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        raw_text = lch.ocr_image(image_bytes.getvalue())  # You can replace this with Vision model later

    st.subheader("📃 Extracted Text:")
    st.text_area("Raw Content", raw_text, height=200)

    with st.spinner("🧠 Processing with Gemini AI..."):
        
        response = lch.run_gemini_prompt(raw_text).strip()

    try:
        json_data = json.loads(response)
        st.subheader("🧾 JSON Output")
        st.code(response, language="json")
        st.json(json_data)

        #st.subheader("📊 Table View")
        #st.table(json_data["data"])
    except Exception as e:
        st.error(f"❌ Failed to parse JSON: {e}")
        st.write("Raw output from Gemini:")
        st.write(response)
