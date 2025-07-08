# main.py

import streamlit as st
import io
import json
from PIL import Image
import langchain_helper as lch
from langchain_helper import extract_json_from_response
from langchain_helper import enforce_all_keys



st.set_page_config(page_title="Gemini PDF/Image → JSON", layout="wide")

st.title("📄 AI Document Extractor - PDF / Image / Excel to JSON using Gemini + LangChain")

uploaded_file = st.file_uploader("📤 Upload PDF, Image, or Excel", type=["pdf", "png", "jpg", "jpeg", "xls", "xlsx"])

if uploaded_file:
    # Extract content
    if uploaded_file.name.lower().endswith(".pdf"):
        raw_text = lch.extract_text_from_pdf(uploaded_file)

    elif uploaded_file.name.lower().endswith((".xls", ".xlsx")):
        raw_text = lch.extract_text_from_excel(uploaded_file)

    else:
        image = Image.open(uploaded_file)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        raw_text = lch.ocr_image(image_bytes.getvalue())  # Stub or send to Vision API

    st.subheader("📃 Extracted Text:")
    st.text_area("Raw Content", raw_text, height=200)

    with st.spinner("🧠 Processing with Gemini AI..."):
        llm_response = None  # Predefine to avoid unbound error

        try:
            llm_response = lch.run_gemini_prompt(raw_text)
            json_data = extract_json_from_response(llm_response)
            json_data = enforce_all_keys(json_data)


            st.subheader("🧾 JSON Output")
            st.code(json.dumps(json_data, indent=2), language="json")
            #st.json(json_data)

            st.subheader("📊 Table View")
            st.table(json_data["data"])

        except Exception as e:
            st.error(f"❌ Failed to parse JSON: {e}")
            st.write("Raw output from Gemini:")
            st.write(llm_response if llm_response else "No response received.")

