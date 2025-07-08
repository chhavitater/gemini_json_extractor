# langchain_helper.py
import os
import json
import fitz  # PyMuPDF
from PIL import Image
import io
from dotenv import load_dotenv
import pandas as pd
import re
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import tool

# Load env vars
load_dotenv()
# Gemini LLM Setup
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.3
)

def get_prompt(raw_text: str) -> str:
    return f"""
You are a document parsing AI.

You must return **only valid JSON**. Do not include any explanations, markdown, or commentary.

🧾 Rules:
1. Each group of lines (separated by a blank line) is one record.
2. Each line within a group is in the format `Field Name: Value`.
3. Use field names exactly as shown — preserve punctuation, spacing, and capitalization.
4. If a value is missing, blank, or "null", include it **explicitly** with a value of `null` in the JSON.
5. If a value is numeric (integer or float), return it as a number (not a string).
6. If a value is the number `0`, keep it as `0`.
7. The final output must be a **list of JSON objects**, each with the **same set of keys**, including those with null values.
8. Do NOT include any explanations, thoughts, headings, or text before or after the JSON.

Format:
[
  {{
    "Field1": "value",
    "Field2": 123,
    "Field3": null
  }},
  ...
]

---

📃 Raw Input Text:
\"\"\"
{raw_text}
\"\"\"
"""

import re  # Add this at top if not already



# OCR Stub for Image Input
def ocr_image(image_bytes):
    return "Read the image and extract relevant text using OCR logic or send image directly to Gemini for visual understanding."

# PDF Text Extractor
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()  # type: ignore
    return text

def extract_text_from_excel(file) -> str:
    try:
        xls = pd.ExcelFile(file)
        text = ""
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name, header=1)  # 👈 Use second row as header (index 1)

            # Drop unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)].copy()
            df.dropna(axis=1, how='all', inplace=True)
            df.fillna("null", inplace=True)

            text += f"--- Sheet: {sheet_name} ---\n"
            for _, row in df.iterrows():
                for key, value in row.items():
                    text += f"{key}: {value}\n"
                text += "\n"
        return text.strip()
    except Exception as e:
        return f"Failed to read Excel file: {e}"


# LangChain Tool to Convert to JSON
@tool
def convert_to_json_tool(raw_text: str) -> str:
    """Use this tool to convert raw text from image or PDF into structured JSON format."""
    prompt = f"""Extract structured JSON data from this input:

### Input Content ###
{raw_text}

### Expected Output Format ###
Respond ONLY with a valid JSON object in the following format:

{{
    "data": [
        {{
            "field1": "value1",
            "field2": "value2"
        }}
    ]
}}
"""

    response = llm.invoke(prompt)
    return str(response.content)

# LangChain Agent Initialization
def get_agent():
    tools = [convert_to_json_tool]
    agent = initialize_agent(
        tools, # type: ignore
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent

# ✅ Clean direct call to Gemini with flexible prompt
def run_gemini_prompt(raw_text: str) -> str:
    prompt = get_prompt(raw_text)
    response = llm.invoke(prompt)
    
    # Handle cases where response is a string, list, or BaseMessage
    if isinstance(response, str):
        return response  # If response is already a string, return it directly.
    
    # If response is a BaseMessage, get the message content
    if hasattr(response, 'content'):
        return str(response.content)
    
    # If response is a list, join it into a single string
    if isinstance(response, list):
        return ' '.join([str(item) for item in response])
    
    # Fallback case if none of the above (should not reach here normally)
    print("Unexpected response type:", response)
    return str(response)

import re  # Make sure this is at the top of the file

def extract_json_from_response(llm_response: str) -> dict:
    """Cleans Gemini output and safely extracts JSON."""
    content = llm_response.strip()

    # Remove any <think> ... </think> tags
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Try to find the JSON array start/end
    json_start = content.find('[')
    json_end = content.rfind(']') + 1

    if json_start == -1 or json_end == -1:
        raise ValueError("❌ Could not find valid JSON array boundaries.")

    json_str = content[json_start:json_end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ JSON decode failed: {e}")
    
def enforce_all_keys(json_data):
    """Ensure all dicts have the same keys by filling missing ones with null."""
    all_keys = set()
    for item in json_data:
        all_keys.update(item.keys())

    for item in json_data:
        for key in all_keys:
            item.setdefault(key, None)
    return json_data

