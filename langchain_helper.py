# langchain_helper.py
import os
import json
import fitz  # PyMuPDF
from PIL import Image
import io
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import tool

# Load env vars
load_dotenv()
# Gemini LLM Setup
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.3
)

def get_prompt(raw_text: str) -> str:
    return f"""
You are an AI document parser.

From the following financial document text, extract structured data and return valid JSON only.

Try to identify what type of document it is (e.g., bank statement, invoice) and extract appropriate fields based on that.

---

📄 If it is an **Invoice**, include fields like:
- Invoice number
- Invoice date
- Seller and Buyer info (name, address, GSTIN)
- Items (description, quantity, unit, rate, amount, HSN, packing)
- Taxes (CGST, SGST, IGST, taxable value)
- Total amount

---

🏦 If it is a **Bank Statement**, extract:
- Account holder name
- Account number
- Statement period
- Opening balance
- Closing balance
- Total deposits
- Total withdrawals
- A list of transactions with:
  - date
  - time (if present)
  - description
  - amount (positive or negative)
  - balance after transaction
  - transaction type (e.g. deposit, withdrawal, UPI, interest, etc.)
  - reference number or transaction ID (if present)

---

🧾 Output Rules:
- Always return a valid JSON object
- Do not include any explanations or summaries
- Use empty strings or null for missing fields
- Infer structure even if the formatting is inconsistent

---

📃 Document Text:
\"\"\"
{raw_text}
\"\"\"
"""


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
    return str(response)

