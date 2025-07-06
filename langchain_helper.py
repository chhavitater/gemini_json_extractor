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

# OCR Stub for Image Input
def ocr_image(image_bytes):
    return "Read the image and extract relevant text using OCR logic or send image directly to Gemini for visual understanding."

# PDF Text Extractor
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") 
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
