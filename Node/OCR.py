import json

from langgraph.constants import Send
from langchain_core.messages import HumanMessage , SystemMessage

import fitz  # PyMuPDF
import base64
import json
from datetime import datetime
from time import perf_counter
from Schemes.schema import OverallState, PageState ,OCRExamResponse
from config import get_gemini_model , get_ollama_model , get_typhoon_model
from utils.log_collecting import log_token_usage

def llm_select(platform_name: str):
    
    if platform_name== "gemini":
        return get_gemini_model()
    elif platform_name == "ollama":
        return get_ollama_model()
    elif platform_name == "typhoon":
        return get_typhoon_model(model = "typhoon-ocr")

    else:
        raise ValueError(f"Unsupported LLM name: {platform_name}")

# Node: ใช้ PyMuPDF อ่านไฟล์ PDF และแปลงแต่ละหน้าเป็น Base64
def read_and_split_pdf(state: OverallState):
    doc = fitz.open(state["pdf_path"])
    
    pages_list = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = 1.25
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_bytes = pix.tobytes("png")
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        pages_list.append(b64_img)
        
    return {"pages": pages_list }

# Conditional Edge (Fan-out): บอก LangGraph ให้แตก Node การทำงานแบบ Parallel ตามจำนวนหน้า
def continue_to_ocr(state: OverallState):
    return [
        Send("process_ocr_page", {"page_b64": page, 
                                  'progress': [i, len(state["pages"])], 
                                  'llm_OCR_platform': state["llm_OCR_platform"]}) 
                                  
                                  for i, page in enumerate(state["pages"])
    ]

# Node: ประมวลผลแต่ละหน้าแบบ Parallel โดยรับ State แบบเดี่ยว (PageState)
def process_ocr_page(state: PageState):
    """
    Process individual PDF pages in parallel using OCR.
    
    This node processes a single page asynchronously and can run concurrently 
    for multiple pages. It receives base64-encoded page data and page number,
    sends them to an LLM model configured to output JSON format only.
    
    Args:
        state (PageState): Contains page_b64 (base64 image) and page_num (page number)
    
    Returns:
        dict: Contains ocr_results with the OCR processing output for the page
    """
    start_date = datetime.now()
    strat_ocr_page = perf_counter()

    page_b64 = state["page_b64"]
    progress = state["progress"]
    

    print(f"⏳Processing OCR page📸 {progress[0]+1} of {progress[1]}...")

    
    llm , callback  = llm_select(state["llm_OCR_platform"])
    llm_structured = llm.with_structured_output(OCRExamResponse)
    
    
    # สร้างโจทย์ (Prompt) เพื่อให้โมเดลทำความเข้าใจโครงสร้างภาพและอ่านไฟล์ข้อสอบ
    prompt_text = """"
                        Extract all text from the image with Thai Language.

                        Instructions:
                        - Only return the clean Markdown.
                        - Do not include any explanation or extra text.
                        - You must include all information on the page.

                        Formatting Rules:
                        - Tables: Render tables using <table>...</table> in clean HTML format.
                        - Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
                        - Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

                        <figure>
                        Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.
                        </figure>

                        - Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
                        - Checkboxes: Use ☐ for unchecked and ☑ for checked boxes.
                        *response format with json format example:*
                        {
                        "ocr_results": [
                            {
                                "question_id": "1", 
                                "question_content": "โจทย์ข้อที่ 1 เนื้อหาโจทย์ (ไม่เอาตัวเลือก)", 
                                "image_description": "คำอธิบายรูปภาพอย่างละเอียด (ถ้ามี) เช่น ประจุ a อยู่ตำแหน่ง x=1 หรือถ้าไม่มีรูปให้ใส่เว้นว่าง", 
                                "choice": ["ก. ตัวเลือก ก", "ข. ตัวเลือก ข", "ค. ตัวเลือก ค", "ง. ตัวเลือก ง"]
                            }, 
                            {
                                "question_id": "2", 
                                "question_content": "โจทย์ข้อที่ 2 เนื้อหาโจทย์ (ไม่เอาตัวเลือก)", 
                                "image_description": "คำอธิบายรูปภาพอย่างละเอียด (ถ้ามี) เช่น ประจุ a อยู่ตำแหน่ง x=1 หรือถ้าไม่มีรูปให้ใส่เว้นว่าง", 
                                "choice": ["ก. ตัวเลือก ก", "ข. ตัวเลือก ข", "ค. ตัวเลือก ค", "ง. ตัวเลือก ง"]
                            }
                        ]
                        }
                         """
    sys_prompt = SystemMessage(content=[
        {"type": "text", "text": prompt_text},
        {"type": "text", "text": "ให้ตอบกลับมาในรูปแบบ JSON format เท่านั้น และต้องมีข้อมูลครบถ้วนตามที่กำหนดไว้ใน prompt เท่านั้น ห้ามมีคำอธิบายหรือข้อความอื่นใดนอกจาก JSON format ที่กำหนดเท่านั้น"}
    ])
    
    # ส่ง Message แบบระบุ base64 ใน image_url โดยใช้โครงสร้าง Dictionary สำหรับ OpenAI/Typhoon
    message = HumanMessage(
        content=[
            {"type": "text", "text": f"นี่คือหน้า {progress[0]+1} จาก {progress[1]} ของไฟล์ PDF ที่ถูกแปลงเป็น Base64 แล้ว"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_b64}"}}
        ]
    )
    
    response = llm_structured.invoke([sys_prompt, message])

    items = response.model_dump()

    end_ocr_page = perf_counter()

    log_token_usage(callback, 
                    start_date = start_date,
                    processtime = (end_ocr_page - strat_ocr_page),
                    platform = state["llm_OCR_platform"],
                    agent_work = "OCR and Extract information each page"
                    )

    
    ocr_results = items.get("ocr_results", [])
    

    return {"ocr_results": ocr_results}




  
    
    
