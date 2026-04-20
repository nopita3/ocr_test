from pydantic import BaseModel , Field , RootModel
from typing import Annotated, TypedDict
import operator
from pathlib import Path



class OCRResult(BaseModel):
    question_id: str = Field( description="เลขข้อ")
    question_content: str = Field( description="เนื้อหาโจทย์ (ไม่เอาตัวเลือก)")
    image_description: str = Field( description="คำอธิบายรูปภาพอย่างละเอียด (ถ้ามี) เช่น ประจุ a อยู่ตำแหน่ง x=1 หรือถ้าไม่มีรูปให้ใส่เว้นว่าง")
    choice: list[str] = Field( description="ตัวเลือก ก. ข. ค. ง. (ถ้ามี) ถ้าไม่มีตัวเลือกให้ใส่เว้นว่าง")

class OCRExamResponse(BaseModel):
    ocr_results: list[OCRResult] = Field( description="ผลลัพธ์การทำ OCR และวิเคราะห์โจทย์ข้อสอบฟิสิกส์ที่อิงตามหลักสูตรแกนกลางของกระทรวงศึกษาธิการไทย ในส่วนของวิชาฟิสิกส์ (เพิ่มเติม) 4 เรื่องไฟฟ้าสถิตและไฟฟ้ากระแสตรง โดยมีรูปแบบเป็น list ของ OCRResult")


# State ย่อยสำหรับการทำ Map-Reduce (ส่งข้อมูลหน้าเดี่ยวไปในแต่ละ Node)
class PageState(TypedDict):
    page_b64: str
    page_text: str
    progress: list[int,int]
    llm_OCR_platform: str
    



# กำหนดรูปแบบ State ของระบบ (อัปเดตกลับมาเป็น Parallel)
class OverallState(TypedDict):

    llm_OCR_platform: str 
    pdf_path: Path
    pages: list[str]
    pages_text: list[str]
    ocr_results: Annotated[OCRExamResponse, operator.add] = Field(default_factory=list) # ใช้ operator.add เพื่อรวบรวม Results จาก Parallel Nodes
    

