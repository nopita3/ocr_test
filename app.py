from pathlib import Path

import json

from graphs.graph_process import graph_process
from Node import OCR

pdf_gemini = OCR.read_and_split_pdf
extracted_gemini = OCR.process_ocr_page
conditional_edges_gemini = OCR.continue_to_ocr

if __name__ == "__main__":

    for i in range(64,69):
        pdf_file_path = Path(f"Documents/posn1-{i}-physics.pdf")

        graph = graph_process(pdf_gemini, 
                                extracted_gemini,  
                                conditional_edges_gemini)
        
        config = {"configurable": {"thread_id": "1"}, "max_concurrency": 2}
        
        final_state = graph.invoke({"pdf_path": pdf_file_path, 
                                    "llm_OCR_platform": "gemini", 
                                    }, config=config)
        
        ocr_results = final_state.get("ocr_results", [])
        
        with open(f'files_log/posn1-{i}-ocr_results.json', 'w', encoding='utf-8') as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)

   
    
    

       