import os
import google.generativeai as genai
from PIL import Image

def get_clinical_insights(image_paths: list, action_predicted: str, color_map_legend: str) -> str:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    imgs = [Image.open(p) for p in image_paths]
    
    prompt = f"""
You are an expert surgical assistant writing a formal clinical report for a laparoscopic cholecystectomy.
You are analyzing a surgical phase identified as: '{action_predicted}'.

Your task is to provide a structured clinical report for this phase. Use the provided visual sequence to understand the temporal progression, but write your report using professional headings.

STRICT FORMATTING RULES:
1. DO NOT mention "Frame 1", "Frame 2", "Frame 3", or "starting/middle/end frame".
2. DO NOT mention "images", "frames", "sequential view", or "provided visual data".
3. DO NOT mention "colors", "masks", "outlines", "labels", or specific colors like "magenta" or "green".
4. Use the provided segmentation outlines strictly as internal ground truth to identify anatomy and tools correctly.
5. USE THESE HEADINGS for your response:
   - **Maneuver Description**
   - **Anatomical Context**
   - **Clinical Intent**
   - **Safety Observations**

Clinical Requirements:
- Describe the tool interaction with the tissues and the progression of the maneuver.
- Explain the anatomical context and landmarks visible (e.g., cystic duct, liver bed, vascular structures).
- Discuss the clinical significance of the action (e.g., "exposure of Calot's triangle").
- Highlight specific surgical risks or observations (e.g., "proximity of the hepatic vein").

Anatomical Map (Internal Reference Only):
{color_map_legend}
"""
    response = model.generate_content([prompt] + imgs)
    return response.text
