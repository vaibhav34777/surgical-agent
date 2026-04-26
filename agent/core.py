import os
import google.generativeai as genai
from PIL import Image

def get_clinical_insights(image_path: str, action_predicted: str, color_map_legend: str) -> str:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    img = Image.open(image_path)
    
    prompt = f"""
You are an expert surgical assistant analyzing an endoscopic surgical frame.
The video model has predicted the current surgical action as: '{action_predicted}'.
The provided image has been processed with an AI segmentation model to help you identify structures. 
The colored outlines map exactly to these anatomical and tool classes:
{color_map_legend}

CRITICAL INSTRUCTION: DO NOT explicitly mention "colors", "masks", "outlines", "magenta", "red", etc. in your response. 
You must analyze the image naturally as a surgeon would, using the color map strictly internally to correctly identify the tissues and tools. 
Based on the predicted action and the structures you observe, provide detailed, professional clinical insights. 
Analyze the tool's positioning, its interaction with the tissues, the anatomical context, and any potential risks or critical observations.
"""
    response = model.generate_content([prompt, img])
    return response.text
