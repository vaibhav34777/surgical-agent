import cv2
import numpy as np
import os
import imageio

COLOR_MAP = [
    (0, 0, 0),        
    (0, 0, 255),      
    (0, 255, 0),      
    (255, 0, 0),      
    (0, 255, 255),    
    (255, 0, 255),    
    (255, 255, 0),    
    (0, 165, 255),    
    (255, 0, 128),    
    (255, 255, 255)   
]

CLASS_NAMES = [
    "Background", "Abd. Wall", "Liver", "GI Tract", "Fat", 
    "Grasper", "Conn. Tissue", "Cystic Duct", "L-Hook", "Hepatic Vein"
]

IMPORTANT_CLASSES = [5, 7, 8, 9] # Grasper, Duct, L-Hook, Vein

def add_labels(img, binary_mask, text, color):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 100:
            x, y, w, h = cv2.boundingRect(largest)
            text_y = y - 5 if y > 20 else y + h + 20
            # Draw black background for text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x, text_y - th - 5), (x + tw, text_y + 5), (0, 0, 0), -1)
            cv2.putText(img, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_outlines(frame, mask):
    outlined = frame.copy()
    for class_id in range(1, len(COLOR_MAP)):
        binary_mask = (mask == class_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outlined, contours, -1, COLOR_MAP[class_id], 2)
        add_labels(outlined, binary_mask, CLASS_NAMES[class_id], COLOR_MAP[class_id])
    return outlined

def draw_filled_masks(frame, mask, alpha=0.5):
    overlay = frame.copy()
    for class_id in IMPORTANT_CLASSES:
        binary_mask = (mask == class_id).astype(np.uint8)
        overlay[binary_mask == 1] = COLOR_MAP[class_id]
    
    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Add labels on top of the blended frame so they don't get washed out
    for class_id in IMPORTANT_CLASSES:
        binary_mask = (mask == class_id).astype(np.uint8)
        add_labels(blended, binary_mask, CLASS_NAMES[class_id], COLOR_MAP[class_id])
        
    return blended

def create_action_clip(video_path, start_frame, end_frame, seg_service, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 25.0
    
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', macro_block_size=None)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret: break
        mask = seg_service.predict(frame)
        processed = draw_filled_masks(frame, mask)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        writer.append_data(processed_rgb)
        
    cap.release()
    writer.close()
