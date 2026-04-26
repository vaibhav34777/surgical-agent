import asyncio
import os
import cv2
import uuid
import sys
import time
from core.config import settings
from core.job_store import job_store
from services.video_service import VideoService
from services.seg_service import SegService
from utils.video_utils import draw_outlines, draw_filled_masks, create_action_clip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from agent.core import get_clinical_insights

TRIPLET_DICT = {
    0: "grasper-dissect-cystic_plate", 1: "grasper-dissect-gallbladder", 2: "grasper-dissect-omentum", 3: "grasper-grasp-cystic_artery",
    4: "grasper-grasp-cystic_duct", 5: "grasper-grasp-cystic_pedicle", 6: "grasper-grasp-cystic_plate", 7: "grasper-grasp-gallbladder",
    8: "grasper-grasp-gut", 9: "grasper-grasp-liver", 10: "grasper-grasp-omentum", 11: "grasper-grasp-peritoneum", 12: "grasper-grasp-specimen_bag",
    13: "grasper-pack-gallbladder", 14: "grasper-retract-cystic_duct", 15: "grasper-retract-cystic_pedicle", 16: "grasper-retract-cystic_plate",
    17: "grasper-retract-gallbladder", 18: "grasper-retract-gut", 19: "grasper-retract-liver", 20: "grasper-retract-omentum",
    21: "grasper-retract-peritoneum", 22: "bipolar-coagulate-abdominal_wall_cavity", 23: "bipolar-coagulate-blood_vessel",
    24: "bipolar-coagulate-cystic_artery", 25: "bipolar-coagulate-cystic_duct", 26: "bipolar-coagulate-cystic_pedicle",
    27: "bipolar-coagulate-cystic_plate", 28: "bipolar-coagulate-gallbladder", 29: "bipolar-coagulate-liver", 30: "bipolar-coagulate-omentum",
    31: "bipolar-coagulate-peritoneum", 32: "bipolar-dissect-adhesion", 33: "bipolar-dissect-cystic_artery", 34: "bipolar-dissect-cystic_duct",
    35: "bipolar-dissect-cystic_plate", 36: "bipolar-dissect-gallbladder", 37: "bipolar-dissect-omentum", 38: "bipolar-grasp-cystic_plate",
    39: "bipolar-grasp-liver", 40: "bipolar-grasp-specimen_bag", 41: "bipolar-retract-cystic_duct", 42: "bipolar-retract-cystic_pedicle",
    43: "bipolar-retract-gallbladder", 44: "bipolar-retract-liver", 45: "bipolar-retract-omentum", 46: "hook-coagulate-blood_vessel",
    47: "hook-coagulate-cystic_artery", 48: "hook-coagulate-cystic_duct", 49: "hook-coagulate-cystic_pedicle", 50: "hook-coagulate-cystic_plate",
    51: "hook-coagulate-gallbladder", 52: "hook-coagulate-liver", 53: "hook-coagulate-omentum", 54: "hook-cut-blood_vessel",
    55: "hook-cut-peritoneum", 56: "hook-dissect-blood_vessel", 57: "hook-dissect-cystic_artery", 58: "hook-dissect-cystic_duct",
    59: "hook-dissect-cystic_plate", 60: "hook-dissect-gallbladder", 61: "hook-dissect-omentum", 62: "hook-dissect-peritoneum",
    63: "hook-retract-gallbladder", 64: "hook-retract-liver", 65: "scissors-coagulate-omentum", 66: "scissors-cut-adhesion",
    67: "scissors-cut-blood_vessel", 68: "scissors-cut-cystic_artery", 69: "scissors-cut-cystic_duct", 70: "scissors-cut-cystic_plate",
    71: "scissors-cut-liver", 72: "scissors-cut-omentum", 73: "scissors-cut-peritoneum", 74: "scissors-dissect-cystic_plate",
    75: "scissors-dissect-gallbladder", 76: "scissors-dissect-omentum", 77: "clipper-clip-blood_vessel", 78: "clipper-clip-cystic_artery",
    79: "clipper-clip-cystic_duct", 80: "clipper-clip-cystic_pedicle", 81: "clipper-clip-cystic_plate", 82: "irrigator-aspirate-fluid",
    83: "irrigator-dissect-cystic_duct", 84: "irrigator-dissect-cystic_pedicle", 85: "irrigator-dissect-cystic_plate",
    86: "irrigator-dissect-gallbladder", 87: "irrigator-dissect-omentum", 88: "irrigator-irrigate-abdominal_wall_cavity",
    89: "irrigator-irrigate-cystic_pedicle", 90: "irrigator-irrigate-liver", 91: "irrigator-retract-gallbladder", 92: "irrigator-retract-liver",
    93: "irrigator-retract-omentum", 94: "grasper-null_verb-null_target", 95: "bipolar-null_verb-null_target", 96: "hook-null_verb-null_target",
    97: "scissors-null_verb-null_target", 98: "clipper-null_verb-null_target", 99: "irrigator-null_verb-null_target"
}

COLOR_LEGEND = """
- Bright Red: Abdominal Wall
- Bright Green: Liver
- Bright Blue: Gastrointestinal Tract
- Bright Yellow: Fat
- Bright Magenta: Grasper
- Bright Cyan: Connective Tissue
- Bright Orange: Cystic Duct
- Bright Purple: L-Hook Electrocautery
- Bright White: Hepatic Vein
"""

def format_timestamp(frame_idx, fps):
    if fps <= 0: fps = 25.0
    sec = int(frame_idx / fps)
    return f"{sec // 60:02d}:{sec % 60:02d}"

def extract_and_save_3_frames(cap, seg_svc, start_f, end_f, job_id, fps):
    mid_f = start_f + (end_f - start_f) // 2
    frames_to_extract = [start_f, mid_f, end_f]
    outlines = []
    filled = []
    timestamps = []
    
    for idx, f_idx in enumerate(frames_to_extract):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            mask = seg_svc.predict(frame)
            out_img = draw_outlines(frame, mask)
            fill_img = draw_filled_masks(frame, mask)
            
            out_path = os.path.join(settings.MEDIA_DIR, f"{job_id}_{f_idx}_out.png")
            fill_path = os.path.join(settings.MEDIA_DIR, f"{job_id}_{f_idx}_fill.png")
            cv2.imwrite(out_path, out_img)
            cv2.imwrite(fill_path, fill_img)
            
            outlines.append(out_path)
            filled.append(fill_path)
            timestamps.append(format_timestamp(f_idx, fps))
            
    return outlines, filled, timestamps

async def run_pipeline(job_id: str, video_path: str):
    queue = job_store.get_queue(job_id)
    cap = None
    try:
        await queue.put({"event": "status", "data": "Initializing models..."})
        video_svc = VideoService(settings.RENDEZVOUS_MODEL_PATH)
        seg_svc = SegService(settings.SEG_MODEL_PATH)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        await queue.put({"event": "status", "data": f"Processing {total_frames} frames..."})
        
        current_action = None
        action_start_frame = 0
        
        for i in range(0, total_frames, 30):
            await asyncio.sleep(0.01)
            t0 = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            actions = await asyncio.to_thread(video_svc.predict_frame, frame)
            rendezvous_ms = int((time.time() - t0) * 1000)
            
            progress = int((i / total_frames) * 100)
            await queue.put({"event": "telemetry", "data": {"progress": progress, "rendezvous_ms": rendezvous_ms}})
            
            if actions:
                new_action = TRIPLET_DICT.get(actions[0], f"Action_{actions[0]}")
                if new_action != current_action:
                    if current_action is not None:
                        await queue.put({"event": "status", "data": f"Generating segmented video clip for {current_action}..."})
                        clip_name = f"{job_id}_{current_action}.mp4"
                        clip_path = os.path.join(settings.MEDIA_DIR, clip_name)
                        
                        t0 = time.time()
                        await asyncio.to_thread(create_action_clip, video_path, action_start_frame, i, seg_svc, clip_path)
                        segformer_s = round((time.time() - t0) / max(1, i - action_start_frame), 3)
                        await queue.put({"event": "telemetry", "data": {"segformer_s": segformer_s}})
                        
                        await queue.put({"event": "status", "data": f"Extracting clinical insights from Gemini for {current_action}..."})
                        
                        outlines, filled, timestamps = await asyncio.to_thread(
                            extract_and_save_3_frames, cap, seg_svc, action_start_frame, i-1, job_id, fps
                        )
                        
                        t0 = time.time()
                        insights = await asyncio.to_thread(get_clinical_insights, outlines, current_action, COLOR_LEGEND)
                        gemini_s = round(time.time() - t0, 1)
                        await queue.put({"event": "telemetry", "data": {"gemini_s": gemini_s}})
                        
                        result_data = {
                            "action": current_action,
                            "clip_url": f"/api/v1/media/{clip_name}",
                            "insights": insights,
                            "timestamp": timestamps[0],
                            "keyframes_filled": filled,
                            "keyframe_timestamps": timestamps,
                            "clip_path": clip_path
                        }
                        
                        job_store.save_action(job_id, result_data)
                        
                        await queue.put({
                            "event": "result",
                            "data": result_data
                        })
                    
                    current_action = new_action
                    action_start_frame = i

        await queue.put({"event": "status", "data": "Complete"})
        await queue.put({"event": "end", "data": "done"})
    except Exception as e:
        await queue.put({"event": "error", "data": str(e)})
    finally:
        if cap is not None:
            cap.release()
