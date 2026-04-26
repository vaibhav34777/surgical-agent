import streamlit as st
import requests
import sseclient
import json
import os
import base64

INTERNAL_API_URL = os.environ.get("INTERNAL_API_URL", "http://localhost:8000")
EXTERNAL_API_URL = os.environ.get("EXTERNAL_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Surgical Agent", page_icon="🩺", layout="centered")

st.title("Surgical AI Agent")

with st.sidebar:
    st.header("Real-Time Telemetry")
    rendezvous_metric = st.empty()
    segformer_metric = st.empty()
    gemini_metric = st.empty()
    
    rendezvous_metric.metric("Rendezvous Latency", "0 ms")
    segformer_metric.metric("SegFormer Latency", "0.000 s")
    gemini_metric.metric("Gemini API Latency", "0.0 s")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "timeline" not in st.session_state:
    st.session_state.timeline = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "job_id" not in st.session_state:
    st.session_state.job_id = None

if "results" not in st.session_state:
    st.session_state.results = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "video_url" in msg and msg["video_url"]:
            st.video(msg["video_url"])

if st.session_state.timeline:
    st.markdown("### Surgical Timeline")
    for item in st.session_state.timeline:
        st.markdown(item)

uploaded_file = st.file_uploader("Upload Surgical Video", type=["mp4", "avi", "mov"])
submit_button = st.button("Start Analysis", disabled=st.session_state.processing or not uploaded_file)

if uploaded_file and submit_button and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.timeline = []
    st.session_state.results = []
    
    with st.chat_message("user"):
        st.markdown(f"Analyze video: {uploaded_file.name}")
    st.session_state.messages.append({"role": "user", "content": f"Analyze video: {uploaded_file.name}"})
    
    with st.chat_message("assistant"):
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        status_placeholder.markdown("Uploading video...")
        
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{INTERNAL_API_URL}/api/v1/analyze", files=files)
        
        if response.status_code == 200:
            job_id = response.json()["job_id"]
            st.session_state.job_id = job_id
            
            stream_response = requests.get(f"{INTERNAL_API_URL}/api/v1/stream/{job_id}", stream=True)
            client = sseclient.SSEClient(stream_response)
            
            for event in client.events():
                if event.event == "status":
                    data_str = json.loads(event.data)
                    status_placeholder.markdown(f"*{data_str}*")
                elif event.event == "telemetry":
                    data = json.loads(event.data)
                    if "progress" in data:
                        progress_bar.progress(data["progress"])
                    if "rendezvous_ms" in data:
                        rendezvous_metric.metric("Rendezvous Latency", f"{data['rendezvous_ms']} ms")
                    if "segformer_s" in data:
                        segformer_metric.metric("SegFormer Latency", f"{data['segformer_s']} s")
                    if "gemini_s" in data:
                        gemini_metric.metric("Gemini API Latency", f"{data['gemini_s']} s")
                elif event.event == "result":
                    data = json.loads(event.data)
                    action = data.get("action")
                    clip_url = f"{EXTERNAL_API_URL}{data.get('clip_url')}"
                    insights = data.get("insights")
                    timestamp = data.get("timestamp")
                    
                    st.markdown(f"**Action:** {action}")
                    st.video(clip_url)
                    st.markdown(insights)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"**Action:** {action}\n\n{insights}",
                        "video_url": clip_url
                    })
                    
                    st.session_state.timeline.append(f"- **{timestamp}** : {action}")
                    st.session_state.results.append(data)
                    
                elif event.event == "error":
                    st.error(json.loads(event.data))
                    break
                elif event.event == "end":
                    status_placeholder.empty()
                    progress_bar.progress(100)
                    st.session_state.processing = False
                    st.rerun()
        else:
            status_placeholder.markdown("Error uploading video.")
            st.session_state.processing = False

if not st.session_state.processing and st.session_state.results:
    if st.button("Generate Clinical Report (PDF)"):
        with st.spinner("Generating PDF..."):
            pdf_response = requests.post(
                f"{INTERNAL_API_URL}/api/v1/generate-report",
                json={"results": st.session_state.results}
            )
            if pdf_response.status_code == 200:
                pdf_bytes = pdf_response.content
                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                st.download_button("Download Report", data=pdf_bytes, file_name="Surgical_Report.pdf", mime="application/pdf")
            else:
                st.error(f"Failed to generate report: {pdf_response.text}")

