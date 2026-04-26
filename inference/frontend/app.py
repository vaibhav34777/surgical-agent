import streamlit as st
import requests
import sseclient
import json

st.set_page_config(page_title="Surgical Agent", page_icon="🩺", layout="centered")

st.title("Surgical AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "video_url" in msg and msg["video_url"]:
            st.video(msg["video_url"])

uploaded_file = st.file_uploader("Upload Surgical Video", type=["mp4", "avi", "mov"])
submit_button = st.button("Start Analysis", disabled=st.session_state.processing or not uploaded_file)

if uploaded_file and submit_button and not st.session_state.processing:
    st.session_state.processing = True
    
    with st.chat_message("user"):
        st.markdown(f"Analyze video: {uploaded_file.name}")
    st.session_state.messages.append({"role": "user", "content": f"Analyze video: {uploaded_file.name}"})
    
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("Uploading video...")
        
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post("http://localhost:8000/api/v1/analyze", files=files)
        
        if response.status_code == 200:
            job_id = response.json()["job_id"]
            
            stream_response = requests.get(f"http://localhost:8000/api/v1/stream/{job_id}", stream=True)
            client = sseclient.SSEClient(stream_response)
            
            for event in client.events():
                if event.event == "status":
                    data_str = json.loads(event.data)
                    status_placeholder.markdown(f"*{data_str}*")
                elif event.event == "result":
                    data = json.loads(event.data)
                    action = data.get("action")
                    clip_url = f"http://localhost:8000{data.get('clip_url')}"
                    insights = data.get("insights")
                    
                    st.markdown(f"**Action:** {action}")
                    st.video(clip_url)
                    st.markdown(insights)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"**Action:** {action}\n\n{insights}",
                        "video_url": clip_url
                    })
                elif event.event == "error":
                    st.error(json.loads(event.data))
                    break
                elif event.event == "end":
                    status_placeholder.empty()
                    st.session_state.processing = False
                    st.rerun()
        else:
            status_placeholder.markdown("Error uploading video.")
            st.session_state.processing = False
