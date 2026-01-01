import streamlit as st
import requests
import requests as req
import json
from dotenv import load_dotenv
from os import getenv

has_anything_loaded = load_dotenv()

if not has_anything_loaded:
    raise ValueError("No .env file found")

api_service_address = getenv("FASTAPI_DOCKER_SERVICE")

scene_generate_address = "http://{}:8000/generate".format(api_service_address)

if "in_progress" not in st.session_state:
    st.session_state.in_progress = False
    
def set_in_progress():
    st.session_state.in_progress = True

st.set_page_config(layout="wide", page_title="Ghostwriter Studio")    

st.title("🎬 Ghostwriter Studio")

# --- SIDEBAR (Settings) ---
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Writer Model", ["gpt-oss", "gpt-5.2"])
    temp = st.slider("Temperature", 0.0, 1.0, 0.7)
    st.markdown("---")
    st.info("💡 **Tip:** Use 'gpt-5.2' for production-quality subtext.")

# left_col, right_col = st.columns([4, 6], gap="large")

# with left_col:
    
#     with st.form(key="scene data"):
#         user_prompt = st.text_area("Describe your scene", disabled=st.session_state.in_progress)
#         temperature = st.slider("writer temperature", 0.0, 1.0, 0.7, 0.1, disabled=st.session_state.in_progress)
#         writer_model = st.radio("Model", ['gpt-oss', 'gpt-5.2'], captions=['gpt oss', 'gpt 5.2'], disabled=st.session_state.in_progress)
        
#         st.form_submit_button(on_click=set_in_progress, disabled=st.session_state.in_progress)


# with right_col:
#     scene_description = st.text("Your scene will be here")
#     scene = st.text("")

#     if user_prompt:
#         scene_description.text("scene description: \n{}\n".format(user_prompt))
        
#         with st.spinner("A new scene is being generated"):
#             raw_resp = req.post(scene_generate_address, data=json.dumps({
#                 "user_prompt": user_prompt
#             }))

#         resp = json.loads(raw_resp.content)

#         scene.write(resp)

# --- MAIN INPUT ---
col_input, col_btn = st.columns([4, 1])
with col_input:
    user_prompt = st.text_input("Describe the scene:", placeholder="e.g. A tense standoff in a diner...")
with col_btn:
    # Spacer to align button
    st.write("") 
    st.write("")
    generate_btn = st.button("Action! 🎬", type="primary", use_container_width=True)

# --- RESULTS AREA ---
if generate_btn and user_prompt:
    with st.spinner("🧠 Analyzing style... 📝 Writing script... 🧐 Critiquing..."):
        try:
            # 1. CALL THE API
            payload = {
                "user_prompt": user_prompt,
                "writer_model": model_choice,
                "temperature": temp
            }
            # Note: We use http://127.0.0.1:8000 because 'localhost' sometimes fails on Mac
            response = requests.post("http://127.0.0.1:8000/generate", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                # 2. DISPLAY LAYOUT
                col_logic, col_script = st.columns([1, 2])
                
                # --- LEFT COLUMN (The Brain) ---
                with col_logic:
                    st.subheader("🧠 The Logic")
                    
                    # Score Card
                    score = data['critique_score']
                    if score >= 4:
                        st.success(f"**Score: {score}/5.0** (Excellent)")
                    elif score >= 3:
                        st.warning(f"**Score: {score}/5.0** (Passable)")
                    else:
                        st.error(f"**Score: {score}/5.0** (Needs Work)")
                        
                    with st.expander("🧐 Critique (Why this score?)", expanded=True):
                        st.write(data['critique_text'])

                    st.markdown("---")
                    
                    with st.expander("📐 Logical Plan"):
                        st.markdown(data['logical_plan'])
                        
                    with st.expander("🎨 Style Plan"):
                        st.markdown(data['style_plan'])
                        
                    with st.expander("📚 References Used"):
                        for i, ref in enumerate(data['referenced_scenes']):
                            st.text_area(f"Ref {i+1}", ref, height=150)

                # --- RIGHT COLUMN (The Script) ---
                with col_script:
                    st.subheader("📝 The Script")
                    # st.text_area("Final Output", value=data['generated_scene'], height=800)
                    st.markdown(data['generated_scene'])
            
            else:
                st.error(f"API Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API. Is the server running? (`uvicorn api.main:app`)")