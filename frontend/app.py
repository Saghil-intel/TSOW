
import streamlit as st
import requests
import json

st.set_page_config(
    page_title="TSoW AI Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("TSoW AI Assistant")

st.write(
    "Upload your documents and provide instructions "
    "to generate structured JSON."
)

st.divider()

# API endpoint
API_URL = "http://localhost:8000/process"

# Document uploads
st.subheader("Upload Documents")

transcript = st.file_uploader(
    "Upload Transcript",
    type=["pdf", "docx", "txt"]
)

questionnaire = st.file_uploader(
    "Upload Questionnaire",
    type=["pdf", "docx", "txt"]
)

tsow_template = st.file_uploader(
    "Upload TSoW Template",
    type=["pdf", "docx", "txt"]
)

st.divider()

# User prompt
st.subheader("Chat Box")

user_prompt = st.text_area(
    "Enter your instructions",
    placeholder="Describe what you want to extract from the questionnaire...",
    height=150
)

# Process button
if st.button("Process Documents", type="primary"):

    if not transcript or not questionnaire or not tsow_template:
        st.error("Please upload all three documents.")

    elif not user_prompt.strip():
        st.error("Please enter your instructions.")

    else:

        with st.spinner("Processing documents..."):

            try:

                files = {
                    "transcript": (
                        transcript.name,
                        transcript.getvalue()
                    ),

                    "questionnaire": (
                        questionnaire.name,
                        questionnaire.getvalue()
                    ),

                    "tsow_template": (
                        tsow_template.name,
                        tsow_template.getvalue()
                    )
                }

                data = {
                    "user_prompt": user_prompt
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    data=data,
                    timeout=600
                )

                response.raise_for_status()

                result = response.json()

                st.success("Processing completed!")

                st.subheader("Generated JSON")

                st.json(result["json_output"])

                st.download_button(
                    label="Download JSON",
                    data=json.dumps(
                        result["json_output"],
                        indent=4,
                        ensure_ascii=False
                    ),
                    file_name="questionnaire_output.json",
                    mime="application/json"
                )

            except requests.exceptions.RequestException as e:

                st.error(f"Backend connection error: {e}")

            except Exception as e:

                st.error(f"Error: {e}")
