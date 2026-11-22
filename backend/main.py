# main.py

import os
import tempfile

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from starlette.concurrency import run_in_threadpool

from parsers import parse_with_docling, get_tsow_template
from llm import convert_to_json


app = FastAPI(title="TSoW AI Backend", version="2.0")


async def save_upload(upload: UploadFile, directory: str) -> str:
    filename = Path(upload.filename or "uploaded_file").name
    file_path = os.path.join(directory, filename)

    content = await upload.read()
    if not content:
        raise ValueError(f"Uploaded file is empty: {filename}")

    with open(file_path, "wb") as f:
        f.write(content)

    return file_path


@app.get("/")
def home():
    return {"message": "TSoW AI Backend is running"}


@app.post("/process")
async def process_documents(
    questionnaire: UploadFile = File(...),
    tsow_template: UploadFile = File(None),
    user_prompt: str = Form("")
):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            # 1. Save uploads
            print("\n========== RECEIVING DOCUMENTS ==========")

            questionnaire_path = await save_upload(questionnaire, temp_dir)

            tsow_path = None
            if tsow_template is not None:
                tsow_path = await save_upload(tsow_template, temp_dir)
                print("New TSOW uploaded:", tsow_template.filename)
            else:
                print("No new TSOW uploaded. Using stored template (if any).")

            # 2. Parse questionnaire
            print("\n========== PARSING QUESTIONNAIRE ==========")
            questionnaire_text = await run_in_threadpool(
                parse_with_docling, questionnaire_path
            )
            print("Questionnaire characters:", len(questionnaire_text))

            # 3. Load TSOW template (optional)
            print("\n========== LOADING TSOW TEMPLATE ==========")
            try:
                tsow_text = await run_in_threadpool(get_tsow_template, tsow_path)
                print("TSOW characters:", len(tsow_text))
            except FileNotFoundError:
                # nothing uploaded and nothing stored -> pipeline uses all KB fields
                tsow_text = None
                print("No TSOW template available. Using default fields.")

            # 4. Generate answers
            print("\n========== GENERATING TSOW JSON ==========")
            json_output = await run_in_threadpool(
                convert_to_json,
                tsow_text,
                questionnaire_text,
                user_prompt
            )

            return {
                "status": "success",
                "message": "Questionnaire parsed and TSOW JSON generated successfully.",
                "data": json_output
            }

    except HTTPException:
        raise

    except Exception as e:
        print("\n========== PROCESSING ERROR ==========")
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))