
# main.py

import os
import tempfile
import json

from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from starlette.concurrency import run_in_threadpool

from parsers import (
    parse_with_docling,
    get_tsow_template
)

from llm import convert_to_json


app = FastAPI(
    title="TSoW AI Backend",
    version="1.0"
)


async def save_upload(
    upload: UploadFile,
    directory: str
) -> str:
    """
    Save uploaded files into a temporary directory.
    """

    filename = Path(
        upload.filename or "uploaded_file"
    ).name

    file_path = os.path.join(
        directory,
        filename
    )

    content = await upload.read()

    if not content:
        raise ValueError(
            f"Uploaded file is empty: {filename}"
        )

    with open(file_path, "wb") as f:
        f.write(content)

    return file_path


@app.get("/")
def home():

    return {
        "message": "TSoW AI Backend is running"
    }


@app.post("/process")
async def process_documents(

    transcript: UploadFile = File(...),

    questionnaire: UploadFile = File(...),

    tsow_template: UploadFile = File(None),

    user_prompt: str = Form("")

):

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            # ==========================================
            # STEP 1: SAVE UPLOADED DOCUMENTS
            # ==========================================

            print("\n========== RECEIVING DOCUMENTS ==========")

            transcript_path = await save_upload(
                transcript,
                temp_dir
            )

            questionnaire_path = await save_upload(
                questionnaire,
                temp_dir
            )

            # TSOW is optional

            tsow_path = None

            if tsow_template is not None:
            
                tsow_path = await save_upload(
                    tsow_template,
                    temp_dir
                )

                print(
                    "New TSOW uploaded:",
                    tsow_template.filename
                )

            else:
            
                print(
                    "No new TSOW uploaded. "
                    "Using previously stored template."
                )


            # ==========================================
            # STEP 2: PARSE TRANSCRIPT USING DOCLING
            # ==========================================

            print(
                "\n========== PARSING TRANSCRIPT =========="
            )

            transcript_text = await run_in_threadpool(
                parse_with_docling,
                transcript_path
            )

            print(
                "Transcript parsed successfully."
            )

            print(
                "Transcript characters:",
                len(transcript_text)
            )


            # ==========================================
            # STEP 3: PARSE QUESTIONNAIRE USING DOCLING
            # ==========================================

            print(
                "\n========== PARSING QUESTIONNAIRE =========="
            )

            questionnaire_text = await run_in_threadpool(
                parse_with_docling,
                questionnaire_path
            )

            print(
                "Questionnaire parsed successfully."
            )

            print(
                "Questionnaire characters:",
                len(questionnaire_text)
            )


            # ==========================================
            # STEP 4: PARSE TSOW USING LLAMAPARSE
            # ==========================================

            print(
                "\n========== LOADING TSOW TEMPLATE =========="
            )

            tsow_text = await run_in_threadpool(
                get_tsow_template,
                tsow_path
            )

            print(
                "TSOW template loaded successfully."
            )

            print(
                "TSOW characters:",
                len(tsow_text)
            )



            

            # # Save parsed output for inspection

            # debug_path = os.path.join(
            #     temp_dir,
            #     "debug_tsow_extracted.md"
            # )

            # with open(debug_path, "w", encoding="utf-8") as f:
            #     f.write(tsow_text)


            # ==========================================
            # STEP 5: GENERATE TSOW JSON USING QWEN
            # ==========================================

            print(
                "\n========== GENERATING TSOW JSON =========="
            )

            json_output = await run_in_threadpool(

                convert_to_json,

                tsow_text,

                transcript_text,

                questionnaire_text,

                user_prompt

            )


            # ==========================================
            # STEP 6: RETURN FINAL JSON
            # ==========================================

            print(
                "\n========== PROCESSING COMPLETED =========="
            )
            print("\n========== FILE PATH VERIFICATION ==========")

            # print("Transcript:", transcript_path)
            # print("Questionnaire:", questionnaire_path)
            # print("TSOW:", tsow_path)

            return {

                "status": "success",

                "message": (
                    "Documents parsed and TSOW JSON "
                    "generated successfully."
                ),

                "data": json_output

            }
        


    except FileNotFoundError as e:
    
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    except Exception as e:
    
        print(
            "\n========== PROCESSING ERROR =========="
        )
    
        print(str(e))
    
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
