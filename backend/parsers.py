
# parsers.py

import os

from docling.document_converter import DocumentConverter
from llama_parse import LlamaParse
from dotenv import load_dotenv

load_dotenv()


# Initialize Docling once and reuse it
docling_converter = DocumentConverter()


def parse_with_docling(file_path: str) -> str:
    """
    Parse transcript or questionnaire using Docling.
    Returns extracted Markdown.
    """

    try:
        result = docling_converter.convert(file_path)

        markdown = result.document.export_to_markdown()

        if not markdown or not markdown.strip():
            raise ValueError(
                f"No content extracted from: {file_path}"
            )

        print(
            f"[DOCLING] Successfully parsed: "
            f"{os.path.basename(file_path)}"
        )

        return markdown

    except Exception as e:
        raise RuntimeError(
            f"Docling parsing failed for "
            f"{os.path.basename(file_path)}: {str(e)}"
        ) from e


def parse_tsow_with_llamaparse(file_path: str) -> str:
    """
    Parse Intel TSOW template using LlamaParse.
    Returns extracted Markdown structure.
    """

    api_key = os.getenv("LLAMA_CLOUD_API_KEY")

    if not api_key:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY is missing from .env"
        )

    try:

        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown"
        )

        documents = parser.load_data(file_path)

        markdown = "\n\n".join(
            document.text
            for document in documents
        )

        if not markdown or not markdown.strip():
            raise ValueError(
                "LlamaParse returned empty TSOW content."
            )

        print("[LLAMAPARSE] TSOW template parsed successfully.")

        return markdown

    except Exception as e:
        raise RuntimeError(
            f"LlamaParse failed: {str(e)}"
        ) from e

import tempfile
import threading

# ==========================================
# PERSISTENT TSOW TEMPLATE STORAGE
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TSOW_STORAGE_DIR = os.getenv(
    "TSOW_STORAGE_DIR",
    os.path.join(BASE_DIR, "data")
)

TSOW_STORAGE_PATH = os.path.join(
    TSOW_STORAGE_DIR,
    "tsow_template.md"
)

# Prevent simultaneous updates within one application process
template_lock = threading.Lock()


def get_tsow_template(file_path: str = None) -> str:
    """
    If a new TSOW is uploaded:
        1. Parse using LlamaParse.
        2. Save extracted Markdown.
        3. Replace the previously stored template.

    If no new TSOW is uploaded:
        1. Read the previously stored Markdown.
        2. Return it.

    Returns:
        Extracted TSOW Markdown.
    """

    # ======================================
    # CASE 1: NEW TSOW UPLOADED
    # ======================================

    if file_path:

        print("\n[TSOW] New template detected.")

        # Parse the new template first.
        # If parsing fails, the old template remains untouched.

        new_markdown = parse_tsow_with_llamaparse(
            file_path
        )

        os.makedirs(
            TSOW_STORAGE_DIR,
            exist_ok=True
        )

        # Write to a temporary file in the same directory.
        # os.replace ensures the old file is replaced atomically.

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".tmp",
                dir=TSOW_STORAGE_DIR,
                delete=False
            ) as temp_file:

                temp_file.write(new_markdown)
                temp_path = temp_file.name

            with template_lock:

                os.replace(
                    temp_path,
                    TSOW_STORAGE_PATH
                )

            print(
                "[TSOW] New template saved successfully."
            )

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        return new_markdown

    # ======================================
    # CASE 2: NO NEW TSOW UPLOADED
    # ======================================

    with template_lock:

        if not os.path.exists(TSOW_STORAGE_PATH):

            raise FileNotFoundError(
                "No TSOW template is stored. "
                "Please upload a TSOW template first."
            )

        with open(
            TSOW_STORAGE_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            stored_markdown = file.read()

    if not stored_markdown.strip():

        raise ValueError(
            "Stored TSOW template is empty."
        )

    print(
        "[TSOW] Using previously stored template."
    )

    return stored_markdown
