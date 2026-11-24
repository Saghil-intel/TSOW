
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
    Parse the TSOW template with LlamaParse.
    .env switches:
      LLAMAPARSE_PREMIUM=true   -> option 2 (higher-accuracy mode, more credits)
      LLAMAPARSE_INSTRUCTION=.. -> override the default parsing instruction
    """
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY is missing from .env")

    instruction = os.getenv(
        "LLAMAPARSE_INSTRUCTION",
        "This document is a form template made of tables. Preserve EVERY label "
        "and field name exactly as printed, in reading order. This includes "
        "single-cell rows, labels with empty answer cells, merged cells and "
        "header rows. Never skip or merge a label because it has no content "
        "next to it. Output each label as its own table row or heading.",
    )
    premium = os.getenv("LLAMAPARSE_PREMIUM", "false").lower() == "true"

    # Try the full option set first; drop options this installed version rejects.
    attempts = [
        dict(parsing_instruction=instruction, premium_mode=premium, invalidate_cache=True),
        dict(parsing_instruction=instruction, premium_mode=premium),
        dict(parsing_instruction=instruction),
        dict(),
    ]

    parser = None
    for extra in attempts:
        try:
            parser = LlamaParse(api_key=api_key, result_type="markdown", **extra)
            print(f"[LLAMAPARSE] Options in use: {list(extra) or 'defaults'}")
            break
        except Exception as e:
            print(f"[LLAMAPARSE] Options {list(extra)} not supported: {e}")

    try:
        documents = parser.load_data(file_path)
        markdown = "\n\n".join(d.text for d in documents)

        if not markdown or not markdown.strip():
            raise ValueError("LlamaParse returned empty TSOW content.")

        print("[LLAMAPARSE] TSOW template parsed successfully.")
        return markdown

    except Exception as e:
        raise RuntimeError(f"LlamaParse failed: {str(e)}") from e

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
