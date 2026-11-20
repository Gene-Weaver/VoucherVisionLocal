"""Prompt text, built exactly as VoucherVisionGO builds it.

Replaces the five `VoucherVision` members the VoucherVisionGO app used
(`initialize_token_counters`, `path_custom_prompts`, `setup_JSON_dict_structure`,
`JSON_dict_structure`, `setup_prompt`) without importing the VoucherVision
package. The parse prompt comes from the vendored `PromptCatalog.prompt_SLTP`,
then is wrapped the way VoucherVisionGO's Gemini handler wraps it (a LangChain
`PromptTemplate` around a `JsonOutputParser` with no schema), so the text an
engine receives is byte-for-byte what VoucherVisionGO sends.

OCR prompt selection mirrors `OCRGeminiProVision.ocr_gemini`: "verbatim_notebook"
selects the notebook prompt; anything else (including None) selects
"verbatim_with_annotations".
"""

from __future__ import annotations

import os

from vvl.engines.schema import json_schema_from_rules
from vvl.vendor_vv.OCR_Prompt_Catalog import OCRPromptCatalog
from vvl.vendor_vv.prompt_catalog import PromptCatalog

# LangChain: PromptTemplate(template="Answer the user query.\n{format_instructions}\n{query}\n")
# with JsonOutputParser().get_format_instructions() == "Return a JSON object."
PARSE_PROMPT_TEMPLATE = "Answer the user query.\n{format_instructions}\n{query}\n"
JSON_FORMAT_INSTRUCTIONS = "Return a JSON object."

OCR_PROMPT_NOTEBOOK = "verbatim_notebook"
OCR_PROMPT_DEFAULT = "verbatim_with_annotations"


def ocr_prompt_key(ocr_prompt_option: str | None) -> str:
    if ocr_prompt_option == OCR_PROMPT_NOTEBOOK:
        return OCR_PROMPT_NOTEBOOK
    return OCR_PROMPT_DEFAULT


def ocr_prompt_text(ocr_prompt_option: str | None) -> tuple[str, str]:
    """(catalog key, prompt text) for an OCR prompt option."""
    key = ocr_prompt_key(ocr_prompt_option)
    return key, OCRPromptCatalog().get_prompt(key)


def wrap_parse_prompt(prompt_text: str) -> str:
    return PARSE_PROMPT_TEMPLATE.format(format_instructions=JSON_FORMAT_INSTRUCTIONS, query=prompt_text)


class PromptBuilder:
    """Prompt setup for one prompt YAML file."""

    def __init__(self, path_custom_prompts: str | os.PathLike):
        self.path_custom_prompts = os.fspath(path_custom_prompts)
        self.initialize_token_counters()
        self.setup_JSON_dict_structure()

    def initialize_token_counters(self) -> None:
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    def setup_JSON_dict_structure(self) -> None:
        catalog = PromptCatalog()
        _, self.JSON_dict_structure = catalog.prompt_SLTP(self.path_custom_prompts, OCR="Text")
        self.rules = dict(catalog.rules_list)
        self.prompt_name = catalog.rules_config.get("prompt_name") if isinstance(catalog.rules_config, dict) else None

    def setup_prompt(self, ocr_text: str) -> str:
        """The prompt_SLTP text for this OCR text (VoucherVision.setup_prompt)."""
        prompt, _ = PromptCatalog().prompt_SLTP(self.path_custom_prompts, OCR=ocr_text or "")
        return prompt

    def parse_prompt_text(self, ocr_text: str) -> str:
        """The full text VoucherVisionGO sends to the parsing model."""
        return wrap_parse_prompt(self.setup_prompt(ocr_text))

    def json_schema(self) -> dict:
        return json_schema_from_rules(self.rules, title=self.prompt_name if isinstance(self.prompt_name, str) else None)
