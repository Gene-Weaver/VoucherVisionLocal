"""The demo engine (`demo:mock`): fixed sample output, no model.

Used so the whole pipeline (validation, resize, TextCollage, prompt loading,
response assembly) can be exercised before any model is installed. The sample
is a real VoucherVisionGO result for a Denver Botanic Gardens specimen
(tests/fixtures/vvgo_results_sample.json); it has nothing to do with the image
the user sent, and every response says so in `engine_notice`.
"""

from __future__ import annotations

import json

from vvl.engines.base import (
    EngineCapabilities,
    OCRRequest,
    OCRResult,
    ParseRequest,
    ParseResult,
)

MODEL_ID = "demo:mock"
LABEL = "Demo engine (sample output, no model)"
SHORT_LABEL = "Demo engine"

DEMO_NOTICE = (
    "Demo output: this is a fixed sample record from the built-in demo engine, "
    "not a transcription of your image. Choose a local or API model for real results."
)

SAMPLE_OCR_TEXT = (
    "inches\n2\n3\nKATHRYN KALMBACH HERBARIUM\nDENVER BOTANIC GARDENS\n18228\nDENVER BOTANIC\n"
    "cm\nGARDENS\ncopyright reserved\nHerbarium Specimen\nScientific Name Galax aphylla L.\n"
    "Common Name Wandflower, Beetleweed\nLocality High Knob, Virginia\nDate\nCollector\n"
    "Identified by\nRemarks\n8-26-64\nA. Armstrong\nDitto\n#1747\nGalax urceolata (Poir.) Brummitt\n"
    "J. Wingate 7/1997\nGalax aphylla L.\nWandflower, Beetleweed\nDIA\nLocality High Knob, Virginia\n"
    "8-26-1964\nA. Armstrong 1747\nOriginal label\nDENVER BOTANIC GARDENS KHD00041592\n\n\n"
    "Galax aphylla\nGalax urceolata\nGalax aphylla\n"
)

SAMPLE_NOTEBOOK_MD = (
    "# KATHRYN KALMBACH HERBARIUM\n\n"
    "## DENVER BOTANIC GARDENS\n\n"
    "Herbarium Specimen\n\n"
    "- Scientific Name: Galax aphylla L.\n"
    "- Common Name: Wandflower, Beetleweed\n"
    "- Locality: High Knob, Virginia\n"
    "- Date: 8-26-64\n"
    "- Collector: A. Armstrong #1747\n"
    "- Identified by: Ditto\n\n"
    "Galax urceolata (Poir.) Brummitt\n"
    "J. Wingate 7/1997\n\n"
    "DENVER BOTANIC GARDENS KHD00041592\n"
)

SAMPLE_RECORD = {
    "catalogNumber": "KHD00041592",
    "scientificName": "Galax aphylla L.",
    "genus": "Galax",
    "specificEpithet": "aphylla",
    "scientificNameAuthorship": "L.",
    "collectedBy": "A. Armstrong",
    "collectorNumber": "1747",
    "identifiedBy": "Ditto, J. Wingate",
    "identifiedDate": "1997-07-00",
    "identifiedConfidence": "",
    "identifiedRemarks": "",
    "identificationHistory": "Galax urceolata (Poir.) Brummitt, J. Wingate 7/1997",
    "verbatimCollectionDate": "8-26-64",
    "collectionDate": "1964-08-26",
    "collectionDateEnd": "",
    "habitat": "",
    "specimenDescription": "",
    "cultivated": "",
    "continent": "North America",
    "country": "United States",
    "stateProvince": "Virginia",
    "county": "",
    "locality": "High Knob, Virginia",
    "verbatimCoordinates": "",
    "decimalLatitude": "",
    "decimalLongitude": "",
    "minimumElevationInMeters": "",
    "maximumElevationInMeters": "",
    "elevationUnits": "",
    "additionalText": (
        "inches\nKATHRYN KALMBACH HERBARIUM\nDENVER BOTANIC GARDENS\n18228\nDENVER BOTANIC\ncm\n"
        "GARDENS\ncopyright reserved\nHerbarium Specimen\nCommon Name Wandflower, Beetleweed\n"
        "Remarks\nOriginal label\nWandflower, Beetleweed DIA"
    ),
}


class MockEngine:
    model_id = MODEL_ID
    label = LABEL
    capabilities = EngineCapabilities(vision=True, structured_output=True, thinking_levels=())
    notice = DEMO_NOTICE

    def ocr(self, request: OCRRequest) -> OCRResult:
        text = SAMPLE_NOTEBOOK_MD if request.prompt_key == "verbatim_notebook" else SAMPLE_OCR_TEXT
        return OCRResult(text=text, model_id=self.model_id)

    def parse(self, request: ParseRequest) -> ParseResult:
        # Exactly the prompt's fields; a value only where the sample has a
        # field of the same name (case-insensitive), else "".
        sample = {k.lower(): v for k, v in SAMPLE_RECORD.items()}
        record = {name: sample.get(name.lower(), "") for name in request.template}
        return ParseResult(record=record, model_id=self.model_id, raw_text=json.dumps(record, ensure_ascii=False))


def create(ref, context) -> MockEngine:
    return MockEngine()
