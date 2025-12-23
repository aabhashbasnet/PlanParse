from pydantic import BaseModel
from typing import List


class PageClassification(BaseModel):
    page_number: int
    classification: str


class ClassificationResponse(BaseModel):
    filename: str
    total_pages: int
    results: List[PageClassification]
