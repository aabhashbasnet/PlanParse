from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import pytesseract
from PIL import Image
import pdf2image
import io
import re
from enum import Enum

app = FastAPI(title="Construction Drawing Classifier API")


class DrawingType(str, Enum):
    COVER = "cover_page"
    PROJECT_INFO = "project_info"
    SITE_PLAN = "site_plan"
    FLOOR_PLAN = "floor_plan"
    ELEVATION = "elevation"
    SECTION = "section"
    ROOF_PLAN = "roof_plan"
    FOUNDATION = "foundation"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    MECHANICAL = "mechanical"
    STRUCTURAL = "structural"
    DETAIL = "detail"
    SCHEDULE = "schedule"
    UNKNOWN = "unknown"


class ClassificationResult(BaseModel):
    page_number: int
    classification: DrawingType
    confidence: float
    keywords_found: List[str]
    drawing_number: Optional[str] = None


class DrawingClassifier:
    def __init__(self):
        # Keywords for each drawing type
        self.keywords = {
            DrawingType.COVER: [
                "cover",
                "title sheet",
                "project title",
                "drawing index",
                "sheet index",
                "consultants",
                "project team",
            ],
            DrawingType.PROJECT_INFO: [
                "general notes",
                "legend",
                "abbreviations",
                "symbols",
                "project information",
                "code analysis",
                "vicinity map",
            ],
            DrawingType.SITE_PLAN: [
                "site plan",
                "plot plan",
                "survey",
                "property line",
                "setback",
                "parking layout",
                "landscape",
            ],
            DrawingType.FLOOR_PLAN: [
                "floor plan",
                "layout",
                "room schedule",
                "door schedule",
                "window schedule",
                "partition",
                "furniture plan",
            ],
            DrawingType.ELEVATION: [
                "elevation",
                "exterior elevation",
                "building elevation",
                "north elevation",
                "south elevation",
                "east elevation",
                "west elevation",
            ],
            DrawingType.SECTION: [
                "section",
                "building section",
                "wall section",
                "detail section",
                "cross section",
            ],
            DrawingType.ROOF_PLAN: [
                "roof plan",
                "roof framing",
                "roof layout",
                "roofing plan",
            ],
            DrawingType.FOUNDATION: [
                "foundation",
                "footing",
                "basement",
                "slab",
                "pier",
                "foundation plan",
                "grading plan",
            ],
            DrawingType.ELECTRICAL: [
                "electrical",
                "power plan",
                "lighting",
                "panel schedule",
                "circuit",
                "receptacle",
                "switch",
            ],
            DrawingType.PLUMBING: [
                "plumbing",
                "sanitary",
                "water",
                "drainage",
                "fixture",
                "domestic water",
                "waste",
            ],
            DrawingType.MECHANICAL: [
                "mechanical",
                "hvac",
                "duct",
                "ventilation",
                "air conditioning",
                "heating",
                "exhaust",
            ],
            DrawingType.STRUCTURAL: [
                "structural",
                "framing",
                "beam",
                "column",
                "truss",
                "steel",
                "concrete",
                "reinforcing",
            ],
            DrawingType.DETAIL: [
                "detail",
                "enlarged",
                "typical detail",
                "construction detail",
            ],
            DrawingType.SCHEDULE: [
                "schedule",
                "finish schedule",
                "door schedule",
                "window schedule",
                "room finish",
            ],
        }

        # Drawing number prefixes (standard AIA convention)
        self.drawing_prefixes = {
            "A": DrawingType.FLOOR_PLAN,  # Architectural
            "S": DrawingType.STRUCTURAL,
            "E": DrawingType.ELECTRICAL,
            "M": DrawingType.MECHANICAL,
            "P": DrawingType.PLUMBING,
            "L": DrawingType.SITE_PLAN,  # Landscape
            "C": DrawingType.SITE_PLAN,  # Civil
        }

    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        try:
            # Focus on title block area (bottom 20% of page)
            width, height = image.size
            title_block = image.crop((0, int(height * 0.8), width, height))

            # Extract text from full page and title block
            full_text = pytesseract.image_to_string(image).lower()
            title_text = pytesseract.image_to_string(title_block).lower()

            # Combine with more weight on title block
            return full_text + "\n" + title_text * 2
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

    def extract_drawing_number(self, text: str) -> Optional[str]:
        """Extract drawing number (e.g., A-101, E-201, S-301)"""
        patterns = [
            r"[A-Z]-\d{3}",  # A-101
            r"[A-Z]\d{3}",  # A101
            r"[A-Z]-\d\.\d",  # A-1.1
        ]

        for pattern in patterns:
            match = re.search(pattern, text.upper())
            if match:
                return match.group(0)
        return None

    def classify_by_keywords(self, text: str) -> tuple[DrawingType, float, List[str]]:
        """Classify based on keyword matching"""
        scores = {}
        found_keywords = {}

        for drawing_type, keywords in self.keywords.items():
            score = 0
            found = []
            for keyword in keywords:
                if keyword in text:
                    score += text.count(keyword)
                    found.append(keyword)
            scores[drawing_type] = score
            found_keywords[drawing_type] = found

        if not any(scores.values()):
            return DrawingType.UNKNOWN, 0.0, []

        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        confidence = min(max_score / 5.0, 1.0)  # Normalize to 0-1

        return best_type, confidence, found_keywords[best_type]

    def classify_by_drawing_number(self, drawing_num: str) -> Optional[DrawingType]:
        """Classify based on drawing number prefix"""
        if drawing_num and len(drawing_num) > 0:
            prefix = drawing_num[0].upper()
            return self.drawing_prefixes.get(prefix)
        return None

    def classify_page(self, image: Image.Image, page_num: int) -> ClassificationResult:
        """Main classification logic"""
        # Extract text
        text = self.extract_text_from_image(image)

        # Extract drawing number
        drawing_num = self.extract_drawing_number(text)

        # Classify by keywords
        keyword_type, keyword_confidence, keywords = self.classify_by_keywords(text)

        # Classify by drawing number
        prefix_type = (
            self.classify_by_drawing_number(drawing_num) if drawing_num else None
        )

        # Combine results (prefer prefix if high confidence, otherwise use keywords)
        if prefix_type and keyword_confidence < 0.5:
            final_type = prefix_type
            final_confidence = 0.7
        elif prefix_type == keyword_type:
            final_type = keyword_type
            final_confidence = min(keyword_confidence + 0.2, 1.0)
        else:
            final_type = keyword_type
            final_confidence = keyword_confidence

        # Special case: first page is likely cover
        if page_num == 1 and "cover" in text:
            final_type = DrawingType.COVER
            final_confidence = 0.9

        return ClassificationResult(
            page_number=page_num,
            classification=final_type,
            confidence=final_confidence,
            keywords_found=keywords[:5],  # Top 5 keywords
            drawing_number=drawing_num,
        )


classifier = DrawingClassifier()


@app.post("/classify/image", response_model=ClassificationResult)
async def classify_image(file: UploadFile = File(...)):
    """Classify a single construction drawing image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = classifier.classify_page(image, page_num=1)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/classify/pdf", response_model=List[ClassificationResult])
async def classify_pdf(file: UploadFile = File(...)):
    """Classify all pages in a PDF construction set"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        contents = await file.read()

        # Convert PDF to images
        images = pdf2image.convert_from_bytes(contents, dpi=150)

        # Classify each page
        results = []
        for i, image in enumerate(images, start=1):
            result = classifier.classify_page(image, page_num=i)
            results.append(result)

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


@app.get("/drawing-types")
async def get_drawing_types():
    """Get list of all supported drawing types"""
    return {
        "drawing_types": [dt.value for dt in DrawingType],
        "description": "Supported construction drawing classifications",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "construction-drawing-classifier"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
