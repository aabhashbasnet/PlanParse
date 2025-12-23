from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
import io
from .classifier import classify_page
from .schemas import ClassificationResponse

app = FastAPI(title="BlueprintLens - Construction Classifier")


@app.post("/classify", response_model=ClassificationResponse)
async def classify_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        pdf_bytes = await file.read()

        results = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                label = classify_page(page)
                results.append({"page_number": i + 1, "classification": label})

            total_pages = len(pdf.pages)

        return {
            "filename": file.filename,
            "total_pages": total_pages,
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
