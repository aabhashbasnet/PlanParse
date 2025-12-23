import pdfplumber
from collections import defaultdict

# -------------------------------------------------

KEYWORDS = {
    "FLOOR_PLAN": [
        "floor plan",
        "level",
        "ground floor",
        "first floor",
        "second floor",
    ],
    "ELEVATION": [
        "elevation",
        "north elevation",
        "south elevation",
        "east elevation",
        "west elevation",
    ],
    "FOUNDATION": [
        "foundation plan",
        "foundation",
        "footing",
        "slab on grade",
    ],
    "GENERAL_NOTES": [
        "general notes",
        "general note",
        "general construction notes",
        "general information",
        "notes and details",
        "abbreviations",
        "symbols",
        "legend",
        "code requirements",
    ],
    "LEDGER": [
        "sheet index",
        "index of drawings",
        "drawing list",
        "schedule",
    ],
}

# -------------------------------------------------


def extract_words_text(page):
    words = page.extract_words(use_text_flow=True) or []
    return " ".join(w["text"] for w in words).lower()


# -------------------------------------------------


def extract_zones(page):
    """
    Extract text from multiple possible title block zones
    """
    w, h = page.width, page.height

    zones = {
        "bottom": (0, h * 0.75, w, h),
        "bottom_right": (w * 0.6, h * 0.7, w, h),
        "right": (w * 0.75, 0, w, h),
        "left": (0, 0, w * 0.25, h),
    }

    zone_text = {}

    for name, bbox in zones.items():
        area = page.within_bbox(bbox)
        text = area.extract_text() or ""
        zone_text[name] = text.lower()

    return zone_text


# -------------------------------------------------


def classify_page(page):
    zone_text = extract_zones(page)
    full_text = extract_words_text(page)

    scores = defaultdict(int)

    # 1️⃣ Strong signal: title block zones
    for zone in zone_text.values():
        for category, tags in KEYWORDS.items():
            for tag in tags:
                if tag in zone:
                    scores[category] += 3

    # 2️⃣ Medium signal: full-page text
    for category, tags in KEYWORDS.items():
        for tag in tags:
            if tag in full_text:
                scores[category] += 1

    # 3️⃣ Notes pages are text-dense
    if len(full_text) > 2000:
        scores["GENERAL_NOTES"] += 3

    if not scores:
        return "UNKNOWN"

    # Pick best score
    return max(scores, key=scores.get)
