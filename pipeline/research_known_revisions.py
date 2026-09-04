import os
from io import BytesIO

import pymupdf
import requests
from PIL import Image, ImageChops
from supabase import create_client


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SECRET_KEY,
)


# ------------------------------------------------------------
# CONTROLLED READ-ONLY NORMALIZED VISUAL TEST
# ------------------------------------------------------------
#
# LEGO set 10214-1 only.
#
# This test:
# - downloads official LEGO instruction PDFs
# - renders each page as an image
# - crops empty margins
# - normalizes scale and canvas size
# - visually compares old/new instruction pages
#
# NOTHING is written to Supabase.
# ------------------------------------------------------------

TEST_SET_NUM = "10214-1"


TEST_PAIRS = [
    {
        "label": "Booklet 2/3 — 2010 vs 2011",
        "older_document": "4611616",
        "newer_document": "4658001",
    },
    {
        "label": "Booklet 3/3 — 2010 vs 2011",
        "older_document": "4611753",
        "newer_document": "4658002",
    },
    {
        "label": "Booklet 2/3 — 2011 vs 2015",
        "older_document": "4658001",
        "newer_document": "6145768",
    },
    {
        "label": "Booklet 3/3 — 2011 vs 2015",
        "older_document": "4658002",
        "newer_document": "6145770",
    },
    {
        "label": "Booklet 1/3 — 2012 vs 2015",
        "older_document": "6020850",
        "newer_document": "6146167",
    },
]


# ------------------------------------------------------------
# IMAGE SETTINGS
# ------------------------------------------------------------

# Render quality.
RENDER_SCALE = 1.0

# All cropped pages are fitted inside this square canvas.
NORMALIZED_SIZE = 512

# Pixels lighter than this are treated as effectively white
# when finding the visible printed area.
WHITE_THRESHOLD = 245

# A small amount of space retained around detected page content.
CROP_PADDING = 8

# Difference of at least this many grayscale levels counts
# as a visibly changed pixel.
PIXEL_CHANGE_THRESHOLD = 20


# ------------------------------------------------------------
# CLASSIFICATION THRESHOLDS
# ------------------------------------------------------------

# Mean grayscale difference divided by 255.

NEAR_IDENTICAL_SCORE = 0.002
SMALL_CHANGE_SCORE = 0.015
MODERATE_CHANGE_SCORE = 0.050


def get_instruction_documents():
    """
    Fetch metadata only for the documents needed by this test.
    """
    required_numbers = set()

    for pair in TEST_PAIRS:
        required_numbers.add(
            pair["older_document"]
        )

        required_numbers.add(
            pair["newer_document"]
        )

    response = (
        supabase
        .table("instruction_documents")
        .select(
            "document_number,"
            "description,"
            "source_url,"
            "source_date_added,"
            "source_date_modified"
        )
        .eq(
            "set_num",
            TEST_SET_NUM,
        )
        .in_(
            "document_number",
            sorted(required_numbers),
        )
        .execute()
    )

    documents = {}

    for document in response.data or []:
        number = str(
            document.get(
                "document_number"
            )
        )

        documents[number] = document

    return documents


def download_pdf(document):
    """
    Download one official LEGO instruction PDF.
    """
    document_number = str(
        document.get(
            "document_number"
        )
    )

    source_url = document.get(
        "source_url"
    )

    if not source_url:
        raise RuntimeError(
            f"Document {document_number} "
            f"has no source_url."
        )

    print(
        f"Downloading {document_number}..."
    )

    response = requests.get(
        source_url,
        timeout=180,
    )

    response.raise_for_status()

    pdf_bytes = response.content

    if not pdf_bytes.startswith(
        b"%PDF"
    ):
        raise RuntimeError(
            f"Document {document_number} "
            f"did not download as a PDF."
        )

    print(
        f"  Downloaded "
        f"{len(pdf_bytes):,} bytes"
    )

    return pdf_bytes


def open_pdf(pdf_bytes):
    """
    Open PDF bytes with PyMuPDF.
    """
    return pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )


def render_page(page):
    """
    Render one PDF page as a grayscale Pillow image.
    """
    matrix = pymupdf.Matrix(
        RENDER_SCALE,
        RENDER_SCALE,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )

    image = Image.frombytes(
        "L",
        (
            pixmap.width,
            pixmap.height,
        ),
        pixmap.samples,
    )

    return image


def find_content_bbox(image):
    """
    Find the visible printed region of a grayscale page.

    White page margins are ignored.
    """
    threshold_image = image.point(
        lambda value:
        0
        if value >= WHITE_THRESHOLD
        else 255
    )

    bbox = threshold_image.getbbox()

    if bbox is None:
        return (
            0,
            0,
            image.width,
            image.height,
        )

    left, top, right, bottom = bbox

    left = max(
        0,
        left - CROP_PADDING,
    )

    top = max(
        0,
        top - CROP_PADDING,
    )

    right = min(
        image.width,
        right + CROP_PADDING,
    )

    bottom = min(
        image.height,
        bottom + CROP_PADDING,
    )

    return (
        left,
        top,
        right,
        bottom,
    )


def crop_content(image):
    """
    Remove excess white margin around page content.
    """
    bbox = find_content_bbox(
        image
    )

    return image.crop(
        bbox
    )


def normalize_page(image):
    """
    Normalize a rendered LEGO instruction page.

    Process:
    1. crop mostly-empty margins
    2. preserve aspect ratio
    3. resize printed content
    4. center it on a fixed white canvas

    This reduces false positives caused by different PDF page
    dimensions, margins, and export scaling.
    """
    cropped = crop_content(
        image
    )

    if (
        cropped.width <= 0
        or cropped.height <= 0
    ):
        return Image.new(
            "L",
            (
                NORMALIZED_SIZE,
                NORMALIZED_SIZE,
            ),
            255,
        )

    maximum_content_size = (
        NORMALIZED_SIZE - 20
    )

    scale = min(
        maximum_content_size
        / cropped.width,
        maximum_content_size
        / cropped.height,
    )

    new_width = max(
        1,
        round(
            cropped.width * scale
        ),
    )

    new_height = max(
        1,
        round(
            cropped.height * scale
        ),
    )

    resized = cropped.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "L",
        (
            NORMALIZED_SIZE,
            NORMALIZED_SIZE,
        ),
        255,
    )

    x = (
        NORMALIZED_SIZE
        - new_width
    ) // 2

    y = (
        NORMALIZED_SIZE
        - new_height
    ) // 2

    canvas.paste(
        resized,
        (
            x,
            y,
        ),
    )

    return canvas


def compare_images(
    older_image,
    newer_image,
):
    """
    Compare two normalized grayscale images using histogram math.

    This avoids slowly looping through every pixel in Python.
    """
    difference_image = (
        ImageChops.difference(
            older_image,
            newer_image,
        )
    )

    histogram = (
        difference_image.histogram()
    )

    total_pixels = sum(
        histogram
    )

    if total_pixels == 0:
        return {
            "difference_score": 0.0,
            "changed_pixel_percent": 0.0,
            "maximum_difference": 0,
        }

    weighted_difference = 0

    for pixel_value, count in enumerate(
        histogram
    ):
        weighted_difference += (
            pixel_value * count
        )

    mean_difference = (
        weighted_difference
        / total_pixels
    )

    difference_score = (
        mean_difference
        / 255.0
    )

    visibly_changed_pixels = sum(
        histogram[
            PIXEL_CHANGE_THRESHOLD:
        ]
    )

    changed_pixel_percent = (
        visibly_changed_pixels
        / total_pixels
        * 100.0
    )

    maximum_difference = 0

    for pixel_value in range(
        255,
        -1,
        -1,
    ):
        if histogram[pixel_value]:
            maximum_difference = (
                pixel_value
            )
            break

    return {
        "difference_score":
            difference_score,
        "changed_pixel_percent":
            changed_pixel_percent,
        "maximum_difference":
            maximum_difference,
    }


def classify_page(result):
    """
    Convert a normalized visual difference score into a label.
    """
    score = result[
        "difference_score"
    ]

    if score == 0:
        return "IDENTICAL"

    if score < NEAR_IDENTICAL_SCORE:
        return "NEAR-IDENTICAL"

    if score < SMALL_CHANGE_SCORE:
        return "SMALL CHANGE"

    if score < MODERATE_CHANGE_SCORE:
        return "MODERATE CHANGE"

    return "SUBSTANTIAL"


def compare_page(
    older_page,
    newer_page,
):
    """
    Render, normalize, and compare one page pair.
    """
    older_render = render_page(
        older_page
    )

    newer_render = render_page(
        newer_page
    )

    older_normalized = normalize_page(
        older_render
    )

    newer_normalized = normalize_page(
        newer_render
    )

    result = compare_images(
        older_normalized,
        newer_normalized,
    )

    result[
        "classification"
    ] = classify_page(
        result
    )

    return result


def summarize_pages(
    pages,
    maximum_display=40,
):
    """
    Keep GitHub logs manageable.
    """
    if not pages:
        return "none"

    if len(pages) <= maximum_display:
        return ", ".join(
            str(page)
            for page in pages
        )

    visible = pages[
        :maximum_display
    ]

    visible_text = ", ".join(
        str(page)
        for page in visible
    )

    remaining = (
        len(pages)
        - maximum_display
    )

    return (
        f"{visible_text}, "
        f"... +{remaining} more"
    )


def compare_pair(
    pair,
    documents,
    pdf_cache,
):
    """
    Compare one old/new instruction booklet pair.
    """
    older_number = pair[
        "older_document"
    ]

    newer_number = pair[
        "newer_document"
    ]

    print("")
    print("=" * 72)
    print(
        pair["label"]
    )
    print(
        f"{older_number} -> "
        f"{newer_number}"
    )
    print("=" * 72)

    older_document = documents.get(
        older_number
    )

    newer_document = documents.get(
        newer_number
    )

    if not older_document:
        raise RuntimeError(
            f"Missing metadata for "
            f"{older_number}."
        )

    if not newer_document:
        raise RuntimeError(
            f"Missing metadata for "
            f"{newer_number}."
        )

    print(
        "Older description: "
        + str(
            older_document.get(
                "description"
            )
        )
    )

    print(
        "Newer description: "
        + str(
            newer_document.get(
                "description"
            )
        )
    )

    if older_number not in pdf_cache:
        pdf_cache[
            older_number
        ] = download_pdf(
            older_document
        )

    if newer_number not in pdf_cache:
        pdf_cache[
            newer_number
        ] = download_pdf(
            newer_document
        )

    older_pdf = open_pdf(
        pdf_cache[
            older_number
        ]
    )

    newer_pdf = open_pdf(
        pdf_cache[
            newer_number
        ]
    )

    try:
        older_page_count = len(
            older_pdf
        )

        newer_page_count = len(
            newer_pdf
        )

        shared_page_count = min(
            older_page_count,
            newer_page_count,
        )

        print("")
        print(
            f"Older page count: "
            f"{older_page_count}"
        )

        print(
            f"Newer page count: "
            f"{newer_page_count}"
        )

        print(
            f"Shared pages: "
            f"{shared_page_count}"
        )

        identical_pages = []
        near_identical_pages = []
        small_change_pages = []
        moderate_change_pages = []
        substantial_pages = []

        page_results = {}

        print("")
        print(
            "Rendering and normalizing pages..."
        )

        for index in range(
            shared_page_count
        ):
            page_number = (
                index + 1
            )

            result = compare_page(
                older_pdf[index],
                newer_pdf[index],
            )

            page_results[
                page_number
            ] = result

            classification = result[
                "classification"
            ]

            if classification == "IDENTICAL":
                identical_pages.append(
                    page_number
                )

            elif classification == "NEAR-IDENTICAL":
                near_identical_pages.append(
                    page_number
                )

            elif classification == "SMALL CHANGE":
                small_change_pages.append(
                    page_number
                )

            elif classification == "MODERATE CHANGE":
                moderate_change_pages.append(
                    page_number
                )

            else:
                substantial_pages.append(
                    page_number
                )

        older_only_pages = list(
            range(
                shared_page_count + 1,
                older_page_count + 1,
            )
        )

        newer_only_pages = list(
            range(
                shared_page_count + 1,
                newer_page_count + 1,
            )
        )

        print("")
        print(
            "NORMALIZED VISUAL RESULTS"
        )
        print(
            "-------------------------"
        )

        print(
            f"Identical pages: "
            f"{len(identical_pages)}"
        )

        print(
            f"Near-identical pages: "
            f"{len(near_identical_pages)}"
        )

        print(
            f"Small-change pages: "
            f"{len(small_change_pages)}"
        )

        print(
            f"Moderate-change pages: "
            f"{len(moderate_change_pages)}"
        )

        print(
            f"Substantial-change pages: "
            f"{len(substantial_pages)}"
        )

        print("")
        print(
            "Near-identical: "
            + summarize_pages(
                near_identical_pages
            )
        )

        print(
            "Small changes: "
            + summarize_pages(
                small_change_pages
            )
        )

        print(
            "Moderate changes: "
            + summarize_pages(
                moderate_change_pages
            )
        )

        print(
            "Substantial changes: "
            + summarize_pages(
                substantial_pages
            )
        )

        print(
            "Older-only pages: "
            + summarize_pages(
                older_only_pages
            )
        )

        print(
            "Newer-only pages: "
            + summarize_pages(
                newer_only_pages
            )
        )

        changed_pages = (
            near_identical_pages
            + small_change_pages
            + moderate_change_pages
            + substantial_pages
        )

        if changed_pages:
            print("")
            print(
                "Changed-page detail:"
            )

            for page_number in sorted(
                changed_pages
            ):
                result = page_results[
                    page_number
                ]

                print(
                    f"  Page "
                    f"{page_number}: "
                    f'{result["classification"]} | '
                    f'difference score '
                    f'{result["difference_score"]:.6f} | '
                    f'changed pixels '
                    f'{result["changed_pixel_percent"]:.3f}% | '
                    f'max pixel delta '
                    f'{result["maximum_difference"]}'
                )

        meaningful_pages = (
            moderate_change_pages
            + substantial_pages
        )

        print("")
        print(
            "PAIR ASSESSMENT:"
        )

        if (
            not small_change_pages
            and not meaningful_pages
            and not older_only_pages
            and not newer_only_pages
        ):
            print(
                "  LIKELY SAME BUILD INSTRUCTIONS"
            )

            print(
                "  Only visually negligible "
                "differences remain after normalization."
            )

        elif (
            not meaningful_pages
            and not older_only_pages
            and not newer_only_pages
        ):
            print(
                "  LIKELY COSMETIC / PRINTING CHANGE"
            )

            print(
                "  Minor visual differences remain, "
                "but no meaningful page-level change "
                "was detected."
            )

        elif (
            len(meaningful_pages) <= 2
            and not older_only_pages
            and not newer_only_pages
        ):
            print(
                "  LOCALIZED VISUAL CHANGE"
            )

            print(
                "  Only a small number of pages changed "
                "meaningfully. These pages should be "
                "inspected before calling this a build revision."
            )

        else:
            print(
                "  STRONG REVISION CANDIDATE"
            )

            print(
                "  Multiple instruction pages remain "
                "meaningfully different even after "
                "normalizing margins and scale."
            )

    finally:
        older_pdf.close()
        newer_pdf.close()


def main():
    print("")
    print(
        "BrickTrip Normalized LEGO PDF Comparison"
    )
    print(
        "========================================"
    )

    print(
        f"Controlled test set: "
        f"{TEST_SET_NUM}"
    )

    print(
        "READ ONLY — database writes: NONE"
    )

    documents = (
        get_instruction_documents()
    )

    print("")
    print(
        f"Required instruction documents found: "
        f"{len(documents)}"
    )

    pdf_cache = {}

    succeeded = 0
    failed = 0

    for pair in TEST_PAIRS:
        try:
            compare_pair(
                pair,
                documents,
                pdf_cache,
            )

            succeeded += 1

        except Exception as error:
            failed += 1

            print("")
            print(
                f"ERROR comparing "
                f'{pair["label"]}: '
                f"{error}"
            )

    print("")
    print("=" * 72)
    print(
        "NORMALIZED PDF TEST COMPLETE"
    )

    print(
        f"Comparisons succeeded: "
        f"{succeeded}"
    )

    print(
        f"Comparisons failed: "
        f"{failed}"
    )

    print(
        "Database writes: 0"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
