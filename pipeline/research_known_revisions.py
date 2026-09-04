import hashlib
import os

import fitz
import requests
from supabase import create_client


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SECRET_KEY,
)


# ------------------------------------------------------------
# CONTROLLED READ-ONLY VISUAL PDF TEST
# ------------------------------------------------------------
#
# LEGO set 10214-1 only.
#
# This test:
# - downloads official LEGO PDFs
# - renders pages visually
# - compares rendered page images
# - distinguishes tiny cosmetic changes from substantial ones
#
# It DOES NOT:
# - insert evidence
# - update revision_candidates
# - create revisions
# - modify Supabase in any way
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
# VISUAL COMPARISON SETTINGS
# ------------------------------------------------------------

# First pass uses a small grayscale rendering.
# This makes comparing hundreds of pages fast.
THUMBNAIL_SCALE = 0.30

# If the thumbnail says a page changed, we render that page again
# at normal PDF resolution for a more accurate measurement.
DETAIL_SCALE = 1.0

# Pixel difference of 20/255 or greater counts as visibly changed.
PIXEL_CHANGE_THRESHOLD = 20

# Classification thresholds.
#
# These are intentionally conservative for this diagnostic test.
NEAR_IDENTICAL_SCORE = 0.002
SMALL_CHANGE_SCORE = 0.020


def get_instruction_documents():
    """
    Fetch metadata for every document required by this test.
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
        .eq("set_num", TEST_SET_NUM)
        .in_(
            "document_number",
            sorted(required_numbers),
        )
        .execute()
    )

    documents = {}

    for document in response.data or []:
        number = str(
            document.get("document_number")
        )

        documents[number] = document

    return documents


def download_pdf(document):
    """
    Download one official LEGO PDF.
    """
    document_number = str(
        document.get("document_number")
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

    if not pdf_bytes.startswith(b"%PDF"):
        raise RuntimeError(
            f"Document {document_number} "
            f"did not download as a PDF."
        )

    print(
        f"  Downloaded "
        f"{len(pdf_bytes):,} bytes"
    )

    return pdf_bytes


def file_hash(pdf_bytes):
    """
    Whole-file SHA256 hash.
    """
    return hashlib.sha256(
        pdf_bytes
    ).hexdigest()


def open_pdf(pdf_bytes):
    """
    Open PDF bytes with PyMuPDF.
    """
    return fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )


def render_page(page, scale):
    """
    Render one PDF page into a grayscale image.

    Returns:
        width
        height
        raw grayscale pixel bytes
    """
    matrix = fitz.Matrix(
        scale,
        scale,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        colorspace=fitz.csGRAY,
        alpha=False,
    )

    return {
        "width": pixmap.width,
        "height": pixmap.height,
        "samples": pixmap.samples,
    }


def compare_rendered_images(
    older_image,
    newer_image,
):
    """
    Compare two grayscale rendered page images.

    Returns:
    - mean absolute pixel difference
    - normalized difference score
    - percentage of pixels exceeding our visible-change threshold
    """
    if (
        older_image["width"]
        != newer_image["width"]
        or older_image["height"]
        != newer_image["height"]
    ):
        return {
            "same_dimensions": False,
            "difference_score": 1.0,
            "changed_pixel_percent": 100.0,
        }

    older_pixels = older_image[
        "samples"
    ]

    newer_pixels = newer_image[
        "samples"
    ]

    if older_pixels == newer_pixels:
        return {
            "same_dimensions": True,
            "difference_score": 0.0,
            "changed_pixel_percent": 0.0,
        }

    total_pixels = len(
        older_pixels
    )

    if total_pixels == 0:
        return {
            "same_dimensions": True,
            "difference_score": 0.0,
            "changed_pixel_percent": 0.0,
        }

    absolute_difference_sum = 0
    visibly_changed_pixels = 0

    for older_value, newer_value in zip(
        older_pixels,
        newer_pixels,
    ):
        difference = abs(
            older_value
            - newer_value
        )

        absolute_difference_sum += (
            difference
        )

        if (
            difference
            >= PIXEL_CHANGE_THRESHOLD
        ):
            visibly_changed_pixels += 1

    mean_difference = (
        absolute_difference_sum
        / total_pixels
    )

    difference_score = (
        mean_difference
        / 255.0
    )

    changed_pixel_percent = (
        visibly_changed_pixels
        / total_pixels
        * 100.0
    )

    return {
        "same_dimensions": True,
        "difference_score":
            difference_score,
        "changed_pixel_percent":
            changed_pixel_percent,
    }


def classify_page(comparison):
    """
    Convert the numerical visual score into a human-readable label.
    """
    if not comparison[
        "same_dimensions"
    ]:
        return "SUBSTANTIAL"

    score = comparison[
        "difference_score"
    ]

    if score == 0:
        return "IDENTICAL"

    if score < NEAR_IDENTICAL_SCORE:
        return "NEAR-IDENTICAL"

    if score < SMALL_CHANGE_SCORE:
        return "SMALL CHANGE"

    return "SUBSTANTIAL"


def compare_page(
    older_page,
    newer_page,
):
    """
    Efficient two-stage visual comparison.

    Stage 1:
        low-resolution thumbnail

    Stage 2:
        full-resolution rendering only when needed
    """
    older_thumbnail = render_page(
        older_page,
        THUMBNAIL_SCALE,
    )

    newer_thumbnail = render_page(
        newer_page,
        THUMBNAIL_SCALE,
    )

    thumbnail_comparison = (
        compare_rendered_images(
            older_thumbnail,
            newer_thumbnail,
        )
    )

    thumbnail_classification = (
        classify_page(
            thumbnail_comparison
        )
    )

    if (
        thumbnail_classification
        == "IDENTICAL"
    ):
        return {
            "classification":
                "IDENTICAL",
            "difference_score":
                0.0,
            "changed_pixel_percent":
                0.0,
        }

    older_detail = render_page(
        older_page,
        DETAIL_SCALE,
    )

    newer_detail = render_page(
        newer_page,
        DETAIL_SCALE,
    )

    detail_comparison = (
        compare_rendered_images(
            older_detail,
            newer_detail,
        )
    )

    classification = (
        classify_page(
            detail_comparison
        )
    )

    return {
        "classification":
            classification,
        "difference_score":
            detail_comparison[
                "difference_score"
            ],
        "changed_pixel_percent":
            detail_comparison[
                "changed_pixel_percent"
            ],
    }


def summarize_page_numbers(
    pages,
    maximum_display=40,
):
    """
    Keep GitHub logs readable.
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

    text = ", ".join(
        str(page)
        for page in visible
    )

    remaining = (
        len(pages)
        - maximum_display
    )

    return (
        f"{text}, "
        f"... +{remaining} more"
    )


def compare_pair(
    pair,
    documents,
    pdf_byte_cache,
):
    """
    Compare one old/new LEGO instruction pair visually.
    """
    older_number = pair[
        "older_document"
    ]

    newer_number = pair[
        "newer_document"
    ]

    print("")
    print("=" * 72)
    print(pair["label"])
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

    if older_number not in pdf_byte_cache:
        pdf_byte_cache[
            older_number
        ] = download_pdf(
            older_document
        )

    if newer_number not in pdf_byte_cache:
        pdf_byte_cache[
            newer_number
        ] = download_pdf(
            newer_document
        )

    older_bytes = pdf_byte_cache[
        older_number
    ]

    newer_bytes = pdf_byte_cache[
        newer_number
    ]

    same_file = (
        file_hash(
            older_bytes
        )
        == file_hash(
            newer_bytes
        )
    )

    older_pdf = open_pdf(
        older_bytes
    )

    newer_pdf = open_pdf(
        newer_bytes
    )

    try:
        older_page_count = len(
            older_pdf
        )

        newer_page_count = len(
            newer_pdf
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
            f"Whole PDFs byte-identical: "
            f"{same_file}"
        )

        shared_page_count = min(
            older_page_count,
            newer_page_count,
        )

        identical_pages = []
        near_identical_pages = []
        small_change_pages = []
        substantial_pages = []

        page_results = {}

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
            f"Visually identical pages: "
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
            f"Substantially changed pages: "
            f"{len(substantial_pages)}"
        )

        print("")
        print(
            "Near-identical page numbers: "
            + summarize_page_numbers(
                near_identical_pages
            )
        )

        print(
            "Small-change page numbers: "
            + summarize_page_numbers(
                small_change_pages
            )
        )

        print(
            "Substantial-change page numbers: "
            + summarize_page_numbers(
                substantial_pages
            )
        )

        print(
            "Pages only in older PDF: "
            + summarize_page_numbers(
                older_only_pages
            )
        )

        print(
            "Pages only in newer PDF: "
            + summarize_page_numbers(
                newer_only_pages
            )
        )

        changed_pages = (
            near_identical_pages
            + small_change_pages
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
                    f"  Page {page_number}: "
                    f'{result["classification"]} | '
                    f'difference score '
                    f'{result["difference_score"]:.6f} | '
                    f'changed pixels '
                    f'{result["changed_pixel_percent"]:.3f}%'
                )

        print("")
        print("PAIR ASSESSMENT:")

        if (
            not small_change_pages
            and not substantial_pages
            and not older_only_pages
            and not newer_only_pages
        ):
            print(
                "  LIKELY SAME BUILD INSTRUCTIONS"
            )

            print(
                "  Differences appear absent or "
                "visually negligible."
            )

        elif (
            not substantial_pages
            and not older_only_pages
            and not newer_only_pages
        ):
            print(
                "  POSSIBLE COSMETIC / PRINTING CHANGE"
            )

            print(
                "  No page shows a substantial "
                "visual difference."
            )

        else:
            print(
                "  MEANINGFUL VISUAL DIFFERENCE FOUND"
            )

            print(
                "  Candidate for deeper LEGO "
                "step/inventory analysis."
            )

    finally:
        older_pdf.close()
        newer_pdf.close()


def main():
    print("")
    print(
        "BrickTrip Visual LEGO PDF Comparison"
    )
    print(
        "===================================="
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

    pdf_byte_cache = {}

    succeeded = 0
    failed = 0

    for pair in TEST_PAIRS:
        try:
            compare_pair(
                pair,
                documents,
                pdf_byte_cache,
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
        "VISUAL PDF TEST COMPLETE"
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
