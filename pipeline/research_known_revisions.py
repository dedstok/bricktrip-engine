import os
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import requests
from supabase import create_client


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SECRET_KEY,
)


# ------------------------------------------------------------
# READ-ONLY STRUCTURAL COMPARISON TEST
# ------------------------------------------------------------
#
# Negative control:
#   10214-1
#   Same build / later visual refresh
#
# Positive control:
#   42129-1
#   Known real production redesign
#
# Nothing is written to Supabase.
# ------------------------------------------------------------


OUTPUT_DIR = Path(
    "artifacts/bricktrip_pdf_comparisons"
)


CONTROLS = [
    {
        "name": "10214_negative_control",
        "set_num": "10214-1",
        "label": "10214 same-build visual refresh",
        "older_document": "4658001",
        "newer_document": "6145768",
    },
    {
        "name": "42129_positive_control",
        "set_num": "42129-1",
        "label": "42129 known production redesign",
        "older_document": "6396686",
        "newer_document": "6411661",
    },
]


# ------------------------------------------------------------
# IMAGE SETTINGS
# ------------------------------------------------------------

RENDER_SCALE = 0.45

NORMALIZED_SIZE = 420

WHITE_THRESHOLD = 245

CROP_PADDING = 8

CANNY_LOW = 50

CANNY_HIGH = 140

# Small dilation makes the comparison tolerant of tiny shifts
# caused by slightly different PDF rendering.
EDGE_DILATION_SIZE = 3

# Save the worst-scoring pages for visual inspection.
LOWEST_PAGES_TO_SAVE = 10


def get_document(set_num, document_number):
    """
    Fetch one known instruction document.
    """
    response = (
        supabase
        .table("instruction_documents")
        .select(
            "set_num,"
            "document_number,"
            "description,"
            "source_url,"
            "source_date_added,"
            "source_date_modified"
        )
        .eq(
            "set_num",
            set_num,
        )
        .eq(
            "document_number",
            document_number,
        )
        .limit(1)
        .execute()
    )

    rows = response.data or []

    if not rows:
        raise RuntimeError(
            f"Instruction document "
            f"{document_number} "
            f"for {set_num} was not found."
        )

    return rows[0]


def download_pdf(document):
    """
    Download one official LEGO instruction PDF.
    """
    document_number = str(
        document["document_number"]
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
        timeout=240,
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
    Render a PDF page to grayscale numpy array.
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

    image = np.frombuffer(
        pixmap.samples,
        dtype=np.uint8,
    )

    image = image.reshape(
        pixmap.height,
        pixmap.width,
    )

    return image


def find_content_bbox(image):
    """
    Find the printed content area while ignoring white margins.
    """
    mask = (
        image < WHITE_THRESHOLD
    ).astype(
        np.uint8
    )

    coordinates = cv2.findNonZero(
        mask
    )

    if coordinates is None:
        return (
            0,
            0,
            image.shape[1],
            image.shape[0],
        )

    x, y, width, height = cv2.boundingRect(
        coordinates
    )

    left = max(
        0,
        x - CROP_PADDING,
    )

    top = max(
        0,
        y - CROP_PADDING,
    )

    right = min(
        image.shape[1],
        x + width + CROP_PADDING,
    )

    bottom = min(
        image.shape[0],
        y + height + CROP_PADDING,
    )

    return (
        left,
        top,
        right,
        bottom,
    )


def normalize_page(image):
    """
    Crop margins and fit the content into a fixed-size canvas.
    """
    left, top, right, bottom = (
        find_content_bbox(
            image
        )
    )

    cropped = image[
        top:bottom,
        left:right,
    ]

    if cropped.size == 0:
        return np.full(
            (
                NORMALIZED_SIZE,
                NORMALIZED_SIZE,
            ),
            255,
            dtype=np.uint8,
        )

    available_size = (
        NORMALIZED_SIZE - 20
    )

    height, width = (
        cropped.shape
    )

    scale = min(
        available_size / width,
        available_size / height,
    )

    new_width = max(
        1,
        round(
            width * scale
        ),
    )

    new_height = max(
        1,
        round(
            height * scale
        ),
    )

    resized = cv2.resize(
        cropped,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (
            NORMALIZED_SIZE,
            NORMALIZED_SIZE,
        ),
        255,
        dtype=np.uint8,
    )

    x = (
        NORMALIZED_SIZE
        - new_width
    ) // 2

    y = (
        NORMALIZED_SIZE
        - new_height
    ) // 2

    canvas[
        y:y + new_height,
        x:x + new_width,
    ] = resized

    return canvas


def create_edge_map(image):
    """
    Convert normalized page into a structural edge map.

    This intentionally ignores LEGO's color palette as much as
    possible and focuses on boundaries, lines, and geometry.
    """
    blurred = cv2.GaussianBlur(
        image,
        (
            3,
            3,
        ),
        0,
    )

    edges = cv2.Canny(
        blurred,
        CANNY_LOW,
        CANNY_HIGH,
    )

    kernel = np.ones(
        (
            EDGE_DILATION_SIZE,
            EDGE_DILATION_SIZE,
        ),
        dtype=np.uint8,
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1,
    )

    return edges


def dice_similarity(
    older_edges,
    newer_edges,
):
    """
    Compare two binary structural edge maps.

    1.0 = structurally identical
    0.0 = no structural overlap
    """
    older_mask = (
        older_edges > 0
    )

    newer_mask = (
        newer_edges > 0
    )

    older_count = np.count_nonzero(
        older_mask
    )

    newer_count = np.count_nonzero(
        newer_mask
    )

    if (
        older_count == 0
        and newer_count == 0
    ):
        return 1.0

    denominator = (
        older_count
        + newer_count
    )

    if denominator == 0:
        return 0.0

    intersection = np.count_nonzero(
        older_mask
        & newer_mask
    )

    score = (
        2.0
        * intersection
        / denominator
    )

    return float(
        score
    )


def compare_page(
    older_page,
    newer_page,
):
    """
    Render and structurally compare one page pair.
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

    older_edges = create_edge_map(
        older_normalized
    )

    newer_edges = create_edge_map(
        newer_normalized
    )

    similarity = dice_similarity(
        older_edges,
        newer_edges,
    )

    return {
        "similarity": similarity,
        "older_normalized":
            older_normalized,
        "newer_normalized":
            newer_normalized,
        "older_edges":
            older_edges,
        "newer_edges":
            newer_edges,
    }


def add_label(
    image,
    text,
):
    """
    Add a title bar above one grayscale diagnostic image.
    """
    if len(image.shape) == 2:
        display = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )
    else:
        display = image.copy()

    header_height = 38

    canvas = np.full(
        (
            display.shape[0]
            + header_height,
            display.shape[1],
            3,
        ),
        255,
        dtype=np.uint8,
    )

    canvas[
        header_height:,
        :,
    ] = display

    cv2.putText(
        canvas,
        text,
        (
            8,
            25,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            0,
            0,
            0,
        ),
        1,
        cv2.LINE_AA,
    )

    return canvas


def save_diagnostic_image(
    control,
    page_number,
    result,
):
    """
    Save a four-panel diagnostic image:

    OLD PAGE | NEW PAGE | OLD EDGES | NEW EDGES
    """
    older_page_panel = add_label(
        result[
            "older_normalized"
        ],
        "OLDER PAGE",
    )

    newer_page_panel = add_label(
        result[
            "newer_normalized"
        ],
        "NEWER PAGE",
    )

    older_edges_panel = add_label(
        result[
            "older_edges"
        ],
        "OLDER EDGES",
    )

    newer_edges_panel = add_label(
        result[
            "newer_edges"
        ],
        "NEWER EDGES",
    )

    combined = cv2.hconcat(
        [
            older_page_panel,
            newer_page_panel,
            older_edges_panel,
            newer_edges_panel,
        ]
    )

    footer_height = 45

    final_image = np.full(
        (
            combined.shape[0]
            + footer_height,
            combined.shape[1],
            3,
        ),
        255,
        dtype=np.uint8,
    )

    final_image[
        :combined.shape[0],
        :,
    ] = combined

    footer_text = (
        f'{control["label"]} | '
        f'page {page_number} | '
        f'structural similarity '
        f'{result["similarity"]:.4f}'
    )

    cv2.putText(
        final_image,
        footer_text,
        (
            12,
            combined.shape[0] + 29,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            0,
            0,
            0,
        ),
        1,
        cv2.LINE_AA,
    )

    filename = (
        f'{control["name"]}_'
        f'page_{page_number:04d}.png'
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    cv2.imwrite(
        str(output_path),
        final_image,
    )

    print(
        f"    Saved diagnostic: "
        f"{output_path}"
    )


def print_score_distribution(
    scores,
):
    """
    Print useful percentile values so we can compare controls.
    """
    values = np.array(
        scores,
        dtype=float,
    )

    print("")
    print(
        "Structural similarity distribution:"
    )

    print(
        f"  Mean: "
        f"{np.mean(values):.4f}"
    )

    print(
        f"  Median: "
        f"{np.median(values):.4f}"
    )

    print(
        f"  10th percentile: "
        f"{np.percentile(values, 10):.4f}"
    )

    print(
        f"  25th percentile: "
        f"{np.percentile(values, 25):.4f}"
    )

    print(
        f"  75th percentile: "
        f"{np.percentile(values, 75):.4f}"
    )

    print(
        f"  90th percentile: "
        f"{np.percentile(values, 90):.4f}"
    )

    print(
        f"  Minimum: "
        f"{np.min(values):.4f}"
    )

    print(
        f"  Maximum: "
        f"{np.max(values):.4f}"
    )


def process_control(
    control,
):
    """
    Compare all same-position pages for one control pair.
    """
    print("")
    print("=" * 72)

    print(
        control["label"]
    )

    print("=" * 72)

    older_document = get_document(
        control["set_num"],
        control[
            "older_document"
        ],
    )

    newer_document = get_document(
        control["set_num"],
        control[
            "newer_document"
        ],
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

    older_bytes = download_pdf(
        older_document
    )

    newer_bytes = download_pdf(
        newer_document
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
            f"Same-position pages compared: "
            f"{shared_page_count}"
        )

        print("")
        print(
            "Comparing structural edge geometry..."
        )

        page_results = []

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

            page_results.append({
                "page_number":
                    page_number,
                "similarity":
                    result[
                        "similarity"
                    ],
            })

            if (
                page_number == 1
                or page_number
                % 50 == 0
                or page_number
                == shared_page_count
            ):
                print(
                    f"  Processed page "
                    f"{page_number}/"
                    f"{shared_page_count}"
                )

        scores = [
            page[
                "similarity"
            ]
            for page
            in page_results
        ]

        print_score_distribution(
            scores
        )

        sorted_results = sorted(
            page_results,
            key=lambda item:
            item["similarity"],
        )

        print("")
        print(
            "Lowest structural-similarity pages:"
        )

        for item in sorted_results[
            :20
        ]:
            print(
                f'  Page '
                f'{item["page_number"]}: '
                f'{item["similarity"]:.4f}'
            )

        print("")
        print(
            f"Generating diagnostics for "
            f"the lowest "
            f"{LOWEST_PAGES_TO_SAVE} "
            f"pages..."
        )

        for item in sorted_results[
            :LOWEST_PAGES_TO_SAVE
        ]:
            page_number = (
                item[
                    "page_number"
                ]
            )

            detailed_result = compare_page(
                older_pdf[
                    page_number - 1
                ],
                newer_pdf[
                    page_number - 1
                ],
            )

            save_diagnostic_image(
                control,
                page_number,
                detailed_result,
            )

        if (
            older_page_count
            != newer_page_count
        ):
            print("")
            print(
                "NOTE: The PDFs have "
                "different page counts."
            )

            print(
                "Same-page-number comparison "
                "cannot account for page insertion "
                "or deletion yet."
            )

        return {
            "control":
                control["name"],
            "page_count_old":
                older_page_count,
            "page_count_new":
                newer_page_count,
            "mean_similarity":
                float(
                    np.mean(scores)
                ),
            "median_similarity":
                float(
                    np.median(scores)
                ),
            "minimum_similarity":
                float(
                    np.min(scores)
                ),
        }

    finally:
        older_pdf.close()
        newer_pdf.close()


def main():
    print("")
    print(
        "BrickTrip Structural Instruction Comparison"
    )

    print(
        "==========================================="
    )

    print(
        "READ ONLY — database writes: NONE"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    succeeded = 0
    failed = 0

    for control in CONTROLS:
        try:
            summary = process_control(
                control
            )

            summaries.append(
                summary
            )

            succeeded += 1

        except Exception as error:
            failed += 1

            print("")
            print(
                f"ERROR processing "
                f'{control["label"]}: '
                f"{error}"
            )

    print("")
    print("=" * 72)

    print(
        "CONTROL COMPARISON SUMMARY"
    )

    print("=" * 72)

    for summary in summaries:
        print("")

        print(
            summary["control"]
        )

        print(
            f'  Old pages: '
            f'{summary["page_count_old"]}'
        )

        print(
            f'  New pages: '
            f'{summary["page_count_new"]}'
        )

        print(
            f'  Mean structural similarity: '
            f'{summary["mean_similarity"]:.4f}'
        )

        print(
            f'  Median structural similarity: '
            f'{summary["median_similarity"]:.4f}'
        )

        print(
            f'  Lowest page similarity: '
            f'{summary["minimum_similarity"]:.4f}'
        )

    print("")
    print(
        f"Controls succeeded: "
        f"{succeeded}"
    )

    print(
        f"Controls failed: "
        f"{failed}"
    )

    print(
        "Database writes: 0"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
