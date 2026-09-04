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
# READ-ONLY PAGE-SEQUENCE ALIGNMENT TEST
# ------------------------------------------------------------
#
# Known positive control:
#   LEGO 42129-1
#
# Older PDF:
#   6396686
#
# Revised PDF:
#   6411661
#
# This worker:
# - downloads both official LEGO PDFs
# - renders every page once
# - creates structural edge maps
# - aligns the two page sequences
# - allows page insertions/deletions
# - identifies genuinely low-similarity aligned pages
#
# NOTHING is written to Supabase.
# ------------------------------------------------------------


TEST_SET_NUM = "42129-1"

OLDER_DOCUMENT = "6396686"

NEWER_DOCUMENT = "6411661"


OUTPUT_DIR = Path(
    "artifacts/bricktrip_pdf_comparisons"
)


# ------------------------------------------------------------
# IMAGE / STRUCTURAL SETTINGS
# ------------------------------------------------------------

RENDER_SCALE = 0.40

NORMALIZED_SIZE = 360

WHITE_THRESHOLD = 245

CROP_PADDING = 8

CANNY_LOW = 50

CANNY_HIGH = 140

EDGE_DILATION_SIZE = 3


# ------------------------------------------------------------
# ALIGNMENT SETTINGS
# ------------------------------------------------------------

# We already know this pair differs by only four PDF pages.
#
# A 12-page search window gives the aligner enough room to recover
# from insertions while keeping the computation reasonable.
ALIGNMENT_BAND = 12


# Similarity is between 0 and 1.
#
# We subtract this baseline before using a page match in the
# sequence alignment.
#
# Example:
#
# similarity 0.95 -> +0.45
# similarity 0.80 -> +0.30
# similarity 0.50 ->  0.00
# similarity 0.20 -> -0.30
#
MATCH_BASELINE = 0.50


# Penalty for inserting/deleting one PDF page.
#
# This lets the aligner prefer:
#
#   skip one unrelated inserted page
#
# instead of:
#
#   incorrectly compare every later page to the wrong page.
#
GAP_PENALTY = -0.18


# Number of lowest-similarity ALIGNED page pairs to save.
LOWEST_PAIRS_TO_SAVE = 15


def get_document(document_number):
    """
    Fetch one instruction-document record.
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
            TEST_SET_NUM,
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
            f"was not found."
        )

    return rows[0]


def download_pdf(document):
    """
    Download one official LEGO PDF.
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
        timeout=300,
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
    Open PDF bytes using PyMuPDF.
    """
    return pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )


def render_page(page):
    """
    Render one PDF page to a grayscale numpy image.
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
    Find the non-white printed content region.
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

    x, y, width, height = (
        cv2.boundingRect(
            coordinates
        )
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
    Crop excess margin and fit the page content into a fixed canvas.
    """
    (
        left,
        top,
        right,
        bottom,
    ) = find_content_bbox(
        image
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
    Convert normalized grayscale page into structural edge geometry.
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


def prepare_pdf_pages(pdf, label):
    """
    Render and prepare every page once.

    Each prepared page stores:
    - normalized grayscale image
    - structural edge map
    """
    prepared = []

    page_count = len(
        pdf
    )

    print("")
    print(
        f"Preparing {label} PDF..."
    )

    for index in range(
        page_count
    ):
        page_number = (
            index + 1
        )

        rendered = render_page(
            pdf[index]
        )

        normalized = normalize_page(
            rendered
        )

        edges = create_edge_map(
            normalized
        )

        prepared.append({
            "page_number":
                page_number,
            "image":
                normalized,
            "edges":
                edges,
        })

        if (
            page_number == 1
            or page_number % 50 == 0
            or page_number == page_count
        ):
            print(
                f"  Prepared page "
                f"{page_number}/"
                f"{page_count}"
            )

    return prepared


def dice_similarity(
    older_edges,
    newer_edges,
):
    """
    Structural Dice similarity.

    1.0 = extremely similar edge geometry.
    0.0 = no overlap.
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

    return float(
        2.0
        * intersection
        / denominator
    )


def build_similarity_cache(
    older_pages,
    newer_pages,
):
    """
    Precompute structural similarities only inside the alignment band.

    This avoids comparing every old page against every new page.
    """
    cache = {}

    old_count = len(
        older_pages
    )

    new_count = len(
        newer_pages
    )

    print("")
    print(
        "Building structural similarity window..."
    )

    comparisons = 0

    for old_index in range(
        old_count
    ):
        expected_new_index = (
            old_index
        )

        minimum_new_index = max(
            0,
            expected_new_index
            - ALIGNMENT_BAND,
        )

        maximum_new_index = min(
            new_count - 1,
            expected_new_index
            + ALIGNMENT_BAND,
        )

        for new_index in range(
            minimum_new_index,
            maximum_new_index + 1,
        ):
            similarity = dice_similarity(
                older_pages[
                    old_index
                ]["edges"],
                newer_pages[
                    new_index
                ]["edges"],
            )

            cache[
                (
                    old_index,
                    new_index,
                )
            ] = similarity

            comparisons += 1

        page_number = (
            old_index + 1
        )

        if (
            page_number == 1
            or page_number % 50 == 0
            or page_number == old_count
        ):
            print(
                f"  Similarity window "
                f"through old page "
                f"{page_number}/"
                f"{old_count}"
            )

    print(
        f"Similarity comparisons computed: "
        f"{comparisons:,}"
    )

    return cache


def get_similarity(
    cache,
    old_index,
    new_index,
):
    """
    Return cached similarity.

    Anything outside our alignment window is treated as impossible.
    """
    return cache.get(
        (
            old_index,
            new_index,
        )
    )


def align_page_sequences(
    older_pages,
    newer_pages,
    similarity_cache,
):
    """
    Global sequence alignment with page insertion/deletion support.

    Operations:
    M = match old page to new page
    D = old page has no corresponding new page
    I = new page was inserted relative to old PDF
    """
    old_count = len(
        older_pages
    )

    new_count = len(
        newer_pages
    )

    negative_infinity = (
        -1e12
    )

    scores = np.full(
        (
            old_count + 1,
            new_count + 1,
        ),
        negative_infinity,
        dtype=np.float64,
    )

    trace = np.full(
        (
            old_count + 1,
            new_count + 1,
        ),
        "",
        dtype="<U1",
    )

    scores[
        0,
        0,
    ] = 0.0

    # Leading deletions.
    for old_position in range(
        1,
        old_count + 1,
    ):
        if old_position <= ALIGNMENT_BAND:
            scores[
                old_position,
                0,
            ] = (
                scores[
                    old_position - 1,
                    0,
                ]
                + GAP_PENALTY
            )

            trace[
                old_position,
                0,
            ] = "D"

    # Leading insertions.
    for new_position in range(
        1,
        new_count + 1,
    ):
        if new_position <= ALIGNMENT_BAND:
            scores[
                0,
                new_position,
            ] = (
                scores[
                    0,
                    new_position - 1,
                ]
                + GAP_PENALTY
            )

            trace[
                0,
                new_position,
            ] = "I"

    print("")
    print(
        "Aligning PDF page sequences..."
    )

    for old_position in range(
        1,
        old_count + 1,
    ):
        minimum_new_position = max(
            1,
            old_position
            - ALIGNMENT_BAND,
        )

        maximum_new_position = min(
            new_count,
            old_position
            + ALIGNMENT_BAND,
        )

        for new_position in range(
            minimum_new_position,
            maximum_new_position + 1,
        ):
            old_index = (
                old_position - 1
            )

            new_index = (
                new_position - 1
            )

            similarity = get_similarity(
                similarity_cache,
                old_index,
                new_index,
            )

            match_score = (
                negative_infinity
            )

            if similarity is not None:
                previous = scores[
                    old_position - 1,
                    new_position - 1,
                ]

                if (
                    previous
                    > negative_infinity / 2
                ):
                    match_score = (
                        previous
                        + similarity
                        - MATCH_BASELINE
                    )

            delete_score = (
                scores[
                    old_position - 1,
                    new_position,
                ]
                + GAP_PENALTY
            )

            insert_score = (
                scores[
                    old_position,
                    new_position - 1,
                ]
                + GAP_PENALTY
            )

            best_score = max(
                match_score,
                delete_score,
                insert_score,
            )

            scores[
                old_position,
                new_position,
            ] = best_score

            if best_score == match_score:
                trace[
                    old_position,
                    new_position,
                ] = "M"

            elif best_score == delete_score:
                trace[
                    old_position,
                    new_position,
                ] = "D"

            else:
                trace[
                    old_position,
                    new_position,
                ] = "I"

        if (
            old_position == 1
            or old_position % 50 == 0
            or old_position == old_count
        ):
            print(
                f"  Aligned through old page "
                f"{old_position}/"
                f"{old_count}"
            )

    old_position = (
        old_count
    )

    new_position = (
        new_count
    )

    alignment = []

    while (
        old_position > 0
        or new_position > 0
    ):
        operation = trace[
            old_position,
            new_position,
        ]

        if operation == "M":
            old_index = (
                old_position - 1
            )

            new_index = (
                new_position - 1
            )

            similarity = get_similarity(
                similarity_cache,
                old_index,
                new_index,
            )

            alignment.append({
                "operation": "match",
                "old_page":
                    old_position,
                "new_page":
                    new_position,
                "similarity":
                    similarity,
            })

            old_position -= 1
            new_position -= 1

        elif operation == "D":
            alignment.append({
                "operation": "delete",
                "old_page":
                    old_position,
                "new_page":
                    None,
                "similarity":
                    None,
            })

            old_position -= 1

        elif operation == "I":
            alignment.append({
                "operation": "insert",
                "old_page":
                    None,
                "new_page":
                    new_position,
                "similarity":
                    None,
            })

            new_position -= 1

        else:
            raise RuntimeError(
                "Sequence alignment traceback "
                f"failed at old={old_position}, "
                f"new={new_position}."
            )

    alignment.reverse()

    return alignment


def summarize_alignment(
    alignment,
):
    """
    Print the discovered page correspondence and offset changes.
    """
    matches = [
        item
        for item in alignment
        if item["operation"]
        == "match"
    ]

    insertions = [
        item
        for item in alignment
        if item["operation"]
        == "insert"
    ]

    deletions = [
        item
        for item in alignment
        if item["operation"]
        == "delete"
    ]

    print("")
    print("=" * 72)
    print(
        "PAGE ALIGNMENT RESULT"
    )
    print("=" * 72)

    print(
        f"Matched page pairs: "
        f"{len(matches)}"
    )

    print(
        f"Pages inserted in newer PDF: "
        f"{len(insertions)}"
    )

    print(
        f"Pages present only in older PDF: "
        f"{len(deletions)}"
    )

    if insertions:
        print("")
        print(
            "Inserted newer pages:"
        )

        for item in insertions:
            print(
                f'  New page '
                f'{item["new_page"]}'
            )

    if deletions:
        print("")
        print(
            "Older-only pages:"
        )

        for item in deletions:
            print(
                f'  Old page '
                f'{item["old_page"]}'
            )

    print("")
    print(
        "Page-offset transitions:"
    )

    previous_offset = None

    transition_count = 0

    for item in matches:
        old_page = item[
            "old_page"
        ]

        new_page = item[
            "new_page"
        ]

        offset = (
            new_page
            - old_page
        )

        if (
            previous_offset is None
            or offset
            != previous_offset
        ):
            print(
                f"  Starting at old page "
                f"{old_page}: "
                f"old {old_page} -> "
                f"new {new_page} "
                f"(offset {offset:+d})"
            )

            previous_offset = (
                offset
            )

            transition_count += 1

    print(
        f"Offset regions found: "
        f"{transition_count}"
    )

    similarities = np.array(
        [
            item["similarity"]
            for item in matches
            if item["similarity"]
            is not None
        ],
        dtype=float,
    )

    print("")
    print(
        "Aligned structural similarity:"
    )

    print(
        f"  Mean: "
        f"{np.mean(similarities):.4f}"
    )

    print(
        f"  Median: "
        f"{np.median(similarities):.4f}"
    )

    print(
        f"  10th percentile: "
        f"{np.percentile(similarities, 10):.4f}"
    )

    print(
        f"  25th percentile: "
        f"{np.percentile(similarities, 25):.4f}"
    )

    print(
        f"  Minimum: "
        f"{np.min(similarities):.4f}"
    )

    return matches


def print_lowest_aligned_pairs(
    matches,
):
    """
    Print lowest-similarity pairs AFTER page alignment.
    """
    sorted_matches = sorted(
        matches,
        key=lambda item:
        item["similarity"],
    )

    print("")
    print(
        "Lowest aligned structural matches:"
    )

    for item in sorted_matches[
        :30
    ]:
        offset = (
            item["new_page"]
            - item["old_page"]
        )

        print(
            f'  Old {item["old_page"]} '
            f'-> New {item["new_page"]} '
            f'| offset {offset:+d} '
            f'| similarity '
            f'{item["similarity"]:.4f}'
        )

    return sorted_matches


def add_label(
    image,
    text,
):
    """
    Add a title header above one diagnostic panel.
    """
    if len(image.shape) == 2:
        display = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )
    else:
        display = image.copy()

    header_height = 40

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
            26,
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


def save_alignment_diagnostic(
    match,
    older_pages,
    newer_pages,
):
    """
    Save four-panel diagnostic for one aligned pair.

    OLD PAGE | NEW PAGE | OLD EDGES | NEW EDGES
    """
    old_page_number = (
        match["old_page"]
    )

    new_page_number = (
        match["new_page"]
    )

    older = older_pages[
        old_page_number - 1
    ]

    newer = newer_pages[
        new_page_number - 1
    ]

    older_page_panel = add_label(
        older["image"],
        f"OLD PAGE {old_page_number}",
    )

    newer_page_panel = add_label(
        newer["image"],
        f"NEW PAGE {new_page_number}",
    )

    older_edge_panel = add_label(
        older["edges"],
        "OLD EDGES",
    )

    newer_edge_panel = add_label(
        newer["edges"],
        "NEW EDGES",
    )

    combined = cv2.hconcat(
        [
            older_page_panel,
            newer_page_panel,
            older_edge_panel,
            newer_edge_panel,
        ]
    )

    footer_height = 48

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

    offset = (
        new_page_number
        - old_page_number
    )

    footer_text = (
        f"42129 aligned comparison | "
        f"old {old_page_number} -> "
        f"new {new_page_number} | "
        f"offset {offset:+d} | "
        f"similarity "
        f'{match["similarity"]:.4f}'
    )

    cv2.putText(
        final_image,
        footer_text,
        (
            12,
            combined.shape[0] + 30,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (
            0,
            0,
            0,
        ),
        1,
        cv2.LINE_AA,
    )

    filename = (
        f"42129_aligned_"
        f"old_{old_page_number:04d}_"
        f"new_{new_page_number:04d}.png"
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
        f"    Saved: "
        f"{output_path}"
    )


def main():
    print("")
    print(
        "BrickTrip LEGO PDF Sequence Alignment"
    )

    print(
        "====================================="
    )

    print(
        f"Controlled positive test: "
        f"{TEST_SET_NUM}"
    )

    print(
        "READ ONLY — database writes: NONE"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    older_document = get_document(
        OLDER_DOCUMENT
    )

    newer_document = get_document(
        NEWER_DOCUMENT
    )

    print("")
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
        print("")
        print(
            f"Older PDF pages: "
            f"{len(older_pdf)}"
        )

        print(
            f"Newer PDF pages: "
            f"{len(newer_pdf)}"
        )

        print(
            f"Raw page-count difference: "
            f"{len(newer_pdf) - len(older_pdf):+d}"
        )

        older_pages = (
            prepare_pdf_pages(
                older_pdf,
                "older",
            )
        )

        newer_pages = (
            prepare_pdf_pages(
                newer_pdf,
                "newer",
            )
        )

        similarity_cache = (
            build_similarity_cache(
                older_pages,
                newer_pages,
            )
        )

        alignment = (
            align_page_sequences(
                older_pages,
                newer_pages,
                similarity_cache,
            )
        )

        matches = summarize_alignment(
            alignment
        )

        sorted_matches = (
            print_lowest_aligned_pairs(
                matches
            )
        )

        print("")
        print(
            f"Saving diagnostics for "
            f"lowest "
            f"{LOWEST_PAIRS_TO_SAVE} "
            f"aligned pairs..."
        )

        for match in sorted_matches[
            :LOWEST_PAIRS_TO_SAVE
        ]:
            save_alignment_diagnostic(
                match,
                older_pages,
                newer_pages,
            )

        print("")
        print("=" * 72)

        print(
            "SEQUENCE ALIGNMENT TEST COMPLETE"
        )

        print(
            f"Total alignment operations: "
            f"{len(alignment)}"
        )

        print(
            f"Matched pairs: "
            f"{sum(1 for item in alignment if item['operation'] == 'match')}"
        )

        print(
            f"New-page insertions: "
            f"{sum(1 for item in alignment if item['operation'] == 'insert')}"
        )

        print(
            f"Old-page deletions: "
            f"{sum(1 for item in alignment if item['operation'] == 'delete')}"
        )

        print(
            "Database writes: 0"
        )

        print("=" * 72)

    finally:
        older_pdf.close()
        newer_pdf.close()


if __name__ == "__main__":
    main()
