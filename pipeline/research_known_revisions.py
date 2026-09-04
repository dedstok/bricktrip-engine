import os
import re
from collections import defaultdict
from datetime import datetime, timezone
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


# ============================================================
# BRICKTRIP INTEGRATED REVISION DIAGNOSTIC
# ============================================================
#
# READ ONLY.
#
# This controlled worker combines:
#
# SET
# -> instruction history
# -> booklet identity
# -> publication generations
# -> official LEGO PDF pairs
# -> normalized structural comparison
# -> page-sequence alignment
# -> provisional revision classification
#
# It DOES NOT:
# - insert evidence
# - update revision_candidates
# - create revisions
# - modify Supabase
#
# ============================================================


TEST_SET_NUMBERS = [
    "42171-1",
    "42129-1",
    "60004-1",
    "60085-1",
    "10194-1",
    "10214-1",
    "31058-1",
]


OUTPUT_DIR = Path(
    "artifacts/bricktrip_pdf_comparisons"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "integrated_revision_diagnostic.txt"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

RENDER_SCALE = 0.35

NORMALIZED_SIZE = 320

WHITE_THRESHOLD = 245

CROP_PADDING = 8

CANNY_LOW = 50

CANNY_HIGH = 140

EDGE_DILATION_SIZE = 3


# ============================================================
# ALIGNMENT SETTINGS
# ============================================================

ALIGNMENT_BAND = 12

MATCH_BASELINE = 0.50

GAP_PENALTY = -0.18


# ============================================================
# CLASSIFICATION SETTINGS
# ============================================================
#
# These are provisional diagnostic thresholds.
#
# We are testing whether the seven benchmark sets behave
# sensibly before any database-writing worker is restored.
# ============================================================

REPRINT_MEDIAN_MIN = 0.80

REPRINT_P10_MIN = 0.70

REPRINT_MINIMUM_MIN = 0.35

REVISION_VERY_LOW = 0.30

REVISION_LOW = 0.60


def log(message=""):
    """
    Print to GitHub log and save the same text to our artifact.
    """
    text = str(message)

    print(text)

    with SUMMARY_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            text + "\n"
        )


def parse_date(value):
    """
    Parse Supabase timestamp.
    """
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:
        return None


def effective_date(document):
    """
    Prefer source modification date.

    Fall back to date added.
    """
    modified = parse_date(
        document.get(
            "source_date_modified"
        )
    )

    if modified:
        return modified

    added = parse_date(
        document.get(
            "source_date_added"
        )
    )

    if added:
        return added

    return None


def date_key(document):
    """
    Calendar-date publication key.
    """
    value = effective_date(
        document
    )

    if not value:
        return None

    return value.date().isoformat()


def document_sort_key(document):
    """
    Safe chronological sorting.
    """
    value = effective_date(
        document
    )

    if value:
        return value

    return datetime.min.replace(
        tzinfo=timezone.utc
    )


def is_main_instruction(document):
    """
    Ignore obvious translation / support / alternate-model PDFs.
    """
    description = (
        document.get(
            "description"
        )
        or ""
    ).lower()

    source_url = (
        document.get(
            "source_url"
        )
        or ""
    ).lower()

    ignored_terms = [
        "translate",
        "translation",
        "additional.main",
        "additional.extra",
    ]

    for term in ignored_terms:
        if (
            term in description
            or term in source_url
        ):
            return False

    return True


def parse_versions(description):
    """
    Parse LEGO regional/version labels.

    Handles:

    V29
    V29/V118
    V46/39
    """
    if not description:
        return tuple()

    text = description.upper()

    versions = []

    matches = re.findall(
        r"\bV(\d+)(?:\s*/\s*(\d+))?",
        text,
    )

    for first, second in matches:

        first_label = (
            f"V{first}"
        )

        if first_label not in versions:
            versions.append(
                first_label
            )

        if second:

            second_label = (
                f"V{second}"
            )

            if (
                second_label
                not in versions
            ):
                versions.append(
                    second_label
                )

    return tuple(
        versions
    )


def parse_booklet_slot(description):
    """
    Parse physical instruction booklet identity.

    Examples:

    1/2
    2/3
    BOOK 1/3
    BOOK2/3

    Large BI/page-printing fractions are rejected.
    """
    if not description:
        return None

    text = description.upper()

    candidates = []

    matches = re.finditer(
        r"(?<!\d)"
        r"(\d{1,2})"
        r"\s*/\s*"
        r"(\d{1,2})"
        r"(?!\d)",
        text,
    )

    for match in matches:

        numerator = int(
            match.group(1)
        )

        denominator = int(
            match.group(2)
        )

        if numerator < 1:
            continue

        if denominator < 2:
            continue

        if denominator > 20:
            continue

        if numerator > denominator:
            continue

        candidates.append(
            (
                numerator,
                denominator,
            )
        )

    if not candidates:
        return None

    numerator, denominator = (
        candidates[-1]
    )

    return (
        f"{numerator}/"
        f"{denominator}"
    )


def get_instruction_documents(
    set_num,
):
    """
    Fetch instruction metadata for one set.
    """
    response = (
        supabase
        .table(
            "instruction_documents"
        )
        .select(
            "id,"
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
        .execute()
    )

    documents = (
        response.data
        or []
    )

    documents.sort(
        key=document_sort_key
    )

    return documents


def publication_generations(
    documents,
):
    """
    Group documents sharing one effective publication date.
    """
    grouped = defaultdict(
        list
    )

    for document in documents:

        key = date_key(
            document
        )

        if key is None:
            continue

        grouped[
            key
        ].append(
            document
        )

    generations = []

    for key, docs in (
        grouped.items()
    ):

        docs.sort(
            key=lambda document:
            str(
                document.get(
                    "document_number"
                )
                or ""
            )
        )

        generations.append({
            "date": key,
            "documents": docs,
        })

    generations.sort(
        key=lambda item:
        item["date"]
    )

    return generations


def representative_document(
    generation,
):
    """
    Pick one deterministic regional PDF from a publication group.
    """
    documents = (
        generation[
            "documents"
        ]
    )

    if not documents:
        return None

    return sorted(
        documents,
        key=lambda document:
        str(
            document.get(
                "document_number"
            )
            or ""
        ),
    )[0]


def build_metadata_pairs(
    documents,
):
    """
    Determine old/new PDFs that are actually comparable.

    Multi-booklet sets:
        compare 1/3 only with later 1/3, etc.

    No-slot sets:
        compare repeated V-label lineages.
    """
    primary = [
        document
        for document
        in documents
        if is_main_instruction(
            document
        )
    ]

    booklet_groups = (
        defaultdict(
            list
        )
    )

    no_slot_documents = []

    for document in primary:

        slot = parse_booklet_slot(
            document.get(
                "description"
            )
        )

        if slot:

            booklet_groups[
                slot
            ].append(
                document
            )

        else:

            no_slot_documents.append(
                document
            )

    pairs = []

    # --------------------------------------------------------
    # MULTI-BOOKLET SETS
    # --------------------------------------------------------

    if booklet_groups:

        for slot, slot_documents in sorted(
            booklet_groups.items()
        ):

            generations = (
                publication_generations(
                    slot_documents
                )
            )

            for index in range(
                len(generations) - 1
            ):

                older_generation = (
                    generations[
                        index
                    ]
                )

                newer_generation = (
                    generations[
                        index + 1
                    ]
                )

                older_document = (
                    representative_document(
                        older_generation
                    )
                )

                newer_document = (
                    representative_document(
                        newer_generation
                    )
                )

                if (
                    older_document
                    and newer_document
                ):
                    pairs.append({
                        "identity":
                            f"booklet {slot}",
                        "older_generation":
                            older_generation,
                        "newer_generation":
                            newer_generation,
                        "older_document":
                            older_document,
                        "newer_document":
                            newer_document,
                    })

        return pairs

    # --------------------------------------------------------
    # SINGLE BOOKLET / NO SLOT
    # --------------------------------------------------------

    version_groups = (
        defaultdict(
            list
        )
    )

    for document in no_slot_documents:

        versions = parse_versions(
            document.get(
                "description"
            )
        )

        for version in versions:

            version_groups[
                version
            ].append(
                document
            )

    seen_pairs = set()

    for version, version_documents in sorted(
        version_groups.items()
    ):

        generations = (
            publication_generations(
                version_documents
            )
        )

        for index in range(
            len(generations) - 1
        ):

            older_generation = (
                generations[
                    index
                ]
            )

            newer_generation = (
                generations[
                    index + 1
                ]
            )

            older_document = (
                representative_document(
                    older_generation
                )
            )

            newer_document = (
                representative_document(
                    newer_generation
                )
            )

            if (
                not older_document
                or not newer_document
            ):
                continue

            older_number = str(
                older_document.get(
                    "document_number"
                )
            )

            newer_number = str(
                newer_document.get(
                    "document_number"
                )
            )

            pair_key = (
                older_number,
                newer_number,
            )

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(
                pair_key
            )

            pairs.append({
                "identity":
                    f"version {version}",
                "older_generation":
                    older_generation,
                "newer_generation":
                    newer_generation,
                "older_document":
                    older_document,
                "newer_document":
                    newer_document,
            })

    return pairs


def download_pdf(
    document,
    cache,
):
    """
    Download official LEGO PDF with in-run caching.
    """
    document_number = str(
        document.get(
            "document_number"
        )
    )

    if document_number in cache:
        return cache[
            document_number
        ]

    source_url = document.get(
        "source_url"
    )

    if not source_url:
        raise RuntimeError(
            f"Document "
            f"{document_number} "
            f"has no source_url."
        )

    log(
        f"    Downloading "
        f"{document_number}..."
    )

    response = requests.get(
        source_url,
        timeout=300,
    )

    response.raise_for_status()

    pdf_bytes = (
        response.content
    )

    if not pdf_bytes.startswith(
        b"%PDF"
    ):
        raise RuntimeError(
            f"Document "
            f"{document_number} "
            f"did not download as PDF."
        )

    log(
        f"      "
        f"{len(pdf_bytes):,} bytes"
    )

    cache[
        document_number
    ] = pdf_bytes

    return pdf_bytes


def open_pdf(
    pdf_bytes,
):
    """
    Open bytes with PyMuPDF.
    """
    return pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )


def render_page(
    page,
):
    """
    Render PDF page into grayscale numpy image.
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

    return image.reshape(
        pixmap.height,
        pixmap.width,
    )


def find_content_bbox(
    image,
):
    """
    Detect printed area.
    """
    mask = (
        image
        < WHITE_THRESHOLD
    ).astype(
        np.uint8
    )

    coordinates = (
        cv2.findNonZero(
            mask
        )
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
        x + width
        + CROP_PADDING,
    )

    bottom = min(
        image.shape[0],
        y + height
        + CROP_PADDING,
    )

    return (
        left,
        top,
        right,
        bottom,
    )


def normalize_page(
    image,
):
    """
    Crop margins and normalize page scale.
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

    maximum = (
        NORMALIZED_SIZE
        - 20
    )

    height, width = (
        cropped.shape
    )

    scale = min(
        maximum / width,
        maximum / height,
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


def create_edges(
    image,
):
    """
    Create color-independent structural representation.
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

    return cv2.dilate(
        edges,
        kernel,
        iterations=1,
    )


def prepare_pdf(
    pdf,
):
    """
    Render all pages once.
    """
    prepared = []

    for page in pdf:

        rendered = render_page(
            page
        )

        normalized = (
            normalize_page(
                rendered
            )
        )

        edges = create_edges(
            normalized
        )

        prepared.append(
            edges
        )

    return prepared


def dice_similarity(
    older_edges,
    newer_edges,
):
    """
    Structural similarity score.
    """
    older_mask = (
        older_edges > 0
    )

    newer_mask = (
        newer_edges > 0
    )

    old_count = (
        np.count_nonzero(
            older_mask
        )
    )

    new_count = (
        np.count_nonzero(
            newer_mask
        )
    )

    if (
        old_count == 0
        and new_count == 0
    ):
        return 1.0

    denominator = (
        old_count
        + new_count
    )

    if denominator == 0:
        return 0.0

    intersection = (
        np.count_nonzero(
            older_mask
            & newer_mask
        )
    )

    return float(
        (
            2.0
            * intersection
        )
        / denominator
    )


def build_similarity_cache(
    older_pages,
    newer_pages,
):
    """
    Compare only pages near the expected sequence position.
    """
    cache = {}

    old_count = len(
        older_pages
    )

    new_count = len(
        newer_pages
    )

    for old_index in range(
        old_count
    ):

        minimum = max(
            0,
            old_index
            - ALIGNMENT_BAND,
        )

        maximum = min(
            new_count - 1,
            old_index
            + ALIGNMENT_BAND,
        )

        for new_index in range(
            minimum,
            maximum + 1,
        ):

            cache[
                (
                    old_index,
                    new_index,
                )
            ] = dice_similarity(
                older_pages[
                    old_index
                ],
                newer_pages[
                    new_index
                ],
            )

    return cache


def align_pages(
    older_pages,
    newer_pages,
    similarity_cache,
):
    """
    Global page-sequence alignment.

    Allows inserted and removed PDF pages.
    """
    old_count = len(
        older_pages
    )

    new_count = len(
        newer_pages
    )

    negative = -1e12

    scores = np.full(
        (
            old_count + 1,
            new_count + 1,
        ),
        negative,
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

    for old_position in range(
        1,
        min(
            old_count,
            ALIGNMENT_BAND,
        ) + 1,
    ):

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

    for new_position in range(
        1,
        min(
            new_count,
            ALIGNMENT_BAND,
        ) + 1,
    ):

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

    for old_position in range(
        1,
        old_count + 1,
    ):

        minimum_new = max(
            1,
            old_position
            - ALIGNMENT_BAND,
        )

        maximum_new = min(
            new_count,
            old_position
            + ALIGNMENT_BAND,
        )

        for new_position in range(
            minimum_new,
            maximum_new + 1,
        ):

            old_index = (
                old_position - 1
            )

            new_index = (
                new_position - 1
            )

            similarity = (
                similarity_cache.get(
                    (
                        old_index,
                        new_index,
                    )
                )
            )

            match_score = (
                negative
            )

            if similarity is not None:

                previous = scores[
                    old_position - 1,
                    new_position - 1,
                ]

                if previous > (
                    negative / 2
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

            best = max(
                match_score,
                delete_score,
                insert_score,
            )

            scores[
                old_position,
                new_position,
            ] = best

            if best == match_score:
                trace[
                    old_position,
                    new_position,
                ] = "M"

            elif best == delete_score:
                trace[
                    old_position,
                    new_position,
                ] = "D"

            else:
                trace[
                    old_position,
                    new_position,
                ] = "I"

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

            similarity = (
                similarity_cache[
                    (
                        old_position - 1,
                        new_position - 1,
                    )
                ]
            )

            alignment.append({
                "operation":
                    "match",
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
                "operation":
                    "delete",
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
                "operation":
                    "insert",
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
                "Page alignment "
                "traceback failed."
            )

    alignment.reverse()

    return alignment


def analyze_pdf_pair(
    pair,
    pdf_cache,
):
    """
    Perform structural aligned comparison of one old/new PDF pair.
    """
    older_document = (
        pair[
            "older_document"
        ]
    )

    newer_document = (
        pair[
            "newer_document"
        ]
    )

    older_number = str(
        older_document.get(
            "document_number"
        )
    )

    newer_number = str(
        newer_document.get(
            "document_number"
        )
    )

    older_bytes = download_pdf(
        older_document,
        pdf_cache,
    )

    newer_bytes = download_pdf(
        newer_document,
        pdf_cache,
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

        older_pages = prepare_pdf(
            older_pdf
        )

        newer_pages = prepare_pdf(
            newer_pdf
        )

        similarity_cache = (
            build_similarity_cache(
                older_pages,
                newer_pages,
            )
        )

        alignment = align_pages(
            older_pages,
            newer_pages,
            similarity_cache,
        )

        matches = [
            item
            for item
            in alignment
            if item["operation"]
            == "match"
        ]

        insertions = [
            item
            for item
            in alignment
            if item["operation"]
            == "insert"
        ]

        deletions = [
            item
            for item
            in alignment
            if item["operation"]
            == "delete"
        ]

        similarities = np.array(
            [
                item[
                    "similarity"
                ]
                for item
                in matches
            ],
            dtype=float,
        )

        if len(
            similarities
        ) == 0:

            raise RuntimeError(
                "No aligned page matches "
                "were produced."
            )

        median = float(
            np.median(
                similarities
            )
        )

        p10 = float(
            np.percentile(
                similarities,
                10,
            )
        )

        minimum = float(
            np.min(
                similarities
            )
        )

        mean = float(
            np.mean(
                similarities
            )
        )

        low_count = int(
            np.count_nonzero(
                similarities
                < REVISION_LOW
            )
        )

        very_low_count = int(
            np.count_nonzero(
                similarities
                < REVISION_VERY_LOW
            )
        )

        low_fraction = (
            low_count
            / len(
                similarities
            )
        )

        return {
            "older_document":
                older_number,
            "newer_document":
                newer_number,
            "older_pages":
                older_page_count,
            "newer_pages":
                newer_page_count,
            "page_difference":
                newer_page_count
                - older_page_count,
            "matched_pages":
                len(
                    matches
                ),
            "insertions":
                len(
                    insertions
                ),
            "deletions":
                len(
                    deletions
                ),
            "mean":
                mean,
            "median":
                median,
            "p10":
                p10,
            "minimum":
                minimum,
            "low_count":
                low_count,
            "very_low_count":
                very_low_count,
            "low_fraction":
                low_fraction,
        }

    finally:

        older_pdf.close()
        newer_pdf.close()


def classify_pair(
    result,
):
    """
    Provisional old/new PDF-pair classification.
    """
    gaps = (
        result[
            "insertions"
        ]
        + result[
            "deletions"
        ]
    )

    median = result[
        "median"
    ]

    p10 = result[
        "p10"
    ]

    minimum = result[
        "minimum"
    ]

    low_fraction = result[
        "low_fraction"
    ]

    very_low_count = result[
        "very_low_count"
    ]

    # --------------------------------------------------------
    # Strong localized revision pattern.
    #
    # Example:
    # 42129
    #
    # Most pages nearly identical, with a localized region of
    # very poor matches and page sequence changes.
    # --------------------------------------------------------

    if (
        median >= 0.85
        and (
            very_low_count >= 1
            or gaps >= 2
            or low_fraction >= 0.08
        )
    ):
        return (
            "revision_candidate"
        )

    # --------------------------------------------------------
    # Broad substantial difference.
    # --------------------------------------------------------

    if (
        low_fraction >= 0.20
        or p10 < 0.45
    ):
        return (
            "revision_candidate"
        )

    # --------------------------------------------------------
    # Same-build / reprint pattern.
    #
    # Example:
    # 10214 visual refresh.
    # --------------------------------------------------------

    if (
        median
        >= REPRINT_MEDIAN_MIN
        and p10
        >= REPRINT_P10_MIN
        and minimum
        >= REPRINT_MINIMUM_MIN
        and gaps <= 1
    ):
        return (
            "same_build_reprint"
        )

    return (
        "needs_deeper_review"
    )


def classify_set(
    pair_results,
):
    """
    Roll all comparable booklet results into one set-level result.
    """
    if not pair_results:
        return (
            "no_comparable_generation"
        )

    classifications = [
        item[
            "classification"
        ]
        for item
        in pair_results
    ]

    if (
        "revision_candidate"
        in classifications
    ):
        return (
            "revision_candidate"
        )

    if all(
        classification
        == "same_build_reprint"
        for classification
        in classifications
    ):
        return (
            "same_build_reprint"
        )

    return (
        "needs_deeper_review"
    )


def research_set(
    set_num,
    pdf_cache,
):
    """
    Run integrated read-only research for one benchmark set.
    """
    log("")
    log(
        "=" * 72
    )

    log(
        f"SET {set_num}"
    )

    log(
        "=" * 72
    )

    documents = (
        get_instruction_documents(
            set_num
        )
    )

    primary_documents = [
        document
        for document
        in documents
        if is_main_instruction(
            document
        )
    ]

    log(
        f"Instruction records: "
        f"{len(documents)}"
    )

    log(
        f"Primary instruction PDFs: "
        f"{len(primary_documents)}"
    )

    pairs = (
        build_metadata_pairs(
            documents
        )
    )

    log(
        f"Comparable metadata pairs: "
        f"{len(pairs)}"
    )

    if not pairs:

        log(
            "SET RESULT: "
            "no_comparable_generation"
        )

        return {
            "set_num":
                set_num,
            "classification":
                "no_comparable_generation",
            "pairs":
                [],
        }

    pair_results = []

    for pair_index, pair in enumerate(
        pairs,
        start=1,
    ):

        older_document = (
            pair[
                "older_document"
            ]
        )

        newer_document = (
            pair[
                "newer_document"
            ]
        )

        older_number = str(
            older_document.get(
                "document_number"
            )
        )

        newer_number = str(
            newer_document.get(
                "document_number"
            )
        )

        log("")
        log(
            f"  Pair "
            f"{pair_index}/"
            f"{len(pairs)}"
        )

        log(
            f"  Identity: "
            f'{pair["identity"]}'
        )

        log(
            f"  "
            f"{older_number} "
            f"-> "
            f"{newer_number}"
        )

        log(
            f"  "
            f'{pair["older_generation"]["date"]}'
            f" -> "
            f'{pair["newer_generation"]["date"]}'
        )

        try:

            result = (
                analyze_pdf_pair(
                    pair,
                    pdf_cache,
                )
            )

            classification = (
                classify_pair(
                    result
                )
            )

            result[
                "identity"
            ] = pair[
                "identity"
            ]

            result[
                "classification"
            ] = (
                classification
            )

            pair_results.append(
                result
            )

            log(
                f"    Old pages: "
                f'{result["older_pages"]}'
            )

            log(
                f"    New pages: "
                f'{result["newer_pages"]}'
            )

            log(
                f"    Page count delta: "
                f'{result["page_difference"]:+d}'
            )

            log(
                f"    Alignment gaps: "
                f'{result["insertions"]} '
                f"new-only / "
                f'{result["deletions"]} '
                f"old-only"
            )

            log(
                f"    Mean similarity: "
                f'{result["mean"]:.4f}'
            )

            log(
                f"    Median similarity: "
                f'{result["median"]:.4f}'
            )

            log(
                f"    10th percentile: "
                f'{result["p10"]:.4f}'
            )

            log(
                f"    Minimum: "
                f'{result["minimum"]:.4f}'
            )

            log(
                f"    Pages below "
                f"{REVISION_LOW:.2f}: "
                f'{result["low_count"]}'
            )

            log(
                f"    Pages below "
                f"{REVISION_VERY_LOW:.2f}: "
                f'{result["very_low_count"]}'
            )

            log(
                f"    PAIR RESULT: "
                f"{classification}"
            )

        except Exception as error:

            log(
                f"    ERROR: "
                f"{error}"
            )

            pair_results.append({
                "identity":
                    pair[
                        "identity"
                    ],
                "classification":
                    "needs_deeper_review",
                "error":
                    str(
                        error
                    ),
            })

    set_classification = (
        classify_set(
            pair_results
        )
    )

    log("")
    log(
        f"SET RESULT: "
        f"{set_classification}"
    )

    return {
        "set_num":
            set_num,
        "classification":
            set_classification,
        "pairs":
            pair_results,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_FILE.write_text(
        "",
        encoding="utf-8",
    )

    log("")
    log(
        "BrickTrip Integrated Revision Diagnostic"
    )

    log(
        "========================================"
    )

    log(
        "READ ONLY — database writes: NONE"
    )

    log("")
    log(
        f"Benchmark sets: "
        f"{len(TEST_SET_NUMBERS)}"
    )

    pdf_cache = {}

    results = []

    succeeded = 0
    failed = 0

    for set_num in (
        TEST_SET_NUMBERS
    ):

        try:

            result = research_set(
                set_num,
                pdf_cache,
            )

            results.append(
                result
            )

            succeeded += 1

        except Exception as error:

            failed += 1

            log("")
            log(
                f"FATAL ERROR for "
                f"{set_num}: "
                f"{error}"
            )

            results.append({
                "set_num":
                    set_num,
                "classification":
                    "needs_deeper_review",
                "error":
                    str(
                        error
                    ),
            })

    log("")
    log(
        "=" * 72
    )

    log(
        "BENCHMARK SUMMARY"
    )

    log(
        "=" * 72
    )

    for result in results:

        log(
            f'{result["set_num"]}: '
            f'{result["classification"]}'
        )

    counts = (
        defaultdict(
            int
        )
    )

    for result in results:

        counts[
            result[
                "classification"
            ]
        ] += 1

    log("")
    log(
        "Classification totals:"
    )

    for classification in [
        "revision_candidate",
        "same_build_reprint",
        "no_comparable_generation",
        "needs_deeper_review",
    ]:

        log(
            f"  "
            f"{classification}: "
            f"{counts[classification]}"
        )

    log("")
    log(
        f"Sets completed: "
        f"{succeeded}"
    )

    log(
        f"Sets failed: "
        f"{failed}"
    )

    log(
        "Database writes: 0"
    )

    log(
        "=" * 72
    )


if __name__ == "__main__":
    main()
