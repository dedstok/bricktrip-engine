import os
import re
from collections import defaultdict
from datetime import datetime, timezone

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
# BRICKTRIP PRODUCTION REVISION RESEARCH WORKER
# ============================================================
#
# This worker processes KNOWN production-redesign sets.
#
# It DOES:
#
# - load up to 10 needs_research candidates
# - inspect LEGO instruction-document history
# - understand multi-booklet structure
# - identify comparable publication generations
# - download official LEGO PDFs
# - normalize PDF pages
# - compare structural instruction geometry
# - align page sequences
# - record evidence when a likely real revision is exposed
# - route unresolved cases to deeper research
# - log the pipeline run
#
#
# It DOES NOT:
#
# - create verified revisions
# - create inventory overrides
# - claim a redesign does not exist
# - treat PDF reprints as proof against a redesign
#
#
# These sets are already known redesigns.
#
# The worker's job is to CHARACTERIZE them.
# ============================================================


MAX_SETS_PER_RUN = 10


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
# PAGE ALIGNMENT SETTINGS
# ============================================================

ALIGNMENT_BAND = 12

MATCH_BASELINE = 0.50

GAP_PENALTY = -0.18


# ============================================================
# CLASSIFICATION SETTINGS
# ============================================================

REPRINT_MEDIAN_MIN = 0.80

REPRINT_P10_MIN = 0.70

REPRINT_MINIMUM_MIN = 0.35

REVISION_LOW = 0.60

REVISION_VERY_LOW = 0.30


# ============================================================
# DATABASE STATUS VALUES
# ============================================================

STATUS_NEEDS_RESEARCH = (
    "needs_research"
)

STATUS_REVISION_EVIDENCE_FOUND = (
    "revision_evidence_found"
)

STATUS_NEEDS_DEEPER_RESEARCH = (
    "needs_deeper_research"
)


# ============================================================
# EVIDENCE TYPES
# ============================================================

EVIDENCE_TYPE_STRUCTURAL_REVISION = (
    "instruction_structural_revision"
)


def parse_date(value):
    """
    Parse a timestamp returned by Supabase.
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
    Prefer source_date_modified.

    Fall back to source_date_added.
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
    Return publication date as YYYY-MM-DD.
    """
    value = effective_date(
        document
    )

    if not value:
        return None

    return value.date().isoformat()


def document_sort_key(document):
    """
    Safe date sorting.
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
    Ignore translation, support, and alternate-model PDFs.
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
    Parse LEGO V labels.

    Examples:

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
    Parse physical booklet identity.

    Examples:

    1/2
    2/3
    BOOK 2/3
    BOOK2/3

    Large BI print-code fractions are rejected.
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


def get_candidates():
    """
    Load up to MAX_SETS_PER_RUN clean queue candidates.
    """
    response = (
        supabase
        .table(
            "revision_candidates"
        )
        .select(
            "id,"
            "set_num,"
            "status,"
            "reason"
        )
        .eq(
            "status",
            STATUS_NEEDS_RESEARCH,
        )
        .order(
            "id"
        )
        .limit(
            MAX_SETS_PER_RUN
        )
        .execute()
    )

    return (
        response.data
        or []
    )


def get_instruction_documents(
    set_num,
):
    """
    Fetch all instruction-document metadata for a set.
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
            "date":
                key,
            "documents":
                docs,
        })

    generations.sort(
        key=lambda generation:
        generation[
            "date"
        ]
    )

    return generations


def representative_document(
    generation,
):
    """
    Pick one deterministic regional PDF from a generation.
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
    Build comparable old/new PDF candidates.

    Multi-booklet:
        compare like booklet slots only.

    No booklet slots:
        compare repeated LEGO V-label lineages.
    """
    primary = [
        document
        for document
        in documents
        if is_main_instruction(
            document
        )
    ]

    booklet_groups = defaultdict(
        list
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
    # MULTI-BOOKLET
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

    version_groups = defaultdict(
        list
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
    Download an official LEGO PDF with in-run caching.
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

    print(
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

    print(
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
    Open PDF bytes.
    """
    return pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )


def render_page(
    page,
):
    """
    Render one PDF page to grayscale.
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
    Detect non-white printed content.
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
    Crop margins and normalize scale.
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
    Create structural edge map.
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
    Render and structurally prepare every page once.
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
    Structural Dice similarity.
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
    Compare only pages near expected sequence position.
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
    Global sequence alignment.

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
    Structurally compare one old/new PDF pair.
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

        older_pages = (
            prepare_pdf(
                older_pdf
            )
        )

        newer_pages = (
            prepare_pdf(
                newer_pdf
            )
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
            if item[
                "operation"
            ]
            == "match"
        ]

        insertions = [
            item
            for item
            in alignment
            if item[
                "operation"
            ]
            == "insert"
        ]

        deletions = [
            item
            for item
            in alignment
            if item[
                "operation"
            ]
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

        mean = float(
            np.mean(
                similarities
            )
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
    Determine what one PDF pair tells us.
    """
    gaps = (
        result[
            "insertions"
        ]
        + result[
            "deletions"
        ]
    )

    median = (
        result[
            "median"
        ]
    )

    p10 = (
        result[
            "p10"
        ]
    )

    minimum = (
        result[
            "minimum"
        ]
    )

    low_fraction = (
        result[
            "low_fraction"
        ]
    )

    very_low_count = (
        result[
            "very_low_count"
        ]
    )

    # --------------------------------------------------------
    # LOCALIZED REVISION PATTERN
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
            "revision_evidence_found"
        )

    # --------------------------------------------------------
    # BROAD STRUCTURAL DIFFERENCE
    # --------------------------------------------------------

    if (
        low_fraction >= 0.20
        or p10 < 0.45
    ):

        return (
            "revision_evidence_found"
        )

    # --------------------------------------------------------
    # SAME-BUILD / REPRINT PAIR
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
            "pdf_reprint_only"
        )

    return (
        "needs_deeper_research"
    )


def set_finding(
    pair_results,
):
    """
    Roll pair findings into one set finding.
    """
    if not pair_results:

        return (
            "no_comparable_pdfs"
        )

    classifications = [
        result[
            "classification"
        ]
        for result
        in pair_results
    ]

    if (
        "revision_evidence_found"
        in classifications
    ):

        return (
            "revision_evidence_found"
        )

    if (
        "needs_deeper_research"
        in classifications
    ):

        return (
            "needs_deeper_research"
        )

    if all(
        classification
        == "pdf_reprint_only"
        for classification
        in classifications
    ):

        return (
            "pdf_reprint_only"
        )

    return (
        "needs_deeper_research"
    )


def build_evidence_description(
    pair,
    result,
):
    """
    Create human-readable structural revision evidence.
    """
    return (
        "Official LEGO instruction comparison found structural "
        "revision evidence. "
        f'{pair["identity"]}: '
        f'{result["older_document"]} '
        f'({pair["older_generation"]["date"]}) '
        f'-> '
        f'{result["newer_document"]} '
        f'({pair["newer_generation"]["date"]}). '
        f'Pages '
        f'{result["older_pages"]}'
        f' -> '
        f'{result["newer_pages"]}; '
        f'alignment gaps '
        f'{result["insertions"]} newer-only / '
        f'{result["deletions"]} older-only; '
        f'median structural similarity '
        f'{result["median"]:.4f}; '
        f'10th percentile '
        f'{result["p10"]:.4f}; '
        f'minimum '
        f'{result["minimum"]:.4f}; '
        f'{result["low_count"]} aligned pages '
        f'below {REVISION_LOW:.2f}.'
    )


def evidence_exists(
    set_num,
    source_url,
    description,
):
    """
    Prevent duplicate evidence if a partially completed set is retried.
    """
    query = (
        supabase
        .table(
            "evidence"
        )
        .select(
            "id"
        )
        .eq(
            "set_num",
            set_num,
        )
        .eq(
            "source_type",
            EVIDENCE_TYPE_STRUCTURAL_REVISION,
        )
        .eq(
            "description",
            description,
        )
        .limit(1)
    )

    if source_url:

        query = query.eq(
            "source_url",
            source_url,
        )

    response = (
        query.execute()
    )

    return bool(
        response.data
    )


def save_revision_evidence(
    set_num,
    pair,
    result,
):
    """
    Save one evidence row for a structurally meaningful PDF pair.
    """
    newer_document = (
        pair[
            "newer_document"
        ]
    )

    source_url = (
        newer_document.get(
            "source_url"
        )
    )

    description = (
        build_evidence_description(
            pair,
            result,
        )
    )

    if evidence_exists(
        set_num,
        source_url,
        description,
    ):

        print(
            "    Evidence already exists; "
            "skipping duplicate."
        )

        return

    evidence_row = {
        "set_num":
            set_num,
        "revision_id":
            None,
        "source_type":
            EVIDENCE_TYPE_STRUCTURAL_REVISION,
        "source_url":
            source_url,
        "description":
            description,
        "confidence":
            0.75,
    }

    (
        supabase
        .table(
            "evidence"
        )
        .insert(
            evidence_row
        )
        .execute()
    )

    print(
        "    Saved structural "
        "revision evidence."
    )


def update_candidate(
    candidate_id,
    status,
    reason,
):
    """
    Update one revision candidate.
    """
    (
        supabase
        .table(
            "revision_candidates"
        )
        .update({
            "status":
                status,
            "reason":
                reason,
            "updated_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        })
        .eq(
            "id",
            candidate_id,
        )
        .execute()
    )


def process_candidate(
    candidate,
    pdf_cache,
):
    """
    Research one known-redesign queue candidate.
    """
    candidate_id = (
        candidate[
            "id"
        ]
    )

    set_num = (
        candidate[
            "set_num"
        ]
    )

    print("")
    print(
        "=" * 72
    )

    print(
        f"RESEARCHING "
        f"{set_num}"
    )

    print(
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

    print(
        f"Instruction records: "
        f"{len(documents)}"
    )

    print(
        f"Primary instruction PDFs: "
        f"{len(primary_documents)}"
    )

    pairs = (
        build_metadata_pairs(
            documents
        )
    )

    print(
        f"Comparable PDF pairs: "
        f"{len(pairs)}"
    )

    # --------------------------------------------------------
    # NO COMPARABLE PDF PAIRS
    # --------------------------------------------------------

    if not pairs:

        reason = (
            "Known production redesign, but current instruction "
            "metadata does not expose comparable old/new PDFs. "
            "Requires deeper research."
        )

        update_candidate(
            candidate_id,
            STATUS_NEEDS_DEEPER_RESEARCH,
            reason,
        )

        print(
            "RESULT: "
            "needs_deeper_research "
            "(no comparable PDFs)"
        )

        return {
            "set_num":
                set_num,
            "finding":
                "no_comparable_pdfs",
            "status":
                STATUS_NEEDS_DEEPER_RESEARCH,
            "evidence_rows":
                0,
        }

    pair_results = []

    saved_evidence_count = 0

    for pair_index, pair in enumerate(
        pairs,
        start=1,
    ):

        older_number = str(
            pair[
                "older_document"
            ].get(
                "document_number"
            )
        )

        newer_number = str(
            pair[
                "newer_document"
            ].get(
                "document_number"
            )
        )

        print("")
        print(
            f"  Pair "
            f"{pair_index}/"
            f"{len(pairs)}"
        )

        print(
            f"  Identity: "
            f'{pair["identity"]}'
        )

        print(
            f"  "
            f"{older_number} "
            f"-> "
            f"{newer_number}"
        )

        print(
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
                "classification"
            ] = (
                classification
            )

            pair_results.append(
                result
            )

            print(
                f"    Pages: "
                f'{result["older_pages"]}'
                f" -> "
                f'{result["newer_pages"]}'
            )

            print(
                f"    Alignment gaps: "
                f'{result["insertions"]} '
                f"new-only / "
                f'{result["deletions"]} '
                f"old-only"
            )

            print(
                f"    Mean similarity: "
                f'{result["mean"]:.4f}'
            )

            print(
                f"    Median similarity: "
                f'{result["median"]:.4f}'
            )

            print(
                f"    10th percentile: "
                f'{result["p10"]:.4f}'
            )

            print(
                f"    Minimum: "
                f'{result["minimum"]:.4f}'
            )

            print(
                f"    Pages below "
                f"{REVISION_LOW:.2f}: "
                f'{result["low_count"]}'
            )

            print(
                f"    PAIR FINDING: "
                f"{classification}"
            )

            if (
                classification
                == "revision_evidence_found"
            ):

                save_revision_evidence(
                    set_num,
                    pair,
                    result,
                )

                saved_evidence_count += 1

        except Exception as error:

            print(
                f"    Pair error: "
                f"{error}"
            )

            pair_results.append({
                "classification":
                    "needs_deeper_research",
                "error":
                    str(
                        error
                    ),
            })

    finding = (
        set_finding(
            pair_results
        )
    )

    # --------------------------------------------------------
    # REVISION EVIDENCE FOUND
    # --------------------------------------------------------

    if (
        finding
        == "revision_evidence_found"
    ):

        reason = (
            "Structural differences were found between comparable "
            "official LEGO instruction generations. Revision "
            "evidence recorded; characterization can continue."
        )

        update_candidate(
            candidate_id,
            STATUS_REVISION_EVIDENCE_FOUND,
            reason,
        )

        print("")
        print(
            "RESULT: "
            "revision_evidence_found"
        )

        return {
            "set_num":
                set_num,
            "finding":
                finding,
            "status":
                STATUS_REVISION_EVIDENCE_FOUND,
            "evidence_rows":
                saved_evidence_count,
        }

    # --------------------------------------------------------
    # PDF REPRINT ONLY
    # --------------------------------------------------------

    if (
        finding
        == "pdf_reprint_only"
    ):

        reason = (
            "Known production redesign, but compared official LEGO "
            "PDFs appear to describe the same build or a publication "
            "refresh. Requires deeper research to locate the redesign."
        )

        update_candidate(
            candidate_id,
            STATUS_NEEDS_DEEPER_RESEARCH,
            reason,
        )

        print("")
        print(
            "RESULT: "
            "needs_deeper_research "
            "(PDF reprint only)"
        )

        return {
            "set_num":
                set_num,
            "finding":
                finding,
            "status":
                STATUS_NEEDS_DEEPER_RESEARCH,
            "evidence_rows":
                saved_evidence_count,
        }

    # --------------------------------------------------------
    # AMBIGUOUS
    # --------------------------------------------------------

    reason = (
        "Known production redesign, but current automated PDF "
        "comparison is ambiguous or incomplete. Requires deeper "
        "research; candidate must not be closed."
    )

    update_candidate(
        candidate_id,
        STATUS_NEEDS_DEEPER_RESEARCH,
        reason,
    )

    print("")
    print(
        "RESULT: "
        "needs_deeper_research "
        "(ambiguous PDF evidence)"
    )

    return {
        "set_num":
            set_num,
        "finding":
            "needs_deeper_research",
        "status":
            STATUS_NEEDS_DEEPER_RESEARCH,
        "evidence_rows":
            saved_evidence_count,
    }


def record_pipeline_run(
    status,
    records_processed,
    error_message=None,
):
    """
    Record one worker run in pipeline_runs.
    """
    now = datetime.now(
        timezone.utc
    ).isoformat()

    row = {
        "job_name":
            "research_known_revisions",
        "status":
            status,
        "records_processed":
            records_processed,
        "error_message":
            error_message,
        "started_at":
            now,
        "finished_at":
            now,
    }

    (
        supabase
        .table(
            "pipeline_runs"
        )
        .insert(
            row
        )
        .execute()
    )


def main():
    print("")
    print(
        "BrickTrip Production Revision Research"
    )

    print(
        "======================================"
    )

    print(
        f"Maximum sets this run: "
        f"{MAX_SETS_PER_RUN}"
    )

    candidates = (
        get_candidates()
    )

    print(
        f"Candidates loaded: "
        f"{len(candidates)}"
    )

    if not candidates:

        print(
            "No needs_research candidates remain."
        )

        try:

            record_pipeline_run(
                status="completed",
                records_processed=0,
                error_message=None,
            )

        except Exception as error:

            print(
                f"Could not record pipeline run: "
                f"{error}"
            )

        return

    pdf_cache = {}

    processed = 0

    failed = 0

    revision_evidence_sets = 0

    deeper_research_sets = 0

    evidence_rows_saved = 0

    for candidate in candidates:

        try:

            result = process_candidate(
                candidate,
                pdf_cache,
            )

            processed += 1

            evidence_rows_saved += (
                result[
                    "evidence_rows"
                ]
            )

            if (
                result[
                    "status"
                ]
                == STATUS_REVISION_EVIDENCE_FOUND
            ):

                revision_evidence_sets += 1

            else:

                deeper_research_sets += 1

        except Exception as error:

            failed += 1

            set_num = (
                candidate.get(
                    "set_num"
                )
                or "unknown"
            )

            print("")
            print(
                f"ERROR processing "
                f"{set_num}: "
                f"{error}"
            )

            # Leave a genuinely failed candidate in needs_research
            # so a later run can retry it.

    print("")
    print(
        "=" * 72
    )

    print(
        "RUN SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"Successfully processed: "
        f"{processed}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        f"Revision-evidence sets: "
        f"{revision_evidence_sets}"
    )

    print(
        f"Needs-deeper-research sets: "
        f"{deeper_research_sets}"
    )

    print(
        f"Structural evidence rows saved: "
        f"{evidence_rows_saved}"
    )

    if failed:

        run_status = (
            "completed_with_errors"
        )

        error_message = (
            f"{failed} set(s) failed."
        )

    else:

        run_status = (
            "completed"
        )

        error_message = None

    try:

        record_pipeline_run(
            status=run_status,
            records_processed=processed,
            error_message=error_message,
        )

        print(
            "Pipeline run recorded."
        )

    except Exception as error:

        print(
            f"Could not record pipeline run: "
            f"{error}"
        )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
