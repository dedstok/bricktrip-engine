import os
from pathlib import Path

import pymupdf
import requests
from PIL import Image, ImageDraw, ImageFont
from supabase import create_client


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SECRET_KEY,
)


# ------------------------------------------------------------
# CONTROLLED READ-ONLY VISUAL INSPECTION TEST
# ------------------------------------------------------------
#
# LEGO set 10214-1 only.
#
# This test creates side-by-side PNG images so we can visually
# inspect what changed between older and newer LEGO instructions.
#
# NOTHING is written to Supabase.
# ------------------------------------------------------------

TEST_SET_NUM = "10214-1"


OUTPUT_DIR = Path(
    "artifacts/bricktrip_pdf_comparisons"
)


# Render slightly larger than native PDF resolution so details
# are easy to inspect.
RENDER_SCALE = 1.5


COMPARISONS = [
    {
        "booklet": "2_of_3",
        "label": "Booklet 2/3 - 2011 vs 2015",
        "older_document": "4658001",
        "newer_document": "6145768",
        "pages": [
            2,
            30,
            59,
        ],
    },
    {
        "booklet": "3_of_3",
        "label": "Booklet 3/3 - 2011 vs 2015",
        "older_document": "4658002",
        "newer_document": "6145770",
        "pages": [
            14,
            42,
        ],
    },
    {
        "booklet": "1_of_3",
        "label": "Booklet 1/3 - 2012 vs 2015",
        "older_document": "6020850",
        "newer_document": "6146167",
        "pages": [
            22,
            30,
        ],
    },
]


def get_instruction_documents():
    """
    Fetch metadata only for the instruction documents used in
    this controlled test.
    """
    required_numbers = set()

    for comparison in COMPARISONS:
        required_numbers.add(
            comparison["older_document"]
        )

        required_numbers.add(
            comparison["newer_document"]
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
        document_number = str(
            document.get(
                "document_number"
            )
        )

        documents[
            document_number
        ] = document

    return documents


def download_pdf(document):
    """
    Download one official LEGO PDF into memory.
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


def render_page(pdf, page_number):
    """
    Render one human-numbered PDF page to a Pillow RGB image.

    page_number 1 means PDF page index 0.
    """
    page_index = (
        page_number - 1
    )

    if page_index < 0:
        raise RuntimeError(
            f"Invalid page number "
            f"{page_number}."
        )

    if page_index >= len(pdf):
        raise RuntimeError(
            f"Requested page "
            f"{page_number}, but PDF has "
            f"only {len(pdf)} pages."
        )

    page = pdf[
        page_index
    ]

    matrix = pymupdf.Matrix(
        RENDER_SCALE,
        RENDER_SCALE,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        colorspace=pymupdf.csRGB,
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        (
            pixmap.width,
            pixmap.height,
        ),
        pixmap.samples,
    )

    return image


def fit_image_to_height(
    image,
    target_height,
):
    """
    Resize an image while preserving aspect ratio.
    """
    if image.height == target_height:
        return image

    scale = (
        target_height
        / image.height
    )

    target_width = max(
        1,
        round(
            image.width
            * scale
        ),
    )

    return image.resize(
        (
            target_width,
            target_height,
        ),
        Image.Resampling.LANCZOS,
    )


def get_font():
    """
    Use Pillow's built-in default font.

    No external font files are required.
    """
    return ImageFont.load_default()


def draw_centered_text(
    draw,
    text,
    center_x,
    y,
    font,
):
    """
    Draw simple centered text.
    """
    bbox = draw.textbbox(
        (
            0,
            0,
        ),
        text,
        font=font,
    )

    text_width = (
        bbox[2]
        - bbox[0]
    )

    x = (
        center_x
        - text_width // 2
    )

    draw.text(
        (
            x,
            y,
        ),
        text,
        fill="black",
        font=font,
    )


def create_side_by_side_image(
    older_image,
    newer_image,
    older_document_number,
    newer_document_number,
    page_number,
    comparison_label,
):
    """
    Create one inspection PNG containing:

    OLD PDF | NEW PDF
    """
    target_height = max(
        older_image.height,
        newer_image.height,
    )

    older_image = fit_image_to_height(
        older_image,
        target_height,
    )

    newer_image = fit_image_to_height(
        newer_image,
        target_height,
    )

    outer_margin = 30
    gap = 30
    header_height = 90
    footer_height = 45

    canvas_width = (
        outer_margin
        + older_image.width
        + gap
        + newer_image.width
        + outer_margin
    )

    canvas_height = (
        header_height
        + target_height
        + footer_height
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        "white",
    )

    older_x = outer_margin

    newer_x = (
        outer_margin
        + older_image.width
        + gap
    )

    image_y = header_height

    canvas.paste(
        older_image,
        (
            older_x,
            image_y,
        ),
    )

    canvas.paste(
        newer_image,
        (
            newer_x,
            image_y,
        ),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    font = get_font()

    old_center_x = (
        older_x
        + older_image.width // 2
    )

    new_center_x = (
        newer_x
        + newer_image.width // 2
    )

    draw_centered_text(
        draw,
        "OLDER",
        old_center_x,
        12,
        font,
    )

    draw_centered_text(
        draw,
        f"Document {older_document_number}",
        old_center_x,
        30,
        font,
    )

    draw_centered_text(
        draw,
        "NEWER",
        new_center_x,
        12,
        font,
    )

    draw_centered_text(
        draw,
        f"Document {newer_document_number}",
        new_center_x,
        30,
        font,
    )

    divider_x = (
        older_x
        + older_image.width
        + gap // 2
    )

    draw.line(
        (
            divider_x,
            0,
            divider_x,
            canvas_height,
        ),
        fill="black",
        width=2,
    )

    footer_text = (
        f"{comparison_label} | "
        f"PDF page {page_number}"
    )

    draw_centered_text(
        draw,
        footer_text,
        canvas_width // 2,
        canvas_height - 28,
        font,
    )

    return canvas


def save_comparison_image(
    comparison,
    page_number,
    older_pdf,
    newer_pdf,
):
    """
    Render and save one old/new side-by-side comparison.
    """
    older_document_number = (
        comparison[
            "older_document"
        ]
    )

    newer_document_number = (
        comparison[
            "newer_document"
        ]
    )

    print(
        f"  Rendering page "
        f"{page_number}..."
    )

    older_image = render_page(
        older_pdf,
        page_number,
    )

    newer_image = render_page(
        newer_pdf,
        page_number,
    )

    comparison_image = (
        create_side_by_side_image(
            older_image=older_image,
            newer_image=newer_image,
            older_document_number=(
                older_document_number
            ),
            newer_document_number=(
                newer_document_number
            ),
            page_number=page_number,
            comparison_label=(
                comparison["label"]
            ),
        )
    )

    filename = (
        f'{TEST_SET_NUM}_'
        f'{comparison["booklet"]}_'
        f'page_{page_number:03d}_'
        f'{older_document_number}_vs_'
        f'{newer_document_number}.png'
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    comparison_image.save(
        output_path,
        format="PNG",
        optimize=True,
    )

    print(
        f"    Saved: "
        f"{output_path}"
    )

    return output_path


def process_comparison(
    comparison,
    documents,
    pdf_byte_cache,
):
    """
    Generate all requested inspection images for one booklet pair.
    """
    older_number = (
        comparison[
            "older_document"
        ]
    )

    newer_number = (
        comparison[
            "newer_document"
        ]
    )

    print("")
    print("=" * 72)

    print(
        comparison["label"]
    )

    print(
        f"{older_number} -> "
        f"{newer_number}"
    )

    print("=" * 72)

    older_document = (
        documents.get(
            older_number
        )
    )

    newer_document = (
        documents.get(
            newer_number
        )
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

    older_pdf = open_pdf(
        pdf_byte_cache[
            older_number
        ]
    )

    newer_pdf = open_pdf(
        pdf_byte_cache[
            newer_number
        ]
    )

    try:
        print(
            f"Older pages: "
            f"{len(older_pdf)}"
        )

        print(
            f"Newer pages: "
            f"{len(newer_pdf)}"
        )

        generated_paths = []

        for page_number in (
            comparison["pages"]
        ):
            output_path = (
                save_comparison_image(
                    comparison=comparison,
                    page_number=page_number,
                    older_pdf=older_pdf,
                    newer_pdf=newer_pdf,
                )
            )

            generated_paths.append(
                output_path
            )

        return generated_paths

    finally:
        older_pdf.close()
        newer_pdf.close()


def main():
    print("")
    print(
        "BrickTrip LEGO Instruction Visual Inspection"
    )
    print(
        "============================================"
    )

    print(
        f"Controlled test set: "
        f"{TEST_SET_NUM}"
    )

    print(
        "READ ONLY — database writes: NONE"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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

    generated_files = []

    succeeded = 0
    failed = 0

    for comparison in COMPARISONS:
        try:
            paths = process_comparison(
                comparison=comparison,
                documents=documents,
                pdf_byte_cache=pdf_byte_cache,
            )

            generated_files.extend(
                paths
            )

            succeeded += 1

        except Exception as error:
            failed += 1

            print("")
            print(
                f"ERROR processing "
                f'{comparison["label"]}: '
                f"{error}"
            )

    print("")
    print("=" * 72)
    print(
        "VISUAL INSPECTION TEST COMPLETE"
    )

    print(
        f"Booklet comparisons succeeded: "
        f"{succeeded}"
    )

    print(
        f"Booklet comparisons failed: "
        f"{failed}"
    )

    print(
        f"PNG files generated: "
        f"{len(generated_files)}"
    )

    print("")
    print(
        "Generated files:"
    )

    for path in generated_files:
        print(
            f"  {path}"
        )

    print("")
    print(
        "Database writes: 0"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
