import hashlib
import io
import os

import requests
from pypdf import PdfReader
from supabase import create_client


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SECRET_KEY,
)


# ------------------------------------------------------------
# CONTROLLED READ-ONLY PDF TEST
# ------------------------------------------------------------
#
# We are testing LEGO set 10214-1 only.
#
# These document pairs represent the SAME physical booklet slot
# at different publication dates.
#
# Nothing is inserted, updated, or deleted in Supabase.
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


def get_instruction_documents():
    """
    Load only the instruction documents needed for this test.
    """
    document_numbers = []

    for pair in TEST_PAIRS:
        document_numbers.append(
            pair["older_document"]
        )
        document_numbers.append(
            pair["newer_document"]
        )

    document_numbers = sorted(
        set(document_numbers)
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
            document_numbers,
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
    Download one official LEGO instruction PDF into memory.
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
        timeout=120,
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


def whole_file_hash(pdf_bytes):
    """
    Hash the entire downloaded PDF file.

    Different whole-file hashes prove the files are not
    byte-for-byte identical, but that alone does NOT prove the
    building instructions changed.
    """
    return hashlib.sha256(
        pdf_bytes
    ).hexdigest()


def page_content_bytes(page):
    """
    Get the decoded PDF content-stream bytes for one page.

    This focuses on the page's actual drawing/text instructions
    rather than the PDF file's outer metadata.
    """
    contents = page.get_contents()

    if contents is None:
        return b""

    try:
        return contents.get_data()

    except AttributeError:
        pass

    try:
        data_parts = []

        for content in contents:
            data_parts.append(
                content.get_data()
            )

        return b"".join(
            data_parts
        )

    except Exception:
        return b""


def page_content_hash(page):
    """
    Hash the decoded content stream for one PDF page.
    """
    content = page_content_bytes(
        page
    )

    return hashlib.sha256(
        content
    ).hexdigest()


def analyze_pdf(pdf_bytes):
    """
    Read a PDF and calculate page-level content hashes.
    """
    reader = PdfReader(
        io.BytesIO(pdf_bytes)
    )

    page_hashes = []

    for page in reader.pages:
        page_hashes.append(
            page_content_hash(
                page
            )
        )

    return {
        "page_count": len(reader.pages),
        "page_hashes": page_hashes,
        "file_hash": whole_file_hash(
            pdf_bytes
        ),
    }


def compare_page_hashes(
    older_analysis,
    newer_analysis,
):
    """
    Compare same-numbered pages between two PDFs.

    Returns:
    - matching pages
    - changed pages
    - pages that exist only in one PDF
    """
    older_hashes = (
        older_analysis["page_hashes"]
    )

    newer_hashes = (
        newer_analysis["page_hashes"]
    )

    shared_page_count = min(
        len(older_hashes),
        len(newer_hashes),
    )

    matching_pages = []
    changed_pages = []

    for index in range(
        shared_page_count
    ):
        page_number = index + 1

        if (
            older_hashes[index]
            == newer_hashes[index]
        ):
            matching_pages.append(
                page_number
            )
        else:
            changed_pages.append(
                page_number
            )

    older_only_pages = list(
        range(
            shared_page_count + 1,
            len(older_hashes) + 1,
        )
    )

    newer_only_pages = list(
        range(
            shared_page_count + 1,
            len(newer_hashes) + 1,
        )
    )

    return {
        "matching_pages":
            matching_pages,
        "changed_pages":
            changed_pages,
        "older_only_pages":
            older_only_pages,
        "newer_only_pages":
            newer_only_pages,
    }


def summarize_page_list(
    pages,
    maximum_display=40,
):
    """
    Keep GitHub logs readable if many pages differ.
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
    Download and compare one old/new instruction pair.
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
            f"Could not find metadata for "
            f"{older_number}."
        )

    if not newer_document:
        raise RuntimeError(
            f"Could not find metadata for "
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
        older_bytes = download_pdf(
            older_document
        )

        pdf_cache[
            older_number
        ] = analyze_pdf(
            older_bytes
        )

    if newer_number not in pdf_cache:
        newer_bytes = download_pdf(
            newer_document
        )

        pdf_cache[
            newer_number
        ] = analyze_pdf(
            newer_bytes
        )

    older_analysis = pdf_cache[
        older_number
    ]

    newer_analysis = pdf_cache[
        newer_number
    ]

    comparison = compare_page_hashes(
        older_analysis,
        newer_analysis,
    )

    print("")
    print(
        f"Older page count: "
        f'{older_analysis["page_count"]}'
    )

    print(
        f"Newer page count: "
        f'{newer_analysis["page_count"]}'
    )

    same_whole_file = (
        older_analysis["file_hash"]
        == newer_analysis["file_hash"]
    )

    print(
        f"Whole PDFs byte-identical: "
        f"{same_whole_file}"
    )

    matching_count = len(
        comparison["matching_pages"]
    )

    changed_count = len(
        comparison["changed_pages"]
    )

    print(
        f"Same-position matching pages: "
        f"{matching_count}"
    )

    print(
        f"Same-position changed pages: "
        f"{changed_count}"
    )

    print(
        "Changed page numbers: "
        + summarize_page_list(
            comparison[
                "changed_pages"
            ]
        )
    )

    print(
        "Pages only in older PDF: "
        + summarize_page_list(
            comparison[
                "older_only_pages"
            ]
        )
    )

    print(
        "Pages only in newer PDF: "
        + summarize_page_list(
            comparison[
                "newer_only_pages"
            ]
        )
    )

    print("")

    if (
        same_whole_file
        and changed_count == 0
        and not comparison[
            "older_only_pages"
        ]
        and not comparison[
            "newer_only_pages"
        ]
    ):
        verdict = (
            "IDENTICAL PDF"
        )

    elif (
        changed_count == 0
        and not comparison[
            "older_only_pages"
        ]
        and not comparison[
            "newer_only_pages"
        ]
    ):
        verdict = (
            "SAME PAGE CONTENT; "
            "PDF FILE WRAPPER/METADATA DIFFERS"
        )

    else:
        verdict = (
            "PAGE CONTENT DIFFERS — "
            "needs deeper visual/step comparison"
        )

    print(
        f"VERDICT: {verdict}"
    )


def main():
    print("")
    print(
        "BrickTrip Official LEGO PDF Comparison"
    )
    print(
        "======================================"
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
    print("PDF TEST COMPLETE")
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
