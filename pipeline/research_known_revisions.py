import os
import re
from collections import defaultdict
from datetime import datetime, timezone

from supabase import create_client


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SECRET_KEY,
)


# ------------------------------------------------------------
# CONTROLLED TEST ONLY
# ------------------------------------------------------------
#
# This worker intentionally ignores revision_candidates status.
#
# It reads ONLY these known test sets and PRINTS its findings.
#
# It does NOT:
# - insert evidence
# - update revision_candidates
# - create revisions
# - modify any BrickTrip data
#
# We are validating the instruction-matching logic before
# allowing the normal unattended queue to resume.
# ------------------------------------------------------------

TEST_SET_NUMBERS = [
    "42171-1",
    "42129-1",
    "60004-1",
    "60085-1",
    "10194-1",
    "10214-1",
    "31058-1",
]


def parse_date(value):
    """
    Convert a Supabase timestamp string into a Python datetime.
    """
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return None


def document_effective_date(document):
    """
    Prefer LEGO/Brickset's modification date.

    Fall back to the date the document was added.
    """
    modified = parse_date(
        document.get("source_date_modified")
    )

    if modified:
        return modified

    added = parse_date(
        document.get("source_date_added")
    )

    if added:
        return added

    return None


def document_sort_date(document):
    """
    Always return a timezone-aware date so documents can
    safely be sorted.
    """
    effective_date = document_effective_date(
        document
    )

    if effective_date:
        return effective_date

    return datetime.min.replace(
        tzinfo=timezone.utc
    )


def generation_date_key(document):
    """
    Use the effective calendar date as the publication-date key.

    Multiple documents on the same date may represent regional
    versions of the same instruction publication.
    """
    effective_date = document_effective_date(
        document
    )

    if not effective_date:
        return None

    return effective_date.date().isoformat()


def is_main_instruction(document):
    """
    Ignore obvious translation, support, and alternate-model PDFs.

    During revision research we want the main building instruction
    documents, not translation sheets or LEGO's additional/extra
    downloadable models.
    """
    description = (
        document.get("description") or ""
    ).lower()

    source_url = (
        document.get("source_url") or ""
    ).lower()

    ignore_terms = [
        "translate",
        "translation",
        "additional.main",
        "additional.extra",
    ]

    for term in ignore_terms:
        if (
            term in description
            or term in source_url
        ):
            return False

    return True


def parse_versions(description):
    """
    Return every LEGO V-label present in a description.

    Examples:

    "42171 V29"
        -> ("V29",)

    "42129 V29/V118"
        -> ("V29", "V118")

    "10214 V46/39"
        -> ("V46", "V39")

    LEGO sometimes writes a later version after a slash without
    repeating the V, so V46/39 means V46 and V39.
    """
    if not description:
        return tuple()

    text = description.upper()

    versions = []

    explicit_matches = re.findall(
        r"\bV(\d+)(?:\s*/\s*(\d+))?",
        text,
    )

    for first_number, second_number in explicit_matches:

        first_version = f"V{first_number}"

        if first_version not in versions:
            versions.append(first_version)

        if second_number:
            second_version = f"V{second_number}"

            if second_version not in versions:
                versions.append(second_version)

    return tuple(versions)


def parse_booklet_slot(description):
    """
    Find a genuine booklet position such as:

    1/2
    2/3
    4/5
    BOOK 1/3
    BOOK2/3

    LEGO descriptions also contain printing/page codes such as:

    BI 3004/60
    264+4/65+200G

    Those must NOT be mistaken for booklet positions.

    For this research pass, a valid booklet fraction must satisfy:

    - numerator >= 1
    - denominator >= 2
    - numerator <= denominator
    - denominator <= 20

    That safely captures the booklet structures we have inspected
    while rejecting the large printing/page fractions.
    """
    if not description:
        return None

    text = description.upper()

    matches = re.finditer(
        r"(?<!\d)"
        r"(\d{1,2})"
        r"\s*/\s*"
        r"(\d{1,2})"
        r"(?!\d)",
        text,
    )

    candidates = []

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
                match.start(),
            )
        )

    if not candidates:
        return None

    # A LEGO BI printing code normally appears near the beginning
    # of the description.
    #
    # Real booklet fractions generally occur later, around the set
    # number / V-label portion.
    #
    # If several small fractions somehow exist, use the last one.
    numerator, denominator, _ = candidates[-1]

    return f"{numerator}/{denominator}"


def get_instruction_documents(set_num):
    """
    Fetch all known instruction-document metadata for one LEGO set.
    """
    response = (
        supabase
        .table("instruction_documents")
        .select(
            "id,"
            "set_num,"
            "document_number,"
            "description,"
            "source_url,"
            "source_date_added,"
            "source_date_modified"
        )
        .eq("set_num", set_num)
        .execute()
    )

    documents = response.data or []

    documents.sort(
        key=document_sort_date
    )

    return documents


def build_publication_generations(documents):
    """
    Collapse documents sharing the same effective calendar date
    into one publication generation.
    """
    generations_by_date = defaultdict(list)

    for document in documents:

        date_key = generation_date_key(
            document
        )

        if date_key is None:
            continue

        generations_by_date[
            date_key
        ].append(
            document
        )

    generations = []

    for date_key, generation_documents in (
        generations_by_date.items()
    ):

        generation_documents.sort(
            key=lambda document: str(
                document.get("document_number")
                or ""
            )
        )

        generations.append({
            "date_key": date_key,
            "documents": generation_documents,
        })

    generations.sort(
        key=lambda generation:
        generation["date_key"]
    )

    return generations


def document_numbers(documents):
    """
    Return printable document numbers.
    """
    numbers = []

    for document in documents:

        number = document.get(
            "document_number"
        )

        if number is not None:
            numbers.append(
                str(number)
            )

    return numbers


def versions_in_documents(documents):
    """
    Return every V-label represented by a group of documents.
    """
    versions = []

    for document in documents:

        parsed_versions = parse_versions(
            document.get("description")
        )

        for version in parsed_versions:

            if version not in versions:
                versions.append(
                    version
                )

    return versions


def choose_representative_document(documents):
    """
    Pick one deterministic document from a group.

    This is only for displaying a representative URL/document
    during testing.
    """
    if not documents:
        return None

    ordered = sorted(
        documents,
        key=lambda document: str(
            document.get("document_number")
            or ""
        ),
    )

    return ordered[0]


def build_booklet_groups(documents):
    """
    Group documents by physical booklet position.

    Example:

    1/3 -> all known PDFs representing booklet 1 of 3
    2/3 -> all known PDFs representing booklet 2 of 3
    3/3 -> all known PDFs representing booklet 3 of 3

    Documents without a recognizable booklet slot are kept
    separately.
    """
    booklet_groups = defaultdict(list)
    unknown_slot_documents = []

    for document in documents:

        if not is_main_instruction(
            document
        ):
            continue

        slot = parse_booklet_slot(
            document.get("description")
        )

        if slot:
            booklet_groups[
                slot
            ].append(
                document
            )
        else:
            unknown_slot_documents.append(
                document
            )

    for slot in booklet_groups:

        booklet_groups[
            slot
        ].sort(
            key=document_sort_date
        )

    unknown_slot_documents.sort(
        key=document_sort_date
    )

    return (
        booklet_groups,
        unknown_slot_documents,
    )


def build_multi_booklet_pairs(booklet_groups):
    """
    Compare only LIKE-FOR-LIKE booklet positions.

    1/3 may compare to a later 1/3.

    1/3 must NEVER be compared with 2/3 simply because their
    publication dates differ.

    Each adjacent publication date inside the same booklet slot
    becomes a PDF-comparison candidate.
    """
    pairs = []

    for slot, documents in sorted(
        booklet_groups.items()
    ):

        generations = (
            build_publication_generations(
                documents
            )
        )

        if len(generations) < 2:
            continue

        for index in range(
            len(generations) - 1
        ):

            older_generation = (
                generations[index]
            )

            newer_generation = (
                generations[index + 1]
            )

            pairs.append({
                "mode": "booklet_slot",
                "identity": slot,
                "older_generation":
                    older_generation,
                "newer_generation":
                    newer_generation,
            })

    return pairs


def build_version_groups(documents):
    """
    For documents without booklet slots, group by LEGO V-label.

    This handles single-booklet sets such as 42171.

    It also catches cases like 42129 where one regional V-label
    survives from the earlier publication into the later one.

    A document containing V29/V118 belongs to both the V29 and
    V118 groups.
    """
    version_groups = defaultdict(list)

    for document in documents:

        versions = parse_versions(
            document.get("description")
        )

        for version in versions:

            version_groups[
                version
            ].append(
                document
            )

    for version in version_groups:

        version_groups[
            version
        ].sort(
            key=document_sort_date
        )

    return version_groups


def build_single_booklet_pairs(documents):
    """
    For sets/documents without booklet positions, compare repeated
    V-labels across distinct publication dates.

    This is intentionally conservative.

    We do NOT compare completely unrelated V-labels just because
    their dates differ.
    """
    pairs = []

    version_groups = (
        build_version_groups(
            documents
        )
    )

    for version, version_documents in sorted(
        version_groups.items()
    ):

        generations = (
            build_publication_generations(
                version_documents
            )
        )

        if len(generations) < 2:
            continue

        for index in range(
            len(generations) - 1
        ):

            older_generation = (
                generations[index]
            )

            newer_generation = (
                generations[index + 1]
            )

            pairs.append({
                "mode": "version",
                "identity": version,
                "older_generation":
                    older_generation,
                "newer_generation":
                    newer_generation,
            })

    return pairs


def pair_key(pair):
    """
    Create a stable identity for deduplicating comparison pairs.

    Two regional V-labels may point to the exact same old/new
    publication dates. We keep their version-specific pairs for
    this diagnostic run because seeing them is useful.

    Multi-booklet pairs remain separated by booklet slot.
    """
    return (
        pair["mode"],
        pair["identity"],
        pair["older_generation"][
            "date_key"
        ],
        pair["newer_generation"][
            "date_key"
        ],
    )


def deduplicate_pairs(pairs):
    """
    Remove accidental exact duplicate pair records.
    """
    unique_pairs = []
    seen = set()

    for pair in pairs:

        key = pair_key(
            pair
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique_pairs.append(
            pair
        )

    return unique_pairs


def summarize_generation(generation):
    """
    Produce a readable publication-generation summary.
    """
    documents = (
        generation["documents"]
    )

    numbers = document_numbers(
        documents
    )

    versions = versions_in_documents(
        documents
    )

    number_text = (
        ", ".join(numbers)
        if numbers
        else "unknown document"
    )

    version_text = (
        ", ".join(versions)
        if versions
        else "no V-label"
    )

    return (
        f'{generation["date_key"]} '
        f'[{number_text}] '
        f'({version_text})'
    )


def summarize_pair(pair):
    """
    Produce a readable old -> new PDF comparison candidate.
    """
    older_text = summarize_generation(
        pair["older_generation"]
    )

    newer_text = summarize_generation(
        pair["newer_generation"]
    )

    if pair["mode"] == "booklet_slot":
        label = (
            f'booklet {pair["identity"]}'
        )
    else:
        label = (
            f'version {pair["identity"]}'
        )

    return (
        f"{label}: "
        f"{older_text} -> {newer_text}"
    )


def print_document_inventory(documents):
    """
    Print every primary instruction document with the metadata
    parsed by the new algorithm.
    """
    print("")
    print("Primary instruction documents:")

    primary_count = 0

    for document in documents:

        if not is_main_instruction(
            document
        ):
            continue

        primary_count += 1

        description = (
            document.get("description")
            or ""
        )

        number = (
            document.get("document_number")
            or "unknown"
        )

        date_key = generation_date_key(
            document
        )

        slot = parse_booklet_slot(
            description
        )

        versions = parse_versions(
            description
        )

        slot_text = (
            slot
            if slot
            else "none"
        )

        version_text = (
            ", ".join(versions)
            if versions
            else "none"
        )

        print(
            f"  {number}"
        )

        print(
            f"    date: {date_key}"
        )

        print(
            f"    booklet slot: {slot_text}"
        )

        print(
            f"    versions: {version_text}"
        )

        print(
            f"    description: {description}"
        )

    print(
        f"Primary document count: "
        f"{primary_count}"
    )


def print_booklet_structure(
    booklet_groups,
    unknown_slot_documents,
):
    """
    Print how the new parser understands the physical booklet
    structure for this set.
    """
    print("")
    print("Parsed instruction structure:")

    if booklet_groups:

        print(
            f"  Recognized booklet slots: "
            f"{len(booklet_groups)}"
        )

        for slot, documents in sorted(
            booklet_groups.items()
        ):

            generations = (
                build_publication_generations(
                    documents
                )
            )

            print(
                f"  {slot}: "
                f"{len(documents)} document(s), "
                f"{len(generations)} dated "
                f"publication generation(s)"
            )

            for generation in generations:

                print(
                    "    "
                    + summarize_generation(
                        generation
                    )
                )

    else:
        print(
            "  No explicit booklet slots found."
        )

    print(
        f"  Documents with no booklet slot: "
        f"{len(unknown_slot_documents)}"
    )


def research_test_set(set_num):
    """
    Analyze one controlled test set.

    READ ONLY.
    """
    print("")
    print("=" * 72)
    print(
        f"TESTING {set_num}"
    )
    print("=" * 72)

    documents = (
        get_instruction_documents(
            set_num
        )
    )

    print(
        f"Instruction records fetched: "
        f"{len(documents)}"
    )

    print_document_inventory(
        documents
    )

    (
        booklet_groups,
        unknown_slot_documents,
    ) = build_booklet_groups(
        documents
    )

    print_booklet_structure(
        booklet_groups,
        unknown_slot_documents,
    )

    pairs = []

    # --------------------------------------------------------
    # MULTI-BOOKLET LOGIC
    # --------------------------------------------------------
    #
    # If explicit booklet positions exist, compare only the same
    # physical booklet slot across publication dates.
    # --------------------------------------------------------

    if booklet_groups:

        multi_booklet_pairs = (
            build_multi_booklet_pairs(
                booklet_groups
            )
        )

        pairs.extend(
            multi_booklet_pairs
        )

        # Documents lacking booklet markers are deliberately NOT
        # matched against numbered booklet slots.
        #
        # If there are enough unnumbered documents to establish
        # repeated V-labels safely, we may still inspect those
        # independently.

        unknown_pairs = (
            build_single_booklet_pairs(
                unknown_slot_documents
            )
        )

        pairs.extend(
            unknown_pairs
        )

    # --------------------------------------------------------
    # SINGLE-BOOKLET / NO-SLOT LOGIC
    # --------------------------------------------------------

    else:

        pairs.extend(
            build_single_booklet_pairs(
                unknown_slot_documents
            )
        )

    pairs = deduplicate_pairs(
        pairs
    )

    print("")
    print(
        "PDF comparison candidates: "
        f"{len(pairs)}"
    )

    if not pairs:

        print(
            "  None found from current metadata."
        )

    for pair in pairs:

        print(
            "  "
            + summarize_pair(
                pair
            )
        )

    print("")
    print(
        "DATABASE WRITES: NONE"
    )


def main():
    print("")
    print(
        "BrickTrip Revision Matching Diagnostic"
    )
    print(
        "======================================"
    )
    print(
        "READ-ONLY CONTROLLED TEST"
    )
    print(
        "No evidence or candidate statuses "
        "will be changed."
    )
    print("")
    print(
        f"Test sets: "
        f"{len(TEST_SET_NUMBERS)}"
    )

    succeeded = 0
    failed = 0

    for set_num in TEST_SET_NUMBERS:

        try:

            research_test_set(
                set_num
            )

            succeeded += 1

        except Exception as error:

            failed += 1

            print("")
            print(
                f"ERROR testing "
                f"{set_num}: "
                f"{error}"
            )

    print("")
    print("=" * 72)
    print("DIAGNOSTIC COMPLETE")
    print(
        f"Succeeded: {succeeded}"
    )
    print(
        f"Failed: {failed}"
    )
    print(
        "Database writes: 0"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
