import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SECRET_KEY
)

# TEMPORARY TEST:
# Research only the four candidates previously touched by the old logic.
TEST_CANDIDATE_IDS = [22, 15, 24, 4]


def parse_version(description):
    """
    Pull LEGO instruction variant labels such as V29, V39, etc.
    """
    if not description:
        return None

    matches = re.findall(
        r"\bV\d+\b",
        description.upper()
    )

    if not matches:
        return None

    return matches[-1]


def is_main_instruction(document):
    """
    Ignore obvious translation/support PDFs.

    We only want primary building instructions during this
    research pass.
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
    ]

    for term in ignore_terms:
        if (
            term in description
            or term in source_url
        ):
            return False

    return True


def parse_date(value):
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
    Prefer LEGO/Brickset modification date.
    Fall back to date added.
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
    Return a timezone-aware value so dated and undated
    documents can always be sorted safely.
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
    Documents with the same effective calendar date are treated
    as belonging to the same publication generation.

    This prevents multiple LEGO PDFs published/modified on the
    same day from being mistaken for separate redesign events.
    """
    effective_date = document_effective_date(
        document
    )

    if not effective_date:
        return None

    return effective_date.date().isoformat()


def get_known_redesign_queue():
    """
    TEMPORARY TEST:
    Fetch only the four known candidates we want to retest.
    """
    response = (
        supabase
        .table("revision_candidates")
        .select("id,set_num,status")
        .in_("id", TEST_CANDIDATE_IDS)
        .eq("status", "needs_research")
        .execute()
    )

    return response.data or []


def get_instruction_documents(set_num):
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

    return response.data or []


def group_instruction_variants(documents):
    """
    Group primary instruction documents by LEGO variant.

    V29 documents are compared only with V29 documents.
    V39 documents are compared only with V39 documents.
    """
    groups = defaultdict(list)

    for document in documents:

        if not is_main_instruction(document):
            continue

        version = parse_version(
            document.get("description")
        )

        if not version:
            continue

        groups[version].append(
            document
        )

    for version in groups:
        groups[version].sort(
            key=document_sort_date
        )

    return groups


def build_publication_generations(documents):
    """
    Collapse documents from the same effective calendar date
    into one publication generation.

    A generation may contain more than one document number.
    That does not automatically mean multiple physical set
    redesigns.
    """
    generations_by_date = defaultdict(list)

    for document in documents:

        date_key = generation_date_key(
            document
        )

        if date_key is None:
            continue

        generations_by_date[date_key].append(
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


def choose_generation_representative(generation):
    """
    Pick one stable representative document for a publication
    generation.
    """
    documents = generation["documents"]

    if not documents:
        return None

    return documents[0]


def build_comparison_pairs(variant_groups):
    """
    Build old -> new comparison pairs between DISTINCT dated
    publication generations inside each matching LEGO variant.

    Multiple PDFs sharing the same effective date are treated
    as one generation rather than being chained together.
    """
    pairs = []

    for version, documents in (
        variant_groups.items()
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

            older = (
                choose_generation_representative(
                    older_generation
                )
            )

            newer = (
                choose_generation_representative(
                    newer_generation
                )
            )

            if not older or not newer:
                continue

            pairs.append({
                "version": version,
                "older": older,
                "newer": newer,
                "older_generation":
                    older_generation,
                "newer_generation":
                    newer_generation,
            })

    return pairs


def generation_document_numbers(generation):
    """
    Return all document numbers found inside one publication
    generation.
    """
    numbers = []

    for document in generation["documents"]:
        number = document.get(
            "document_number"
        )

        if number is not None:
            numbers.append(
                str(number)
            )

    return numbers


def summarize_pair(pair):
    older_generation = (
        pair["older_generation"]
    )

    newer_generation = (
        pair["newer_generation"]
    )

    older_numbers = (
        generation_document_numbers(
            older_generation
        )
    )

    newer_numbers = (
        generation_document_numbers(
            newer_generation
        )
    )

    older_number_text = (
        ", ".join(older_numbers)
        if older_numbers
        else "unknown document"
    )

    newer_number_text = (
        ", ".join(newer_numbers)
        if newer_numbers
        else "unknown document"
    )

    return (
        f'{pair["version"]}: '
        f'{older_generation["date_key"]} '
        f'[{older_number_text}] -> '
        f'{newer_generation["date_key"]} '
        f'[{newer_number_text}]'
    )


def save_evidence(set_num, pairs):
    """
    Save what the worker discovered.

    This does NOT claim that a physical revision has been
    verified.

    It only records that distinct dated instruction publication
    generations exist and are ready for later official-PDF
    comparison.
    """
    for pair in pairs:

        description = (
            "Possible instruction generation change. "
            + summarize_pair(pair)
        )

        newer_url = (
            pair["newer"].get(
                "source_url"
            )
        )

        existing = (
            supabase
            .table("evidence")
            .select("id")
            .eq("set_num", set_num)
            .eq(
                "source_type",
                "instruction_generation"
            )
            .eq(
                "description",
                description
            )
            .execute()
        )

        if existing.data:
            continue

        (
            supabase
            .table("evidence")
            .insert({
                "set_num": set_num,
                "source_type":
                    "instruction_generation",
                "source_url":
                    newer_url,
                "description":
                    description,
                "confidence":
                    0.50,
            })
            .execute()
        )


def update_candidate(
    candidate_id,
    pairs
):
    """
    Mark this queue item according to what this metadata pass
    discovered.
    """
    if pairs:
        status = (
            "instruction_pairs_found"
        )

        reason = (
            f"Known redesign. Found "
            f"{len(pairs)} distinct dated "
            f"instruction generation pair(s). "
            f"Ready for PDF analysis."
        )
    else:
        status = (
            "manual_or_deeper_research"
        )

        reason = (
            "Known redesign, but no comparable "
            "distinct dated instruction generations "
            "were found from current metadata."
        )

    (
        supabase
        .table("revision_candidates")
        .update({
            "status": status,
            "reason": reason,
            "updated_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        })
        .eq(
            "id",
            candidate_id
        )
        .execute()
    )


def research_set(candidate):
    set_num = candidate["set_num"]

    print("")
    print("=" * 60)
    print(
        f"Researching {set_num}"
    )
    print("=" * 60)

    documents = (
        get_instruction_documents(
            set_num
        )
    )

    print(
        f"Instruction records: "
        f"{len(documents)}"
    )

    variant_groups = (
        group_instruction_variants(
            documents
        )
    )

    print(
        f"Comparable variant groups: "
        f"{len(variant_groups)}"
    )

    for version, variant_documents in (
        variant_groups.items()
    ):
        generations = (
            build_publication_generations(
                variant_documents
            )
        )

        print(
            f"  {version}: "
            f"{len(variant_documents)} document(s), "
            f"{len(generations)} dated "
            f"generation(s)"
        )

    pairs = build_comparison_pairs(
        variant_groups
    )

    print(
        f"Generation pairs found: "
        f"{len(pairs)}"
    )

    for pair in pairs:
        print(
            "  "
            + summarize_pair(pair)
        )

    save_evidence(
        set_num,
        pairs
    )

    update_candidate(
        candidate["id"],
        pairs
    )


def main():
    print("")
    print(
        "BrickTrip Known Revision Research Worker"
    )
    print(
        "========================================"
    )
    print(
        "TEMPORARY TARGETED TEST"
    )

    queue = (
        get_known_redesign_queue()
    )

    print(
        f"Queue items loaded: "
        f"{len(queue)}"
    )

    processed = 0
    failed = 0

    for candidate in queue:

        try:
            research_set(
                candidate
            )

            processed += 1

        except Exception as error:
            failed += 1

            print("")
            print(
                f"ERROR researching "
                f'{candidate["set_num"]}: '
                f"{error}"
            )

    (
        supabase
        .table("pipeline_runs")
        .insert({
            "job_name":
                "research_known_revisions",
            "status":
                "success"
                if failed == 0
                else "completed_with_errors",
            "records_processed":
                processed,
            "records_created":
                0,
            "error_message":
                None
                if failed == 0
                else (
                    f"{failed} set(s) "
                    f"failed."
                ),
            "finished_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        })
        .execute()
    )

    print("")
    print(
        "========================================"
    )
    print("RUN COMPLETE")
    print(
        f"Processed: {processed}"
    )
    print(
        f"Failed: {failed}"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
