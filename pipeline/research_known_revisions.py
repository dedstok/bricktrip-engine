import os
import re
from collections import defaultdict
from datetime import datetime
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SECRET_KEY
)

# Safety limit while we test the worker.
MAX_SETS_PER_RUN = 10


def parse_version(description):
    """
    Pulls LEGO instruction variant labels such as V29, V39, etc.
    """
    if not description:
        return None

    matches = re.findall(r"\bV\d+\b", description.upper())

    if not matches:
        return None

    return matches[-1]


def is_main_instruction(document):
    """
    Ignore obvious translation/support PDFs.

    We only want primary building instructions during this
    first research pass.
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
        if term in description or term in source_url:
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


def document_sort_date(document):
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

    return datetime.min


def get_known_redesign_queue():
    """
    Fetch known redesigns that still need research.
    """
    response = (
        supabase
        .table("revision_candidates")
        .select("id,set_num,status")
        .eq("status", "needs_research")
        .limit(MAX_SETS_PER_RUN)
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


def group_instruction_generations(documents):
    """
    Group comparable instructions by LEGO variant.

    Example:

    6509262 V29
    6562097 V29

    become one comparison group.

    Likewise V39 is compared only against V39.
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

        groups[version].append(document)

    for version in groups:
        groups[version].sort(
            key=document_sort_date
        )

    return groups


def build_comparison_pairs(groups):
    """
    Build old -> new comparison pairs inside each
    matching instruction variant.
    """
    pairs = []

    for version, documents in groups.items():

        if len(documents) < 2:
            continue

        for index in range(len(documents) - 1):

            older = documents[index]
            newer = documents[index + 1]

            pairs.append({
                "version": version,
                "older": older,
                "newer": newer,
            })

    return pairs


def summarize_pair(pair):
    older = pair["older"]
    newer = pair["newer"]

    older_date = (
        older.get("source_date_modified")
        or older.get("source_date_added")
    )

    newer_date = (
        newer.get("source_date_modified")
        or newer.get("source_date_added")
    )

    return (
        f'{pair["version"]}: '
        f'{older.get("document_number")} '
        f'({older_date}) -> '
        f'{newer.get("document_number")} '
        f'({newer_date})'
    )


def save_evidence(set_num, pairs):
    """
    Save what the worker discovered.

    This does NOT claim that a revision has been verified.
    It records instruction-generation evidence for later
    PDF comparison.
    """

    for pair in pairs:

        description = (
            "Possible instruction generation change. "
            + summarize_pair(pair)
        )

        newer_url = (
            pair["newer"].get("source_url")
        )

        # Avoid adding the exact same evidence repeatedly.
        existing = (
            supabase
            .table("evidence")
            .select("id")
            .eq("set_num", set_num)
            .eq(
                "source_type",
                "instruction_generation"
            )
            .eq("description", description)
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
                "source_url": newer_url,
                "description": description,
                "confidence": 0.50,
            })
            .execute()
        )


def update_candidate(candidate_id, pairs):
    """
    Mark this queue item according to what this
    metadata pass discovered.
    """

    if pairs:
        status = "instruction_pairs_found"
        reason = (
            f"Known redesign. Found "
            f"{len(pairs)} comparable instruction "
            f"generation pair(s). Ready for PDF analysis."
        )
    else:
        status = "manual_or_deeper_research"
        reason = (
            "Known redesign, but no comparable "
            "instruction generations were found "
            "from current metadata."
        )

    (
        supabase
        .table("revision_candidates")
        .update({
            "status": status,
            "reason": reason,
            "updated_at": datetime.now().isoformat(),
        })
        .eq("id", candidate_id)
        .execute()
    )


def research_set(candidate):
    set_num = candidate["set_num"]

    print("")
    print("=" * 60)
    print(f"Researching {set_num}")
    print("=" * 60)

    documents = get_instruction_documents(
        set_num
    )

    print(
        f"Instruction records: {len(documents)}"
    )

    groups = group_instruction_generations(
        documents
    )

    print(
        f"Comparable variant groups: "
        f"{len(groups)}"
    )

    pairs = build_comparison_pairs(groups)

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
    print("BrickTrip Known Revision Research Worker")
    print("========================================")
    print(
        f"Maximum sets this run: "
        f"{MAX_SETS_PER_RUN}"
    )

    queue = get_known_redesign_queue()

    print(
        f"Queue items loaded: {len(queue)}"
    )

    processed = 0
    failed = 0

    for candidate in queue:

        try:
            research_set(candidate)
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
            "records_processed": processed,
            "records_created": 0,
            "error_message":
                None
                if failed == 0
                else f"{failed} set(s) failed.",
            "finished_at":
                datetime.now().isoformat(),
        })
        .execute()
    )

    print("")
    print("========================================")
    print("RUN COMPLETE")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print("========================================")


if __name__ == "__main__":
    main()
