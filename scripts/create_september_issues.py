"""
Create the September milestone, one Issue per task, and add them to the
team's GitHub Project board (Task #3).

Safe to re-run: existing milestones, issues (matched by "[Task #N]" in the
title) and project items are reused, not duplicated.

Requirements
------------
- GitHub CLI logged in with access to the repo:
      gh auth login
      gh auth refresh -s project      # needed to add items to the board
- Run from the repo root:
      python scripts/create_september_issues.py --dry-run
      python scripts/create_september_issues.py

Assignees: fill GITHUB_HANDLES below (name -> GitHub username). Names left
blank are still listed in the issue body, just not assigned.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = "Break-Through-Tech/Banking-1A-online-payment-fraud-detection-using-high-dimensional-transaction-data"
ORG = "Break-Through-Tech"
PROJECT_NUMBER = 168
MILESTONE_TITLE = "September: Data understanding, preparation, and baselines"
MILESTONE_DUE = "2026-09-30T23:59:59Z"
MILESTONE_DESC = (
    "Complete data understanding, preparation, and baseline model development "
    "(Milestone #1, Banking 1A)."
)
# Label scheme: area (what kind of work), priority (P0 = blocks other tasks,
# P1 = milestone deliverable, P2 = supporting), plus risk flags.
LABEL_COLORS = {
    "milestone-1": ("1D76DB", "September milestone tasks"),
    "priority:P0": ("B60205", "Blocks other tasks; do first"),
    "priority:P1": ("D93F0B", "Milestone deliverable"),
    "priority:P2": ("FBCA04", "Supporting work"),
    "area:data": ("0E8A16", "Download, merge, storage, split"),
    "area:infra": ("5319E7", "Repo, board, registry, conventions"),
    "area:eda": ("C2E0C6", "Exploratory analysis"),
    "area:features": ("BFD4F2", "Encoding and feature engineering"),
    "area:evaluation": ("006B75", "Metrics, plots, capture tables"),
    "area:modeling": ("0052CC", "Model training"),
    "area:docs": ("D4C5F9", "Write-ups and research"),
    "area:review": ("F9D0C4", "Milestone review and handoff"),
    "leakage-risk": ("E99695", "Easy to leak future or label information; review carefully"),
    "gates-october": ("FEF2C0", "Outcome decides October scope"),
    "reproducibility": ("C5DEF5", "Needed so results can be rebuilt in November"),
}

# Fill in GitHub usernames so issues get assigned.
GITHUB_HANDLES = {
    "Mohit": "",
    "Leonel": "",
    "Nicholas": "",
    "Refilwe": "",
    "Faizah": "",
    "Orator": "",
}

TASKS_FILE = Path(__file__).with_name("september_tasks.json")


def gh(*args: str, check: bool = True) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit(f"gh {' '.join(args)}\n{result.stderr.strip()}")
    return result.stdout


def get_or_create_milestone(dry: bool) -> int | None:
    existing = json.loads(gh("api", f"repos/{REPO}/milestones?state=all&per_page=100"))
    for m in existing:
        if m["title"] == MILESTONE_TITLE:
            print(f"Milestone exists: #{m['number']}")
            return m["number"]
    if dry:
        print(f"[dry-run] would create milestone '{MILESTONE_TITLE}'")
        return None
    created = json.loads(gh(
        "api", f"repos/{REPO}/milestones", "-X", "POST",
        "-f", f"title={MILESTONE_TITLE}",
        "-f", f"description={MILESTONE_DESC}",
        "-f", f"due_on={MILESTONE_DUE}",
    ))
    print(f"Created milestone #{created['number']}")
    return created["number"]


def ensure_labels(needed: set[str], dry: bool) -> None:
    have = {l["name"] for l in json.loads(gh("api", f"repos/{REPO}/labels?per_page=100"))}
    for name in sorted(needed - have):
        color, desc = LABEL_COLORS.get(name, ("EDEDED", ""))
        if dry:
            print(f"[dry-run] would create label {name}")
            continue
        gh("api", f"repos/{REPO}/labels", "-X", "POST",
           "-f", f"name={name}", "-f", f"color={color}", "-f", f"description={desc}")


def existing_issues() -> dict[int, dict]:
    issues = json.loads(gh("api", f"repos/{REPO}/issues?state=all&per_page=100"))
    found = {}
    for i in issues:
        if "pull_request" in i:
            continue
        for n in range(1, 22):
            if f"[Task #{n}]" in i["title"]:
                found[n] = i
    return found


def issue_body(t: dict) -> str:
    owners = ", ".join(t["owners"])
    return (
        f"{t['body']}\n\n"
        f"**Owner(s):** {owners}\n"
        f"**Target date:** {t['due']}\n"
        f"**Milestone:** #1 (September), Banking 1A\n"
    )


def project_status_field() -> tuple[str, dict[str, str]] | None:
    """Return (field_id, {option name: option id}) for the board's Status field."""
    out = gh("project", "field-list", str(PROJECT_NUMBER), "--owner", ORG, "--format", "json", check=False)
    if not out:
        return None
    for f in json.loads(out).get("fields", []):
        if f.get("name") == "Status":
            return f["id"], {o["name"].lower(): o["id"] for o in f.get("options", [])}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-project", action="store_true", help="Only create issues")
    args = ap.parse_args()
    dry = args.dry_run

    tasks = json.loads(TASKS_FILE.read_text())
    milestone = get_or_create_milestone(dry)
    ensure_labels({l for t in tasks for l in t["labels"]}, dry)
    have = existing_issues()

    project_id = None
    status_field = None
    if not args.skip_project and not dry:
        proj = json.loads(gh("project", "view", str(PROJECT_NUMBER), "--owner", ORG, "--format", "json"))
        project_id = proj["id"]
        status_field = project_status_field()

    for t in tasks:
        title = f"[Task #{t['n']}] {t['title']}"
        assignees = [GITHUB_HANDLES.get(o, "") for o in t["owners"]]
        assignees = [a for a in assignees if a]

        if t["n"] in have:
            url = have[t["n"]]["html_url"]
            print(f"exists   {title} -> {url}")
            if not dry:
                num = have[t["n"]]["number"]
                if milestone:
                    gh("api", f"repos/{REPO}/issues/{num}", "-X", "PATCH", "-F", f"milestone={milestone}")
                gh("issue", "edit", str(num), "--repo", REPO, "--add-label", ",".join(t["labels"]))
        elif dry:
            print(f"[dry-run] create {title}  status={t['status']}  labels={','.join(t['labels'])}")
            continue
        else:
            cmd = ["issue", "create", "--repo", REPO, "--title", title,
                   "--body", issue_body(t), "--label", ",".join(t["labels"]),
                   "--milestone", MILESTONE_TITLE]
            for a in assignees:
                cmd += ["--assignee", a]
            url = gh(*cmd).strip()
            print(f"created  {title} -> {url}")

        if dry or args.skip_project:
            continue

        item = json.loads(gh("project", "item-add", str(PROJECT_NUMBER), "--owner", ORG,
                             "--url", url, "--format", "json"))
        if status_field:
            field_id, options = status_field
            option = options.get(t["status"].lower())
            if option:
                gh("project", "item-edit", "--id", item["id"], "--project-id", project_id,
                   "--field-id", field_id, "--single-select-option-id", option)
            else:
                print(f"   (no '{t['status']}' option on the board; set status by hand)")

        if t["status"] == "Done":
            gh("issue", "close", url, "--reason", "completed", check=False)

    print("\nDone. Check the board:"
          f" https://github.com/orgs/{ORG}/projects/{PROJECT_NUMBER}/views/1")


if __name__ == "__main__":
    main()
