"""Reads policies.md and manifest.json and cuts the policies into citeable chunks.

policies.md holds one document per "# Title". Inside a document, "## §2 Heading"
groups sections and "### §2.1 Heading" is one citeable section. The manifest
(matched by doc_id) says whether a document is current and when it starts.
"""
import json
import re
from dataclasses import dataclass
from datetime import date

DOC_ID_PATTERN = re.compile(r"\*\*Document ID:\*\*\s*([A-Za-z0-9_\-]+)")
SECTION_HEADING = re.compile(r"^(#{2,3})\s+(§[\d.]+)\s+(.*)$")
EMAIL_PATTERN = re.compile(r"[\w.\-]+@[\w\-]+(?:\.[\w\-]+)+")


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    doc_type: str  # policy | faq
    effective_date: date
    in_force: bool  # False = current document that starts after APP_TODAY
    section: str  # like §2.1
    heading: str
    parent_heading: str
    text: str

    def embed_text(self):
        """What gets embedded: the titles give the paragraph its context."""
        parts = [self.doc_title, self.parent_heading, self.heading, self.text]
        return " — ".join(part for part in parts if part)


def split_documents(markdown):
    """Return [(doc_id, body_lines)] for every '# Title' block that has a Document ID."""
    blocks = []
    current = None
    for line in markdown.splitlines():
        if line.startswith("# "):
            current = []
            blocks.append(current)
        if current is not None:
            current.append(line)
    documents = []
    for lines in blocks:
        match = DOC_ID_PATTERN.search("\n".join(lines))
        if match:
            documents.append((match.group(1), lines))
    return documents


def split_sections(lines):
    """Yield (section_id, heading, parent_heading, text) for each citeable section."""
    parent = ""
    section = None  # [section_id, heading, parent, text_lines]
    for line in lines:
        match = SECTION_HEADING.match(line)
        if match:
            if section:
                yield section
            level, section_id, title = match.groups()
            heading = f"{section_id} {title}".strip()
            if level == "##":
                parent = heading
                section = [section_id, heading, "", []]
            else:
                section = [section_id, heading, parent, []]
        elif line.startswith("#"):
            if section:
                yield section
            section = None  # the document title line
        elif section is not None:
            section[3].append(line)
    if section:
        yield section


def load_chunks(policies_path, manifest_path, today):
    """Chunks of every usable document. Superseded and unknown documents are dropped.

    Usable = status is current. If it starts after `today` it is kept but marked
    in_force=False, so it can only be mentioned as a future change.
    """
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = {entry["doc_id"]: entry for entry in json.load(handle)}
    with open(policies_path, encoding="utf-8") as handle:
        markdown = handle.read()

    chunks = []
    for doc_id, lines in split_documents(markdown):
        entry = manifest.get(doc_id)
        if entry is None or entry.get("status") != "current":
            continue
        starts = date.fromisoformat(entry["effective_date"])
        for section_id, heading, parent, text_lines in split_sections(lines):
            text = "\n".join(text_lines).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    doc_title=entry["title"],
                    doc_type=entry["doc_type"],
                    effective_date=starts,
                    in_force=starts <= today,
                    section=section_id,
                    heading=heading,
                    parent_heading=parent,
                    text=text,
                )
            )
    return chunks


def split_long_chunks(chunks, count_tokens, max_tokens, overlap_ratio=0.12):
    """Optional: cut sections longer than max_tokens into overlapping windows.

    Only used for the chunk-size experiment. Windows never cross a section
    boundary. Returns new Chunk objects; short sections are returned unchanged.
    """
    result = []
    for chunk in chunks:
        words = chunk.text.split()
        if count_tokens(chunk.embed_text()) <= max_tokens:
            result.append(chunk)
            continue
        start = 0
        while start < len(words):
            end = start + 1
            while end < len(words) and count_tokens(" ".join(words[start : end + 1])) <= max_tokens:
                end += 1
            window = Chunk(**{**chunk.__dict__, "text": " ".join(words[start:end])})
            result.append(window)
            if end >= len(words):
                break
            overlap = max(1, int((end - start) * overlap_ratio))
            start = end - overlap
    return result


def find_support_email(chunks):
    """The support address is read from the documents, not typed into the code."""
    for chunk in chunks:
        match = EMAIL_PATTERN.search(chunk.text)
        if match:
            return match.group(0)
    return None
