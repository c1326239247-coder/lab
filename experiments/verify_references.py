# -*- coding: utf-8 -*-
"""Check every bibliography entry against arXiv, Crossref and OpenAlex.

For each entry in rewrite/refs.py this reports whether the cited title,
first author and year agree with what the registries hold.  Venue claims are
recorded rather than asserted: a preprint year earlier than the cited venue
year is normal, and neither Crossref nor OpenAlex indexes ICLR or NeurIPS
reliably, so those need the proceedings page itself.

Run with a reachable network (or HTTPS_PROXY set):
    python experiments/verify_references.py
Writes results/reference_audit.json and prints only the problems.
"""

import io
import json
import os
import re
import sys
import time
from typing import NamedTuple

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common

REFS_PATH = os.path.join(common.REPO_ROOT, "rewrite", "refs.py")
MAILTO = "318521259+c1326239247-coder@users.noreply.github.com"
# OpenAlex rate-limits anonymous bursts hard; it only adds venue metadata on
# top of arXiv and Crossref, so set REFERENCE_AUDIT_OPENALEX=1 to enable it.
USE_OPENALEX = os.environ.get("REFERENCE_AUDIT_OPENALEX") == "1"
VENUES = ["ICML", "ICLR", "NeurIPS", "CVPR", "ICCV", "ECCV", "ACL", "EMNLP",
          "NAACL", "IJCV", "Neurocomputing", "SPIE", "EuroSys", "Findings",
          "TPAMI", "COLM"]
TITLE_ENDS = [" In: ", ". In:", ". arXiv", " arXiv:", ". doi:",
              ". International Journal", ". Neurocomputing", ". Technical report",
              ". New York", ". Advances", ". Proceedings", ". Findings",
              ". Lecture Notes", ". National Science Review"]


class Entry(NamedTuple):
    key: str
    text: str
    arxiv: str
    doi: str
    years: list[int]
    venue_claims: list[str]


def normalize(text):
    """Lowercase, drop punctuation, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def parse_entries(path):
    """Read rewrite/refs.py into typed entries without importing it."""
    source = io.open(path, encoding="utf-8").read()
    blocks = re.findall(r'\n    \("([a-z0-9]+)",\n(.*?)(?=\n    \("[a-z0-9]+",\n|\n\])',
                        source, re.S)
    entries = []
    for key, body in blocks:
        text = " ".join(re.findall(r'"([^"]*)"', body))
        arxiv = re.search(r"arXiv:(\d{4}\.\d{4,5})", text)
        doi = re.search(r"doi:(10\.\d{4,9}/\S+?)\.?$", text, re.M)
        if doi is None:
            doi = re.search(r"(10\.1145/\d+\.\d+|10\.18653/v1/[\w.\-]+)", text)
        entries.append(Entry(key, text,
                             arxiv.group(1) if arxiv else "",
                             doi.group(1) if doi else "",
                             [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", text)],
                             [v for v in VENUES if re.search(r"\b%s\b" % v, text)]))
    return entries


def fetch_arxiv(ids):
    """One batched arXiv API call for the given ids."""
    import xml.etree.ElementTree as ET

    namespace = {"a": "http://www.w3.org/2005/Atom",
                 "ar": "http://arxiv.org/schemas/atom"}
    response = requests.get("https://export.arxiv.org/api/query",
                            params={"id_list": ",".join(ids),
                                    "max_results": len(ids)},
                            timeout=60)
    response.raise_for_status()
    found = {}
    for element in ET.fromstring(response.content).findall("a:entry", namespace):
        raw = element.find("a:id", namespace).text.split("/abs/")[-1]
        comment = element.find("ar:comment", namespace)
        found[raw.split("v")[0]] = {
            "title": " ".join(element.find("a:title", namespace).text.split()),
            "authors": [a.find("a:name", namespace).text
                        for a in element.findall("a:author", namespace)],
            "published": element.find("a:published", namespace).text[:10],
            "updated": element.find("a:updated", namespace).text[:10],
            "comment": " ".join(comment.text.split()) if comment is not None else "",
        }
    return found


def fetch_crossref(doi):
    """Crossref record for a DOI, or None when it does not resolve."""
    response = requests.get("https://api.crossref.org/works/" + doi,
                            timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    message = response.json()["message"]
    year = (message.get("published", {}).get("date-parts", [[None]])[0][0]
            or message.get("issued", {}).get("date-parts", [[None]])[0][0])
    return {"title": (message.get("title") or [""])[0],
            "container": (message.get("container-title") or [""])[0],
            "year": year,
            "authors": [author.get("family", "")
                        for author in message.get("author", [])]}


def fetch_openalex(title):
    """Best OpenAlex match for a title, used only for venue and year."""
    response = requests.get("https://api.openalex.org/works",
                            params={"filter": "title.search:" + title,
                                    "per-page": 3, "mailto": MAILTO},
                            timeout=60)
    if response.status_code == 429:
        time.sleep(3.0)
        response = requests.get("https://api.openalex.org/works",
                                params={"filter": "title.search:" + title,
                                        "per-page": 3, "mailto": MAILTO},
                                timeout=60)
    response.raise_for_status()
    hits = response.json().get("results", [])
    if not hits:
        return None
    top = hits[0]
    source = ((top.get("primary_location") or {}).get("source") or {})
    return {"display_name": top.get("display_name", ""),
            "venue": source.get("display_name") or "",
            "year": top.get("publication_year"),
            "doi": (top.get("doi") or "").replace("https://doi.org/", ""),
            "authorships": [a.get("author", {}).get("display_name", "")
                            for a in top.get("authorships", [])]}


def quoted_title(text):
    """Title as written in the entry: after the author list to the venue mark."""
    body = re.sub(r'^[A-Z][^.]{0,120}?\.\s+', "", text, count=1)
    cuts = [body.find(marker) for marker in TITLE_ENDS]
    cuts = [cut for cut in cuts if cut > 0]
    if cuts:
        body = body[:min(cuts)]
    return body.strip(" .,")


def check_url(url):
    """HTTP status of a cited web address, or None when it never answered."""
    try:
        return requests.get(url, timeout=45, allow_redirects=True).status_code
    except requests.RequestException:
        return None


def audit():
    """Run every registry check and return the per-entry findings."""
    entries = parse_entries(REFS_PATH)
    arxiv_ids = [e.arxiv for e in entries if e.arxiv]
    arxiv = fetch_arxiv(arxiv_ids)
    report = {}
    for entry in entries:
        problems = []
        info = {"key": entry.key, "cited_years": entry.years,
                "venue_claims": entry.venue_claims, "arxiv": entry.arxiv,
                "doi": entry.doi}
        title = quoted_title(entry.text)
        info["parsed_title"] = title
        urls = [u.rstrip(".,;") for u in re.findall(r"https?://\S+", entry.text)]
        info["urls"] = urls
        url_status = {}
        for url in urls:
            url_status[url] = check_url(url)
            if url_status[url] != 200:
                problems.append("cited link %s returned %s"
                                % (url, url_status[url]))
        info["url_status"] = url_status
        if not entry.arxiv and not entry.doi and not urls:
            problems.append("no arXiv id, DOI or link to machine-check")
        if entry.arxiv:
            record = arxiv.get(entry.arxiv)
            if record is None:
                problems.append("arXiv id %s returned no record" % entry.arxiv)
            else:
                info["arxiv_title"] = record["title"]
                info["arxiv_published"] = record["published"]
                info["arxiv_comment"] = record["comment"]
                if normalize(record["title"]) != normalize(title):
                    problems.append("title differs from arXiv (%s)" % record["title"])
                if record["authors"]:
                    surname = record["authors"][0].split()[-1].lower()
                    if surname not in normalize(entry.text):
                        problems.append("arXiv first-author surname %s not in entry"
                                        % surname)
                arxiv_year = int(record["published"][:4])
                # A conference paper usually appears on arXiv a year earlier.
                if entry.years and not (arxiv_year <= max(entry.years)
                                        <= arxiv_year + 2):
                    problems.append("arXiv v1 year %d is not within two years "
                                    "of the cited years %s"
                                    % (arxiv_year, entry.years))
        if entry.doi:
            record = fetch_crossref(entry.doi)
            if record is None:
                problems.append("DOI %s does not resolve at Crossref" % entry.doi)
            else:
                info["crossref_title"] = record["title"]
                info["crossref_container"] = record["container"]
                info["crossref_year"] = record["year"]
                if normalize(record["title"]) != normalize(title):
                    problems.append("title differs from Crossref (%s)" % record["title"])
                if record["year"] and entry.years and record["year"] not in entry.years:
                    problems.append("Crossref year %s not among cited years %s"
                                    % (record["year"], entry.years))
        hit = None
        if USE_OPENALEX:
            try:
                hit = fetch_openalex(title)
            except requests.RequestException as error:
                problems.append("OpenAlex lookup failed: %s" % error)
        if hit:
            info["openalex_title"] = hit["display_name"]
            info["openalex_venue"] = hit["venue"]
            info["openalex_year"] = hit["year"]
            info["openalex_doi"] = hit["doi"]
            if normalize(hit["display_name"]) != normalize(title):
                problems.append("no OpenAlex record with this exact title")
        time.sleep(0.6)
        info["problems"] = problems
        report[entry.key] = info
    return report


def main():
    report = audit()
    path = common.save_json("reference_audit.json", report)
    flagged = {k: v["problems"] for k, v in report.items() if v["problems"]}
    print("checked %d entries; %d flagged" % (len(report), len(flagged)))
    for key, problems in flagged.items():
        print("\n%s" % key)
        for problem in problems:
            print("  - %s" % problem)
    print("\nsaved ->", path)


if __name__ == "__main__":
    main()
