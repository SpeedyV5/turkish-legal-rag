from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class CorpusDocument:
    doc_id: str
    source_family: str          # mevzuat | tbmm | yargitay
    source_name: str            # specific source/system name
    doc_type: str               # anayasa | kanun | yonetmelik | teblig | karar | teklif | rapor
    title: str
    official_no: Optional[str]
    official_date: Optional[str]
    url: Optional[str]
    language: str
    version_status: str         # current | historical | unknown
    jurisdiction: str           # tr
    text: str
    summary: Optional[str]
    article_refs: Optional[str] # comma-separated or serialized later
    court_chamber: Optional[str]
    tags: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)