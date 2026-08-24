import hashlib
import html as html_lib
import math
import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np
from bs4 import BeautifulSoup, Comment


DOM_SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2}
COMMON_TAGS = [
    "html", "head", "body", "title", "meta", "link", "style", "script", "main",
    "header", "footer", "nav", "section", "article", "aside", "div", "span", "p",
    "a", "img", "form", "label", "input", "button", "select", "option", "textarea",
    "table", "thead", "tbody", "tr", "td", "th", "ul", "ol", "li", "h1", "h2",
    "h3", "h4", "h5", "h6", "br", "hr", "iframe", "svg", "path", "canvas",
    "template", "video", "audio", "object", "embed",
]
DOM_VOCAB = dict(DOM_SPECIAL)
for i, tag in enumerate(COMMON_TAGS, start=len(DOM_VOCAB)):
    DOM_VOCAB[tag] = i


STATIC_FEATURE_NAMES = [
    "html_length_log", "visible_text_length_log", "dom_nodes_log", "max_dom_depth",
    "form_count", "input_count", "password_count", "button_count", "link_count",
    "image_count", "iframe_count", "script_count", "external_script_count",
    "external_link_count", "empty_link_count", "event_handler_count", "hidden_count",
    "form_external_action_count", "suspicious_keyword_count", "urgent_keyword_count",
    "javascript_url_count", "base64_indicator_count", "eval_indicator_count",
    "url_length_log", "url_subdomain_count", "url_entropy", "digit_ratio",
]

SUSPICIOUS_WORDS = {
    "account", "bank", "confirm", "credential", "login", "password", "payment",
    "security", "signin", "suspended", "update", "verification", "verify", "wallet",
}
URGENT_WORDS = {
    "act now", "immediately", "limited time", "required", "urgent", "within 24",
    "expired", "suspended",
}


def safe_soup(raw_html: str) -> BeautifulSoup:
    return BeautifulSoup(raw_html or "", "html.parser")


def html_to_semantic_text(raw_html: str, max_chars: int = 20000) -> str:
    """Extract non-executable content and interaction semantics from archived HTML."""
    soup = safe_soup(raw_html)
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    parts = []
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if title:
        parts.append("TITLE " + title)

    text = soup.get_text(" ", strip=True)
    if text:
        parts.append(text)

    for el in soup.find_all(["form", "input", "button", "a", "select", "textarea"]):
        attrs = []
        for key in (
            "type", "name", "placeholder", "autocomplete", "aria-label", "role", "action"
        ):
            value = el.attrs.get(key)
            if isinstance(value, str) and value:
                attrs.append(f"{key}={value}")
        visible = el.get_text(" ", strip=True)
        token = " ".join([el.name] + attrs + ([visible] if visible else []))
        if token.strip():
            parts.append(token)

    out = html_lib.unescape(" ".join(parts))
    return re.sub(r"\s+", " ", out).strip()[:max_chars]


def html_to_dom_tags(raw_html: str, max_nodes: int = 512):
    """Return a bounded preorder-like DOM tag sequence without executing the page."""
    tags = []
    for el in safe_soup(raw_html).find_all(True):
        name = (el.name or "").lower()
        if name:
            tags.append(name)
        if len(tags) >= max_nodes:
            break
    return tags


def dom_tags_to_ids(tags, max_len: int = 256):
    ids = [DOM_VOCAB["<CLS>"]]
    ids.extend(DOM_VOCAB.get(tag, DOM_VOCAB["<UNK>"]) for tag in tags[: max_len - 1])
    ids.extend([DOM_VOCAB["<PAD>"]] * (max_len - len(ids)))
    return ids[:max_len]


def _host(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    if value.startswith("//"):
        value = "https:" + value
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
        return ""
    try:
        return (urlparse(value).hostname or "").lower()
    except ValueError:
        return ""


def _is_external(value: str, page_host: str) -> bool:
    target_host = _host(value)
    return bool(target_host and page_host and target_host != page_host)


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def extract_static_features(raw_html: str, url: str = "") -> np.ndarray:
    """Extract inexpensive, non-executing URL/HTML indicators in a stable order."""
    raw_html = raw_html or ""
    url = url or ""
    soup = safe_soup(raw_html)
    tags = soup.find_all(True)
    page_host = _host(url)
    visible = soup.get_text(" ", strip=True).lower()
    raw_lower = raw_html.lower()

    max_depth = 0
    for tag in tags:
        depth = sum(1 for parent in tag.parents if getattr(parent, "name", None))
        max_depth = max(max_depth, depth)

    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    links = soup.find_all("a")
    scripts = soup.find_all("script")
    event_handlers = sum(
        1 for tag in tags for key in tag.attrs if str(key).lower().startswith("on")
    )
    hidden = sum(
        1
        for tag in tags
        if str(tag.attrs.get("type", "")).lower() == "hidden"
        or "display:none" in str(tag.attrs.get("style", "")).replace(" ", "").lower()
    )
    suspicious = sum(len(re.findall(rf"\b{re.escape(word)}\b", visible)) for word in SUSPICIOUS_WORDS)
    urgent = sum(visible.count(word) for word in URGENT_WORDS)
    host_parts = [part for part in page_host.split(".") if part]
    subdomains = max(0, len(host_parts) - 2)
    digit_ratio = sum(ch.isdigit() for ch in url) / max(1, len(url))

    values = [
        math.log1p(len(raw_html)), math.log1p(len(visible)), math.log1p(len(tags)), max_depth,
        len(forms), len(inputs),
        sum(str(tag.attrs.get("type", "")).lower() == "password" for tag in inputs),
        len(soup.find_all("button")), len(links), len(soup.find_all("img")),
        len(soup.find_all("iframe")), len(scripts),
        sum(_is_external(str(tag.attrs.get("src", "")), page_host) for tag in scripts),
        sum(_is_external(str(tag.attrs.get("href", "")), page_host) for tag in links),
        sum(str(tag.attrs.get("href", "")).strip().lower() in {"", "#", "javascript:void(0)"} for tag in links),
        event_handlers, hidden,
        sum(_is_external(str(tag.attrs.get("action", "")), page_host) for tag in forms),
        suspicious, urgent, raw_lower.count("javascript:"), raw_lower.count("base64"),
        len(re.findall(r"\beval\s*\(", raw_lower)), math.log1p(len(url)), subdomains,
        _entropy(url), digit_ratio,
    ]
    return np.asarray(values, dtype=np.float32)


def html_sha256(raw_html: str) -> str:
    normalized = re.sub(r"\s+", " ", raw_html or "").strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def dom_fingerprint(raw_html: str) -> str:
    """Coarse structural fingerprint used to group near-identical templates."""
    soup = safe_soup(raw_html)
    tokens = []
    for tag in soup.find_all(True)[:1000]:
        depth = min(20, sum(1 for parent in tag.parents if getattr(parent, "name", None)))
        semantic_attrs = []
        for key in ("type", "role", "autocomplete"):
            value = tag.attrs.get(key)
            if isinstance(value, str) and value:
                semantic_attrs.append(f"{key}={value.lower()}")
        tokens.append(f"{depth}:{tag.name}:{'|'.join(semantic_attrs)}")
    payload = "\n".join(tokens)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def stable_int_id(value: str, modulo: int = 2**31 - 1) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def normalize_generator(name: str) -> str:
    if name is None:
        return "unknown"
    value = str(name).strip().lower()
    aliases = {
        "gpt": ["gpt", "openai", "chatgpt"], "claude": ["claude", "anthropic"],
        "gemini": ["gemini", "bard", "google"], "grok": ["grok", "xai"],
        "llama": ["llama", "meta"], "mistral": ["mistral", "mixtral"],
        "qwen": ["qwen", "alibaba"], "nova": ["nova", "amazon"],
        "human": ["human", "conventional", "human_kit"],
    }
    for family, needles in aliases.items():
        if any(needle in value for needle in needles):
            return family
    return re.sub(r"[^a-z0-9_-]+", "_", value) or "unknown"


def strip_html_comments(raw_html: str) -> str:
    soup = safe_soup(raw_html)
    for node in soup.find_all(string=lambda value: isinstance(value, Comment)):
        node.extract()
    return str(soup)


def normalize_html_identifiers(raw_html: str) -> str:
    soup = safe_soup(raw_html)
    for tag in soup.find_all(True):
        if "id" in tag.attrs:
            tag.attrs["id"] = "normalized-id"
        if "class" in tag.attrs:
            tag.attrs["class"] = ["normalized-class"]
    return str(soup)


def remove_visible_text(raw_html: str) -> str:
    soup = safe_soup(raw_html)
    for node in list(soup.find_all(string=True)):
        if node.parent and node.parent.name not in {"script", "style"}:
            node.replace_with(" ")
    return str(soup)


ARTIFACT_TRANSFORMS = {
    "comments_removed": strip_html_comments,
    "identifiers_normalized": normalize_html_identifiers,
    "visible_text_removed": remove_visible_text,
}


def apply_artifact_transform(raw_html: str, mode: str) -> str:
    """Apply one of the artifact stress-test transformations without executing HTML."""
    try:
        transform = ARTIFACT_TRANSFORMS[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown artifact transform '{mode}'; choices={sorted(ARTIFACT_TRANSFORMS)}"
        ) from exc
    return transform(raw_html)
