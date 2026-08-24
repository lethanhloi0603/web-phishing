from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import dom_fingerprint, html_sha256


def row(html, label, source, generator, family, template, index, url=""):
    html_hash = html_sha256(html)
    return {
        "sample_id": html_hash[:16],
        "html": html,
        "label": label,
        "source": source,
        "generator": generator,
        "generator_family": family,
        "brand": ["university", "cloud", "commerce", "mail"][index % 4],
        "template_id": template,
        "prompt_id": f"prompt-{template}",
        "url": url,
        "collection_date": f"2026-0{1 + index % 6}-{1 + index % 27:02d}",
        "html_hash": html_hash,
        "dom_hash": dom_fingerprint(html),
        "quality_level": "demo",
        "manual_validated": 1,
    }


def conventional_rows(prefix, per_class=40):
    rows = []
    for index in range(per_class):
        template = f"{prefix}-benign-{index % 10}"
        html = (
            f"<html data-split='{prefix}'><head><title>{prefix} community page {index}</title></head><body>"
            f"<main><h1>{prefix} University Library {index}</h1><p>Search books, events and journals.</p>"
            f"<a href='https://library.example/articles/{index}'>Read article</a></main></body></html>"
        )
        rows.append(row(html, 0, "conventional", "human_kit", "human", template, index,
                        f"https://library.example/{prefix}/pages/{index}"))

    for index in range(per_class):
        template = f"{prefix}-phish-{index % 10}"
        html = (
            f"<html data-split='{prefix}'><head><title>{prefix} account verification {index}</title></head><body>"
            f"<form action='https://collect.invalid/session/{index}'><h2>Verify your account</h2>"
            f"<input name='email-{index}'><input type='password' name='password-{index}'>"
            f"<button>Sign in immediately</button></form></body></html>"
        )
        rows.append(row(html, 1, "conventional", "human_kit", "human", template, index,
                        f"https://secure-check-{prefix}-{index}.invalid/login"))
    return rows


def llm_rows():
    rows = []
    generators = [
        ("gpt-4o-agent", "gpt"),
        ("claude-3-7-agent", "claude"),
        ("gemini-2-agent", "gemini"),
    ]
    for generator_index, (generator, family) in enumerate(generators):
        for index in range(30):
            template = f"llm-{family}-{index % 10}"
            html = (
                f"<!-- generated demo {generator} sample {index} -->"
                f"<html><head><title>Security center {index}</title></head><body>"
                f"<main class='{family}-layout-{index}'><section><h1>Session expired</h1>"
                f"<p>Confirm your identity to restore access.</p>"
                f"<form action='https://receiver.invalid/{family}/{index}'>"
                f"<label>Email</label><input autocomplete='username' name='u-{index}'>"
                f"<label>Password</label><input type='password' name='p-{index}'>"
                f"<button onclick='submitForm()'>Continue securely</button>"
                f"</form></section></main></body></html>"
            )
            rows.append(row(
                html, 1, "llm", generator, family, template,
                index + generator_index * 30,
                f"https://{family}-{index}.invalid/verify",
            ))
    return rows


def main():
    output = Path("data/demo")
    output.mkdir(parents=True, exist_ok=True)
    phresh_train = pd.DataFrame(conventional_rows("train", per_class=40))
    phresh_test = pd.DataFrame(conventional_rows("test", per_class=20))
    llm = pd.DataFrame(llm_rows())
    phresh_train.to_csv(output / "phresh_train.csv", index=False)
    phresh_test.to_csv(output / "phresh_test.csv", index=False)
    llm.to_csv(output / "llm_phish.csv", index=False)
    print(f"Demo conventional train: {len(phresh_train)}")
    print(f"Demo conventional test:  {len(phresh_test)}")
    print(f"Demo LLM phishing:       {len(llm)}")
    print(f"Wrote files to {output}")


if __name__ == "__main__":
    main()
