import os

docs = [
    "README.md",
    "WORKFLOWS.md",
    "API_REFERENCE.md",
    "CHANGELOG.md",
    # Add more as needed
]

formats = [
    ("pdf", "pdf"),
    ("html", "html"),
    ("docx", "docx"),
    ("txt", "plain"),
]

os.makedirs("docs", exist_ok=True)

for doc in docs:
    base = os.path.splitext(doc)[0]
    for ext, pandoc_fmt in formats:
        cmd = f"pandoc {doc} -o docs/{base}.{ext} -t {pandoc_fmt}"
        print(f"Exporting: {cmd}")
        os.system(cmd) 