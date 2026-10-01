"""
Dev-only helper: convert a percent-format (`# %%` / `# %% [markdown]`) .py
script into a matching .ipynb notebook via nbformat, mirroring how the
existing Gini_Monitoring notebooks/executables pair up. Not part of the
calibration monitoring pipeline itself -- run manually / by the build
tooling when producing the notebook twin for a given executable.

Usage: python _convert_to_ipynb.py <script.py> <output.ipynb>
"""
import sys
import re
import nbformat


def convert(py_path: str, ipynb_path: str) -> None:
    with open(py_path, "r", encoding="utf-8") as f:
        text = f.read()

    # split on cell markers, keeping the marker type
    parts = re.split(r"^# %%(.*)$", text, flags=re.MULTILINE)
    # parts[0] is preamble before first marker (should be empty/whitespace)
    nb = nbformat.v4.new_notebook()
    cells = []

    preamble = parts[0].strip()
    if preamble:
        cells.append(nbformat.v4.new_code_cell(preamble))

    i = 1
    while i < len(parts):
        marker = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        body = body.strip("\n")
        if marker.startswith("[markdown]"):
            md_text = "\n".join(
                line[2:] if line.startswith("# ") else line.lstrip("#").lstrip()
                for line in body.splitlines()
            ).strip()
            cells.append(nbformat.v4.new_markdown_cell(md_text))
        else:
            if body.strip():
                cells.append(nbformat.v4.new_code_cell(body))
        i += 2

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }

    with open(ipynb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"wrote {ipynb_path} ({len(cells)} cells)")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
