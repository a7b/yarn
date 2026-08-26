"""The documentation must describe the package that actually ships.

Two failure modes this guards:

1. Snippets that stop being valid Python (renamed symbol, changed signature).
   Every fenced ``python`` block in README.md and docs/kernels.md is compiled;
   blocks that are self-contained are executed against the bundled artifacts.
2. Numbers copied into prose going stale. The HTML deep-dive's S1 knob table
   and the README's default-knob claims are checked against ``config.py``.
"""
import html
import json
import re
import tomllib
from pathlib import Path

import pytest

from telescoping_decoder.config import (DEFAULT_S1_SYSTEM, S1_PRESETS,
                                        S1Config)

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
KERNELS_MD = ROOT / "docs" / "kernels.md"
DECODER_HTML = ROOT / "docs" / "s1_s2_decoders.html"
DECODER_HTMLS = (
    DECODER_HTML,
)
DECODER_KERNELS = (
    ROOT / "telescoping_decoder/kernels/s1_layered_bp.py",
    ROOT / "telescoping_decoder/kernels/s2_relay_bp.py",
    ROOT / "telescoping_decoder/kernels/s2_relay_bp_gari.py",
)

# Snippets that cannot run as-is: they stand in for user-supplied objects
# (a stim circuit, a detector mask, an H matrix) or would need a GPU.
_UNRUNNABLE = ("circuit", "model.dem", "stem_gari_matrices.npz",
               "model_matrices.npz",
               "hardware_syndromes", "gari_transform(", "S1LayeredBP(",
               "/path/to/")


def _python_blocks(path: Path):
    text = path.read_text()
    return [(path.name, i, block) for i, block in
            enumerate(re.findall(r"```python\n(.*?)```", text, re.DOTALL))]


ALL_BLOCKS = _python_blocks(README) + _python_blocks(KERNELS_MD)


def test_docs_have_python_blocks():
    assert len(ALL_BLOCKS) >= 3, "expected at least three documentation snippets"


@pytest.mark.parametrize(
    "name,index,source", ALL_BLOCKS,
    ids=[f"{n}#{i}" for n, i, _ in ALL_BLOCKS])
def test_doc_snippet_compiles(name, index, source):
    compile(source, f"{name}#{index}", "exec")


@pytest.mark.parametrize(
    "name,index,source",
    [b for b in ALL_BLOCKS if not any(t in b[2] for t in _UNRUNNABLE)],
    ids=[f"{n}#{i}" for n, i, s in ALL_BLOCKS
         if not any(t in s for t in _UNRUNNABLE)])
def test_runnable_doc_snippet_executes(name, index, source):
    """Runs the snippets that only touch the package's own API."""
    exec(compile(source, f"{name}#{index}", "exec"), {"__name__": "__doc__"})


def test_readme_documents_the_shipped_s1_knobs():
    text = README.read_text()
    row = S1_PRESETS[DEFAULT_S1_SYSTEM]
    for field in ("k", "n_iters", "hybrid_sp_iters"):
        assert f"`{field}={row[field]}`" in text, (
            f"README does not state the shipped S1 {field}={row[field]}")


def test_readme_sizing_table_covers_the_default_batches():
    """The sizing advice must mention the knobs it is advising about."""
    text = README.read_text()
    for knob in ("s1.shots_per_batch", "s2.batch_size"):
        assert knob in text


def test_readme_documents_the_npz_schema():
    """Every key consumed by the original and GARI loaders is documented."""
    text = README.read_text()
    required = (
        "h_data", "h_indices", "h_indptr", "h_shape",
        "l_data", "l_indices", "l_indptr", "l_shape",
        "probs", "dc_pad", "dv_pad",
        "gari_n_detectors", "gari_is_x_detector",
        "gari_col_block_bounds", "gari_row_block_bounds",
        "gari_relevant_rows", "gari_relevant_priors", "gari_layers",
        "gari_u_map", "gari_v_map", "gari_init_basis",
        "gari_answer_block",
    )
    for key in required:
        assert f"`{key}`" in text, f"README does not document NPZ key {key}"


def test_readme_documents_the_own_circuit_contract():
    text = README.read_text()
    required = (
        "## Using your own Stim circuit",
        "examples/toy_xz_surface_code_memory.stim",
        "OBSERVABLE_INCLUDE",
        "init_basis=\"X\"",
        "init_basis=\"Z\"",
        "Mixed-basis detectors are not supported",
        "every mixed `eY` error column",
        "TelescopeConfig(use_gari=False)",
    )
    for phrase in required:
        assert phrase in text, f"README omits circuit requirement {phrase!r}"


def test_s4_is_a_core_dependency():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = metadata["project"]["dependencies"]
    extras = metadata["project"].get("optional-dependencies", {})
    assert any(dep.startswith("gurobipy") for dep in dependencies)
    assert "ip" not in extras
    assert "[ip]" not in README.read_text()


def test_runnable_circuit_example_files_exist_and_compile():
    circuit = ROOT / "examples" / "toy_xz_surface_code_memory.stim"
    script = ROOT / "examples" / "from_stim_circuit.py"
    assert circuit.is_file()
    assert script.is_file()
    compile(script.read_text(), str(script), "exec")


def test_readme_kernel_directory_links_resolve():
    text = README.read_text()
    for target in ("telescoping_decoder/kernels/", "telescoping_decoder/_c/"):
        assert f"]({target})" in text
        assert (ROOT / target).is_dir()


@pytest.mark.parametrize("path", [README, KERNELS_MD], ids=["README", "kernels"])
def test_internal_links_resolve(path):
    """All local Markdown anchors must resolve."""
    text = path.read_text()
    slugs = {re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
             for h in re.findall(r"^#{1,6} (.+)$", text, re.MULTILINE)}
    targets = re.findall(r"\]\(#([a-z0-9-]+)\)", text)
    missing = sorted(t for t in targets if t not in slugs)
    assert not missing, f"{path.name} links to missing sections: {missing}"


def test_html_s1_table_matches_config_defaults():
    """The deep-dive's §5f table must be the literal S1Config defaults."""
    html = DECODER_HTML.read_text()
    section = html[html.index('id="s1-cfg"'):]
    section = section[:section.index("</table>")]
    documented = dict(re.findall(
        r'<tr data-default-field="([^"]+)" data-default-value="([^"]*)">',
        section))
    expected = {name: str(value) for name, value in vars(S1Config()).items()}
    assert documented == expected


def test_html_does_not_cite_code_outside_this_repo():
    """The guide must not refer to source outside this repository."""
    html = DECODER_HTML.read_text().lower()
    for token in ("c10s3", "ldpc-decoding", "probe script"):
        assert token not in html, f"{token!r} refers to code not in this repo"


def test_html_explainer_omits_benchmark_results():
    """The explainer documents code, not one machine or experiment."""
    html = DECODER_HTML.read_text()
    prose = html[:html.index("<!-- DECODER_SOURCE_DATA_START -->")].lower()
    for token in ("benchmark", "h100", "rtx 500 ada", "shots/s",
                  "historical profile"):
        assert token not in prose


def test_html_embedded_source_regions_are_current():
    """The scroll-linked guide must point into the source that actually ships."""
    html = DECODER_HTML.read_text()
    match = re.search(
        r'<script id="decoder-source-data" type="application/json">'
        r'(.*?)</script>', html, re.DOTALL)
    assert match, "the standalone explainer must contain its source data"
    payload = json.loads(match.group(1))
    used_regions = set(re.findall(r'data-code-region="([^"]+)"', html))

    assert used_regions == set(payload["regions"]), (
        "HTML source mappings and embedded region definitions differ")
    expected_paths = {
        "s1": "telescoping_decoder/kernels/s1_layered_bp.py",
        "s2": "telescoping_decoder/kernels/s2_relay_bp.py",
        "s2-gari": "telescoping_decoder/kernels/s2_relay_bp_gari.py",
    }
    for source_id, path in expected_paths.items():
        source = payload["sources"][source_id]
        assert source["path"] == path
        assert source["text"] == (ROOT / path).read_text(), (
            f"embedded {source_id} source is stale")

    for region_id in used_regions:
        region = payload["regions"][region_id]
        source = payload["sources"][region["source"]]
        assert 1 <= region["start"] <= region["end"] <= len(
            source["text"].splitlines())


@pytest.mark.parametrize("path", DECODER_HTMLS, ids=lambda path: path.stem)
def test_html_linked_snippets_are_exact_and_statically_highlighted(path):
    """Source excerpts must work with JavaScript disabled and never drift."""
    document = path.read_text()
    payload_match = re.search(
        r'<script id="decoder-source-data" type="application/json">'
        r'(.*?)</script>', document, re.DOTALL)
    assert payload_match
    payload = json.loads(payload_match.group(1))

    excerpts = re.findall(
        r'<pre([^>]*)><code[^>]*>(.*?)</code></pre>',
        document,
        re.DOTALL,
    )
    excerpts = [
        (attrs, markup) for attrs, markup in excerpts
        if "data-code-region=" in attrs
        and "data-code-start=" in attrs
        and "data-code-end=" in attrs
    ]
    assert excerpts
    for attrs, markup in excerpts:
        region_id = re.search(r'data-code-region="([^"]+)"', attrs).group(1)
        start_text = re.search(r'data-code-start="(\d+)"', attrs).group(1)
        end_text = re.search(r'data-code-end="(\d+)"', attrs).group(1)
        assert '<span class="syn-' in markup
        region = payload["regions"][region_id]
        source = payload["sources"][region["source"]]["text"]
        expected = "\n".join(
            source.splitlines()[int(start_text) - 1:int(end_text)]
        )
        actual = html.unescape(re.sub(r"<[^>]+>", "", markup))
        annotated = 'data-code-annotated="true"' in attrs
        if annotated:
            actual_lines = actual.splitlines()
            expected_lines = expected.splitlines()
            prefix_count = len(actual_lines) - len(expected_lines)
            prefixes_are_comments = (
                prefix_count >= 0
                and all(line.strip().startswith("//")
                        for line in actual_lines[:prefix_count])
            )
            actual_lines = actual_lines[prefix_count:]
            matches = len(actual_lines) == len(expected_lines) and all(
                a == e or a.startswith(e + "  // ")
                for a, e in zip(actual_lines, expected_lines)
            ) and prefixes_are_comments and (
                prefix_count > 0
                or any(a != e for a, e in zip(actual_lines, expected_lines))
            )
            assert matches, f"stale annotated {region_id} excerpt in {path.name}"
        else:
            assert actual == expected, f"stale {region_id} excerpt in {path.name}"


@pytest.mark.parametrize("path", DECODER_KERNELS, ids=lambda p: p.stem)
def test_explainer_cuda_functions_are_commented(path):
    """Every CUDA entry point shown in the source panel needs a clear purpose."""
    text = path.read_text()
    lines = text.splitlines()
    missing_cuda = []
    for index, line in enumerate(lines):
        match = re.search(
            r'extern\s+"C"\s+__global__.*?void\s+([A-Za-z_]\w*)\s*\(',
            line)
        if not match and 'extern "C" __global__ __launch_bounds__' in line:
            for following in lines[index + 1:index + 4]:
                match = re.search(r'void\s+([A-Za-z_]\w*)\s*\(', following)
                if match:
                    break
        if match:
            prefix = "\n".join(lines[:index])
            blocks = re.findall(r"/\*\*.*?\*/", prefix, re.DOTALL)
            doc = blocks[-1] if blocks else ""
            trailing = prefix[prefix.rfind(doc) + len(doc):] if doc else prefix
            complete = (
                not trailing.strip()
                and "@brief" in doc
                and "@param[in]" in doc
                and ("@param[out]" in doc or "@param[in,out]" in doc)
            )
            if not complete:
                missing_cuda.append(match.group(1))
    assert not missing_cuda, (
        "CUDA functions need @brief, input, and output documentation: "
        f"{missing_cuda}")
