"""Sample naming, discovery and the stage directory layout.

The file *stem* is the join key across every stage. ``generate`` writes
``data/events/<stem>.parquet``, ``analyze`` writes
``data/observables/<stem>.parquet``, and so on: parallel directories, same
stem. One grammar, one parser, one builder.

Stems::

    qcd_seed1                 inclusive QCD dijet
    qcdbb_seed1               b-enriched (hard b bbar)
    signal_ctau10mm_seed1     Z' -> qD qDbar at a dark-pion lifetime
    minbias_seed1             minimum bias, for the pileup overlay
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# sample -> (card stem, file tag)
SAMPLES = {
    "signal": ("signal_zprime_hv", "signal"),
    "qcd": ("qcd_dijet", "qcd"),
    "qcd_bb": ("qcd_bbbar", "qcdbb"),
    "minbias": ("minbias", "minbias"),
}
TAG_TO_SAMPLE = {tag: name for name, (_, tag) in SAMPLES.items()}

CTAU_GRID_MM = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

_STEM_RE = re.compile(
    r"^(?P<tag>signal_ctau(?P<ctau>[\d.]+)mm|qcdbb|qcd|minbias)_seed(?P<seed>\d+)$")


def sample_tag(sample: str, ctau_mm: float | None = None) -> str:
    """Canonical stem prefix (without the seed) for a sample point."""
    if sample not in SAMPLES:
        raise ValueError(f"unknown sample {sample!r}")
    if sample == "signal":
        if ctau_mm is None:
            raise ValueError("signal requires ctau_mm")
        return f"signal_ctau{ctau_mm:g}mm"
    return SAMPLES[sample][1]


def stem(sample: str, seed: int, ctau_mm: float | None = None) -> str:
    return f"{sample_tag(sample, ctau_mm)}_seed{seed}"


def parse_stem(name: str) -> dict:
    """Parse a stem (or a filename) back into its sample point.

    ``ctau`` is -1.0 for backgrounds rather than None, so filtering is uniform.
    """
    m = _STEM_RE.match(Path(name).stem)
    if not m:
        raise ValueError(f"cannot parse sample stem: {name!r}")
    tag = m["tag"]
    sample = "signal" if tag.startswith("signal") else TAG_TO_SAMPLE[tag]
    return {"sample": sample, "seed": int(m["seed"]),
            "ctau": float(m["ctau"]) if m["ctau"] else -1.0}


@dataclass(frozen=True)
class SampleFile:
    """One file of one sample point, with paths to its siblings downstream."""
    path: Path
    sample: str
    seed: int
    ctau: float

    @classmethod
    def from_path(cls, path) -> "SampleFile":
        path = Path(path)
        return cls(path=path, **parse_stem(path.name))

    @property
    def stem(self) -> str:
        return self.path.stem

    def sibling(self, stage_dir, suffix: str = ".parquet") -> Path:
        """The same sample point in another stage directory."""
        return Path(stage_dir) / f"{self.stem}{suffix}"


def find(stage_dir, *, samples: list[str] | None = None,
         ctaus: list[float] | None = None, suffix: str = ".parquet") -> list[SampleFile]:
    """Every parseable file in a stage directory, in canonical order.

    Canonical order is (sample rank, ctau, name). Sorting signal on ctau alone
    would leave ties broken by filesystem glob order, which makes runs
    irreproducible; the name breaks them deterministically.
    """
    rank = {"qcd": 0, "qcd_bb": 1, "signal": 2, "minbias": 3}
    out = []
    for p in sorted(Path(stage_dir).glob(f"*{suffix}")):
        try:
            f = SampleFile.from_path(p)
        except ValueError:
            continue
        if samples and f.sample not in samples:
            continue
        if ctaus is not None and f.sample == "signal" and f.ctau not in ctaus:
            continue
        out.append(f)
    return sorted(out, key=lambda f: (rank[f.sample], f.ctau, f.path.name))
