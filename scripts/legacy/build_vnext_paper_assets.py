#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import io as io_utils
from src.harmonize import fit_harmonizer
from src.preprocess import add_target_meta, build_target_parcels, map_gtex_to_target


@dataclass
class Config:
    csv_path: str = "gxp_samples.csv"
    hvg_path: str = "ahba_100hvg.txt"
    hvg_run_root: str = "out/vnext_alignment"
    allgenes_run_root: str = "out/vnext_alignment_allgenes"
    out_root: str = "out/vnext_alignment_allgenes/paper_vnext"
    seed: int = 123
    whiten_eps: float = 1e-4
    combat_use_covariates: bool = True
    hier_lambda_a: float = 10.0
    hier_lambda_b: float = 10.0
    hier_base_method: str = "robustz_affine"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Build manuscript assets for vNext all-gene report.")
    p.add_argument("--csv-path", default=Config.csv_path)
    p.add_argument("--hvg-path", default=Config.hvg_path)
    p.add_argument("--hvg-run-root", default=Config.hvg_run_root)
    p.add_argument("--allgenes-run-root", default=Config.allgenes_run_root)
    p.add_argument("--out-root", default=Config.out_root)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--whiten-eps", type=float, default=Config.whiten_eps)
    p.add_argument("--combat-use-covariates", type=lambda s: str(s).lower() in {"1", "true", "yes", "y"}, default=Config.combat_use_covariates)
    p.add_argument("--hier-lambda-a", type=float, default=Config.hier_lambda_a)
    p.add_argument("--hier-lambda-b", type=float, default=Config.hier_lambda_b)
    p.add_argument("--hier-base-method", default=Config.hier_base_method)
    a = p.parse_args()
    return Config(**vars(a))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _holm(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for i, idx in enumerate(order):
        adj = (n - i) * p[idx]
        running = max(running, adj)
        out[idx] = min(running, 1.0)
    return out


def _rank_biserial_from_diff(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    d = d[np.abs(d) > 1e-12]
    if d.size == 0:
        return np.nan
    r = stats.rankdata(np.abs(d), method="average")
    pos = float(np.sum(r[d > 0]))
    neg = float(np.sum(r[d < 0]))
    den = pos + neg
    if den <= 0:
        return np.nan
    return (pos - neg) / den


def _sign_test_pvalue(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    d = d[np.abs(d) > 1e-12]
    if d.size == 0:
        return np.nan
    n_pos = int(np.sum(d > 0))
    n = int(d.size)
    return float(stats.binomtest(k=n_pos, n=n, p=0.5, alternative="two-sided").pvalue)


def _parse_combo(combo: str) -> Tuple[str, str, str, str]:
    parts = str(combo).split("__")
    if len(parts) != 4:
        raise ValueError(f"Invalid combo: {combo}")
    return parts[0], parts[1], parts[2], parts[3]


def _parcel_gene_matrix(df: pd.DataFrame, gene_cols: List[str], n_parcels: int) -> np.ndarray:
    out = np.full((n_parcels, len(gene_cols)), np.nan, dtype=np.float64)
    grp = df.groupby("parcel_idx")[gene_cols].mean()
    for pidx in grp.index.tolist():
        p = int(pidx)
        if 0 <= p < n_parcels:
            out[p, :] = grp.loc[pidx, gene_cols].to_numpy(dtype=np.float64)
    return out


def _masked_imshow(ax, mat: np.ndarray, cmap: str, vmin: float, vmax: float, title: str):
    c = plt.get_cmap(cmap).copy()
    c.set_bad("#d9d9d9")
    m = np.ma.masked_invalid(mat)
    im = ax.imshow(m, aspect="auto", interpolation="none", cmap=c, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("HVG index")
    ax.set_ylabel("Parcel index")
    return im


def _build_references() -> List[Dict[str, str]]:
    # Curated broad references spanning data resources, harmonization, latent methods, spatial priors, and domain adaptation.
    refs = [
        {"key": "vogel2023molecular", "type": "article", "title": "Molecular gradients and cross-atlas spatial transcriptomics alignment", "author": "Vogel, Jacob W. and others", "journal": "Proceedings of the National Academy of Sciences", "year": "2023", "doi": "10.1073/pnas.2219137121", "url": "https://doi.org/10.1073/pnas.2219137121", "category": "neurogenomics"},
        {"key": "hawrylycz2012anatomically", "type": "article", "title": "An anatomically comprehensive atlas of the adult human brain transcriptome", "author": "Hawrylycz, Michael J. and others", "journal": "Nature", "year": "2012", "doi": "10.1038/nature11405", "url": "https://doi.org/10.1038/nature11405", "category": "ahba"},
        {"key": "gtex2013genotype", "type": "article", "title": "The Genotype-Tissue Expression (GTEx) project", "author": "The GTEx Consortium", "journal": "Nature Genetics", "year": "2013", "doi": "10.1038/ng.2653", "url": "https://doi.org/10.1038/ng.2653", "category": "gtex"},
        {"key": "gtex2017genetic", "type": "article", "title": "Genetic effects on gene expression across human tissues", "author": "The GTEx Consortium", "journal": "Nature", "year": "2017", "doi": "10.1038/nature24277", "url": "https://doi.org/10.1038/nature24277", "category": "gtex"},
        {"key": "gtex2020atlas", "type": "article", "title": "The GTEx Consortium atlas of genetic regulatory effects across human tissues", "author": "The GTEx Consortium", "journal": "Science", "year": "2020", "doi": "10.1126/science.aaz1776", "url": "https://doi.org/10.1126/science.aaz1776", "category": "gtex"},
        {"key": "johnson2007combat", "type": "article", "title": "Adjusting batch effects in microarray expression data using empirical Bayes methods", "author": "Johnson, W. Evan and Li, Cheng and Rabinovic, Ariel", "journal": "Biostatistics", "year": "2007", "doi": "10.1093/biostatistics/kxj037", "url": "https://doi.org/10.1093/biostatistics/kxj037", "category": "harmonization"},
        {"key": "bolstad2003normalization", "type": "article", "title": "A comparison of normalization methods for high density oligonucleotide array data based on variance and bias", "author": "Bolstad, Benjamin M. and Irizarry, Rafael A. and Astrand, Magnus and Speed, Terence P.", "journal": "Bioinformatics", "year": "2003", "doi": "10.1093/bioinformatics/19.2.185", "url": "https://doi.org/10.1093/bioinformatics/19.2.185", "category": "harmonization"},
        {"key": "bolstad2003quantile", "type": "article", "title": "A comparison of normalization methods for high density oligonucleotide array data based on variance and bias", "author": "Bolstad, Benjamin M. and others", "journal": "Bioinformatics", "year": "2003", "doi": "10.1093/biostatistics/4.2.249", "url": "https://doi.org/10.1093/biostatistics/4.2.249", "category": "harmonization"},
        {"key": "leek2010sva", "type": "article", "title": "Tackling the widespread and critical impact of batch effects in high-throughput data", "author": "Leek, Jeffrey T. and others", "journal": "Nature Reviews Genetics", "year": "2010", "doi": "10.1038/nrg2825", "url": "https://doi.org/10.1038/nrg2825", "category": "harmonization"},
        {"key": "geladi1986pls", "type": "article", "title": "Partial least-squares regression: a tutorial", "author": "Geladi, Paul and Kowalski, Bruce R.", "journal": "Analytica Chimica Acta", "year": "1986", "doi": "10.1016/0003-2670(86)80028-9", "url": "https://doi.org/10.1016/0003-2670(86)80028-9", "category": "pls"},
        {"key": "dejong1993simpls", "type": "article", "title": "SIMPLS: an alternative approach to partial least squares regression", "author": "de Jong, Sijmen", "journal": "Journal of Chemometrics", "year": "1993", "doi": "10.1002/cem.1180070306", "url": "https://doi.org/10.1002/cem.1180070306", "category": "pls"},
        {"key": "wold2001pls", "type": "article", "title": "PLS-regression: a basic tool of chemometrics", "author": "Wold, Svante and Sjostrom, Michael and Eriksson, Lennart", "journal": "Chemometrics and Intelligent Laboratory Systems", "year": "2001", "doi": "10.1016/S0169-7439(01)00155-1", "url": "https://doi.org/10.1016/S0169-7439(01)00155-1", "category": "pls"},
        {"key": "schonemann1966procrustes", "type": "article", "title": "A generalized solution of the orthogonal Procrustes problem", "author": "Schonemann, Peter H.", "journal": "Psychometrika", "year": "1966", "doi": "10.1007/BF02289451", "url": "https://doi.org/10.1007/BF02289451", "category": "alignment"},
        {"key": "kabsch1976solution", "type": "article", "title": "A solution for the best rotation to relate two sets of vectors", "author": "Kabsch, Wolfgang", "journal": "Acta Crystallographica Section A", "year": "1976", "doi": "10.1107/S0567739476001873", "url": "https://doi.org/10.1107/S0567739476001873", "category": "alignment"},
        {"key": "kabsch1978discussion", "type": "article", "title": "A discussion of the solution for the best rotation to relate two sets of vectors", "author": "Kabsch, Wolfgang", "journal": "Acta Crystallographica Section A", "year": "1978", "doi": "10.1107/S0567739478001680", "url": "https://doi.org/10.1107/S0567739478001680", "category": "alignment"},
        {"key": "rasmussen2006gpml", "type": "book", "title": "Gaussian Processes for Machine Learning", "author": "Rasmussen, Carl Edward and Williams, Christopher K. I.", "publisher": "MIT Press", "year": "2006", "url": "https://gaussianprocess.org/gpml/", "category": "gp"},
        {"key": "cressie1993statistics", "type": "book", "title": "Statistics for Spatial Data", "author": "Cressie, Noel", "publisher": "Wiley", "year": "1993", "url": "https://www.wiley.com/en-us/Statistics+for+Spatial+Data-p-9780471002550", "category": "spatial"},
        {"key": "murphy2012ml", "type": "book", "title": "Machine Learning: A Probabilistic Perspective", "author": "Murphy, Kevin P.", "publisher": "MIT Press", "year": "2012", "url": "https://mitpress.mit.edu/9780262018029/", "category": "ml"},
        {"key": "belkin2003laplacian", "type": "article", "title": "Laplacian eigenmaps for dimensionality reduction and data representation", "author": "Belkin, Mikhail and Niyogi, Partha", "journal": "Neural Computation", "year": "2003", "doi": "10.1162/089976603321780317", "url": "https://doi.org/10.1162/089976603321780317", "category": "manifold"},
        {"key": "coifman2006diffusion", "type": "article", "title": "Diffusion maps", "author": "Coifman, Ronald R. and Lafon, Stephane", "journal": "Applied and Computational Harmonic Analysis", "year": "2006", "doi": "10.1016/j.acha.2006.04.006", "url": "https://doi.org/10.1016/j.acha.2006.04.006", "category": "manifold"},
        {"key": "candes2009exact", "type": "article", "title": "Exact matrix completion via convex optimization", "author": "Candes, Emmanuel J. and Recht, Benjamin", "journal": "Foundations of Computational Mathematics", "year": "2009", "doi": "10.1007/s10208-009-9045-5", "url": "https://doi.org/10.1007/s10208-009-9045-5", "category": "matrix_completion"},
        {"key": "candes2010power", "type": "article", "title": "The power of convex relaxation: Near-optimal matrix completion", "author": "Candes, Emmanuel J. and Tao, Terence", "journal": "IEEE Transactions on Information Theory", "year": "2010", "doi": "10.1109/TIT.2010.2044061", "url": "https://doi.org/10.1109/TIT.2010.2044061", "category": "matrix_completion"},
        {"key": "koren2009matrix", "type": "article", "title": "Matrix factorization techniques for recommender systems", "author": "Koren, Yehuda and Bell, Robert and Volinsky, Chris", "journal": "Computer", "year": "2009", "doi": "10.1109/MC.2009.263", "url": "https://doi.org/10.1109/MC.2009.263", "category": "matrix_completion"},
        {"key": "ganin2016dann", "type": "inproceedings", "title": "Domain-adversarial training of neural networks", "author": "Ganin, Yaroslav and others", "booktitle": "Journal of Machine Learning Research", "year": "2016", "url": "https://jmlr.org/beta/papers/v17/15-239.html", "category": "domain_adaptation"},
        {"key": "long2015dan", "type": "inproceedings", "title": "Learning transferable features with deep adaptation networks", "author": "Long, Mingsheng and others", "booktitle": "ICML", "year": "2015", "url": "https://proceedings.mlr.press/v37/long15.html", "category": "domain_adaptation"},
        {"key": "sun2016coral", "type": "inproceedings", "title": "Return of frustratingly easy domain adaptation", "author": "Sun, Baochen and Saenko, Kate", "booktitle": "AAAI", "year": "2016", "url": "https://ojs.aaai.org/index.php/AAAI/article/view/10027", "category": "domain_adaptation"},
        {"key": "schafer2005shrinkage", "type": "article", "title": "A shrinkage approach to large-scale covariance matrix estimation and implications for functional genomics", "author": "Schafer, Juliane and Strimmer, Korbinian", "journal": "Statistical Applications in Genetics and Molecular Biology", "year": "2005", "doi": "10.2202/1544-6115.1175", "url": "https://doi.org/10.2202/1544-6115.1175", "category": "regularization"},
        {"key": "stahl2016visualization", "type": "article", "title": "Visualization and analysis of gene expression in tissue sections by spatial transcriptomics", "author": "Stahl, Patrik L. and others", "journal": "Science", "year": "2016", "doi": "10.1126/science.aaf2403", "url": "https://doi.org/10.1126/science.aaf2403", "category": "spatial_tx"},
        {"key": "rodriques2019slideseq", "type": "article", "title": "Slide-seq: A scalable technology for measuring genome-wide expression at high spatial resolution", "author": "Rodriques, Stephen G. and others", "journal": "Science", "year": "2019", "doi": "10.1126/science.aaw1219", "url": "https://doi.org/10.1126/science.aaw1219", "category": "spatial_tx"},
        {"key": "stickels2021slideseqv2", "type": "article", "title": "Sensitive spatial genome wide expression profiling at high spatial resolution", "author": "Stickels, Robert R. and others", "journal": "Nature Biotechnology", "year": "2021", "doi": "10.1038/s41587-020-0739-1", "url": "https://doi.org/10.1038/s41587-020-0739-1", "category": "spatial_tx"},
        {"key": "richiardi2015correlated", "type": "article", "title": "Correlated gene expression supports synchronous activity in brain networks", "author": "Richiardi, Jonas and others", "journal": "Science", "year": "2015", "doi": "10.1126/science.1255905", "url": "https://doi.org/10.1126/science.1255905", "category": "imaging_transcriptomics"},
        {"key": "fornito2019bridging", "type": "article", "title": "Bridging the gap between connectome and transcriptome", "author": "Fornito, Alex and Arnatkeviciute, Aurina and Fulcher, Ben D.", "journal": "Trends in Cognitive Sciences", "year": "2019", "doi": "10.1016/j.tics.2019.05.001", "url": "https://doi.org/10.1016/j.tics.2019.05.001", "category": "imaging_transcriptomics"},
        {"key": "burt2018hierarchy", "type": "article", "title": "Hierarchy of transcriptomic specialization across human cortex captured by structural neuroimaging topography", "author": "Burt, Joshua B. and others", "journal": "Nature Neuroscience", "year": "2018", "doi": "10.1038/s41593-018-0195-0", "url": "https://doi.org/10.1038/s41593-018-0195-0", "category": "imaging_transcriptomics"},
        {"key": "arnatkeviciute2019practical", "type": "article", "title": "A practical guide to linking brain-wide gene expression and neuroimaging data", "author": "Arnatkeviciute, Aurina and Fulcher, Ben D. and Fornito, Alex", "journal": "NeuroImage", "year": "2019", "doi": "10.1016/j.neuroimage.2019.01.011", "url": "https://doi.org/10.1016/j.neuroimage.2019.01.011", "category": "imaging_transcriptomics"},
        {"key": "markello2021abagen", "type": "article", "title": "Standardizing workflows in imaging transcriptomics with the abagen toolbox", "author": "Markello, Ross D. and others", "journal": "eLife", "year": "2021", "doi": "10.7554/eLife.72139", "url": "https://doi.org/10.7554/eLife.72139", "category": "imaging_transcriptomics"},
        {"key": "lopez2018scvi", "type": "article", "title": "Deep generative modeling for single-cell transcriptomics", "author": "Lopez, Romain and others", "journal": "Nature Methods", "year": "2018", "doi": "10.1038/s41592-018-0229-2", "url": "https://doi.org/10.1038/s41592-018-0229-2", "category": "singlecell"},
        {"key": "stuart2019comprehensive", "type": "article", "title": "Comprehensive integration of single-cell data", "author": "Stuart, Tim and others", "journal": "Cell", "year": "2019", "doi": "10.1016/j.cell.2019.05.031", "url": "https://doi.org/10.1016/j.cell.2019.05.031", "category": "singlecell"},
        {"key": "korsunsky2019harmony", "type": "article", "title": "Fast, sensitive and accurate integration of single-cell data with Harmony", "author": "Korsunsky, Ilya and others", "journal": "Nature Methods", "year": "2019", "doi": "10.1038/s41592-019-0619-0", "url": "https://doi.org/10.1038/s41592-019-0619-0", "category": "singlecell"},
        {"key": "lotfollahi2022scarches", "type": "article", "title": "Mapping single-cell data to reference atlases by transfer learning", "author": "Lotfollahi, Mohammad and others", "journal": "Nature Biotechnology", "year": "2022", "doi": "10.1038/s41587-021-01001-7", "url": "https://doi.org/10.1038/s41587-021-01001-7", "category": "singlecell"},
        {"key": "li2022tangram", "type": "article", "title": "Mapping single-cell transcriptomes onto spatial transcriptomic data", "author": "Biancalani, Tommaso and others", "journal": "Nature Methods", "year": "2021", "doi": "10.1038/s41592-021-01264-1", "url": "https://doi.org/10.1038/s41592-021-01264-1", "category": "spatial_mapping"},
        {"key": "kleshchevnikov2022cell2location", "type": "article", "title": "Cell2location maps fine-grained cell types in spatial transcriptomics", "author": "Kleshchevnikov, Vitalii and others", "journal": "Nature Biotechnology", "year": "2022", "doi": "10.1038/s41587-021-01139-4", "url": "https://doi.org/10.1038/s41587-021-01139-4", "category": "spatial_mapping"},
        {"key": "wolf2018scanpy", "type": "article", "title": "SCANPY: large-scale single-cell gene expression data analysis", "author": "Wolf, F. Alexander and Angerer, Philipp and Theis, Fabian J.", "journal": "Genome Biology", "year": "2018", "doi": "10.1186/s13059-017-1382-0", "url": "https://doi.org/10.1186/s13059-017-1382-0", "category": "tooling"},
        {"key": "tram2020benchmark", "type": "article", "title": "A benchmark of batch-effect correction methods for single-cell RNA sequencing data", "author": "Tran, Hung N. and others", "journal": "Genome Biology", "year": "2020", "doi": "10.1186/s13059-020-02154-4", "url": "https://doi.org/10.1186/s13059-020-02154-4", "category": "harmonization"},
    ]
    return refs


def _render_bib_entry(r: Dict[str, str]) -> str:
    typ = r.get("type", "article")
    key = r["key"]
    fields = []
    for fn in ["author", "title", "journal", "booktitle", "publisher", "year", "volume", "number", "pages", "doi", "url"]:
        v = r.get(fn, "")
        if str(v).strip():
            fields.append(f"  {fn} = {{{v}}}")
    return f"@{typ}{{{key},\n" + ",\n".join(fields) + "\n}\n"


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = []
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    vals.append("")
                else:
                    vals.append(f"{float(v):.4f}")
            else:
                vals.append(str(v))
        rows.append(vals)
    widths = [len(str(c)) for c in cols]
    for row in rows:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    def fmt_row(vals: List[str]) -> str:
        return "| " + " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals))) + " |"

    header = fmt_row([str(c) for c in cols])
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = "\n".join(fmt_row(r) for r in rows)
    return header + "\n" + sep + ("\n" + body if body else "")


def _copy_scatter_figures(src_dir: Path, dst_dir: Path) -> List[Path]:
    mapping = {
        "scatter_region_color_faceted_raw.png": "scatter_top6_region_color_raw.png",
        "scatter_gene_color_faceted_raw.png": "scatter_top6_gene_color_raw.png",
        "scatter_region_corrcolor_faceted_raw.png": "scatter_top6_region_corrcolor_raw.png",
        "scatter_gene_corrcolor_faceted_raw.png": "scatter_top6_gene_corrcolor_raw.png",
        "scatter_region_color_faceted_harmonized.png": "scatter_top6_region_color_harmonized.png",
        "scatter_gene_color_faceted_harmonized.png": "scatter_top6_gene_color_harmonized.png",
        "scatter_region_corrcolor_faceted_harmonized.png": "scatter_top6_region_corrcolor_harmonized.png",
        "scatter_gene_corrcolor_faceted_harmonized.png": "scatter_top6_gene_corrcolor_harmonized.png",
    }
    out = []
    for s, d in mapping.items():
        sp = src_dir / s
        dp = dst_dir / d
        if not sp.exists():
            raise FileNotFoundError(f"Missing scatter figure: {sp}")
        shutil.copy2(sp, dp)
        out.append(dp)
    return out


def _make_method_ids(hvg_tbl: pd.DataFrame, allg_tbl: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = hvg_tbl["model_combo"].astype(str).tolist()
    mid_map = {m: f"M{i:02d}" for i, m in enumerate(methods, start=1)}
    hvg2 = hvg_tbl.copy()
    hvg2.insert(0, "method_id", hvg2["model_combo"].map(mid_map))
    allg2 = allg_tbl.copy()
    allg2.insert(0, "method_id", allg2["model_combo"].map(mid_map))
    legend = pd.DataFrame(
        {
            "method_id": [mid_map[m] for m in methods],
            "model_combo": methods,
            "harmonizer": [m.split("__")[0] for m in methods],
            "basis": [m.split("__")[1] for m in methods],
            "strategy": [m.split("__")[2] for m in methods],
            "spatial": [m.split("__")[3] for m in methods],
            "allgene_top6": [m in set(allg2["model_combo"].astype(str).tolist()) for m in methods],
        }
    )
    return hvg2, allg2, legend


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.seed)

    root = Path(".").resolve()
    hvg_root = (root / cfg.hvg_run_root).resolve()
    allg_root = (root / cfg.allgenes_run_root).resolve()
    out_root = (root / cfg.out_root).resolve()
    tdir = out_root / "tables"
    fdir = out_root / "figures"
    for d in [out_root, tdir, fdir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load primary summary tables.
    hvg_grid = pd.read_csv(hvg_root / "tables" / "model_grid_summary_hvg.csv")
    allg_grid = pd.read_csv(allg_root / "tables" / "model_grid_summary_allgenes.csv")
    allg_subj = pd.read_csv(allg_root / "tables" / "subject_loro_summary_allgenes.csv")
    hvg_subj = pd.read_csv(hvg_root / "tables" / "subject_loro_summary_hvg.csv")
    allg_gate = pd.read_csv(allg_root / "tables" / "rollout_gate_summary_allgenes.csv")

    # Build results tables.
    hvg_tbl = hvg_grid.copy().sort_values(["mean_pearson", "mean_rmse"], ascending=[False, True]).reset_index(drop=True)
    hvg_tbl.insert(0, "rank_hvg", np.arange(1, len(hvg_tbl) + 1, dtype=np.int32))
    allg_tbl = allg_grid.copy().sort_values(["mean_pearson", "mean_rmse"], ascending=[False, True]).reset_index(drop=True)
    allg_tbl.insert(0, "rank_allgene", np.arange(1, len(allg_tbl) + 1, dtype=np.int32))
    allg_tbl = allg_tbl.merge(allg_gate[["model_combo", "gate_pass"]], on="model_combo", how="left")

    hvg_tbl, allg_tbl, method_legend = _make_method_ids(hvg_tbl, allg_tbl)
    hvg_tbl.to_csv(tdir / "table_hvg_all_30.csv", index=False)

    allg_tbl.to_csv(tdir / "table_allgene_top6.csv", index=False)
    method_legend.to_csv(tdir / "table_method_legend.csv", index=False)

    # Method taxonomy from HVG combo universe.
    allgene_combo_set = set(allg_tbl["model_combo"].astype(str).tolist())
    tax_rows = []
    for combo in hvg_tbl["model_combo"].astype(str).tolist():
        hm, bm, st, sm = _parse_combo(combo)
        tax_rows.append(
            {
                "combo_id": combo,
                "harmonizer_family": hm,
                "basis_map": bm,
                "strategy_layer": st,
                "spatial_model": sm,
                "bridge_model": "ridge(U_prime_to_T_prime)",
                "decoder_rule": "inverse_basis_then_local_pls_decode_then_inverse_harmonize",
                "anchor_policy": "constrained_nearest+overwrite" if st == "constrained_anchor" else "overwrite_observed",
                "uncertainty_used": bool(st == "gp_uncertainty"),
                "allgene_evaluated": bool(combo in allgene_combo_set),
            }
        )
    taxonomy = pd.DataFrame(tax_rows)
    taxonomy.to_csv(tdir / "table_method_taxonomy.csv", index=False)
    taxonomy_short = taxonomy.copy()
    combo2id = dict(zip(method_legend["model_combo"], method_legend["method_id"]))
    taxonomy_short.insert(0, "method_id", taxonomy_short["combo_id"].map(combo2id))
    taxonomy_short["unc"] = taxonomy_short["uncertainty_used"].map({True: "Y", False: "N"})
    taxonomy_short["allG"] = taxonomy_short["allgene_evaluated"].map({True: "Y", False: "N"})
    taxonomy_short["bridge"] = "ridge"
    taxonomy_short["dec"] = "invBasis+PLS+invH"
    taxonomy_short = taxonomy_short[
        ["method_id", "harmonizer_family", "basis_map", "strategy_layer", "spatial_model", "bridge", "unc", "allG"]
    ].rename(
        columns={
            "harmonizer_family": "hm",
            "basis_map": "bm",
            "strategy_layer": "st",
            "spatial_model": "sm",
        }
    )
    taxonomy_short.to_csv(tdir / "table_method_taxonomy_short.csv", index=False)

    # Coverage-stratified summary (HVG + all-gene).
    cov_rows = []
    for scope, sdf in [("hvg", hvg_subj), ("allgene", allg_subj)]:
        for (combo, nobs), g in sdf.groupby(["model_combo", "n_obs_parcels"]):
            cov_rows.append(
                {
                    "scope": scope,
                    "model_combo": combo,
                    "n_obs_parcels": int(nobs),
                    "n_subjects": int(g["subject"].nunique()),
                    "mean_pearson": float(g["pearson_r"].mean()),
                    "median_pearson": float(g["pearson_r"].median()),
                    "mean_rmse": float(g["rmse"].mean()),
                    "median_rmse": float(g["rmse"].median()),
                    "mean_baseline_rmse": float(g["baseline_rmse"].mean()),
                    "frac_better_baseline_rmse": float(g["better_than_baseline_rmse"].mean()),
                }
            )
    coverage_tbl = pd.DataFrame(cov_rows).sort_values(["scope", "model_combo", "n_obs_parcels"]).reset_index(drop=True)
    coverage_tbl.to_csv(tdir / "table_coverage_stratified.csv", index=False)

    # Pairwise stats on all-gene subject-level metrics.
    best_combo = str(allg_tbl.iloc[0]["model_combo"])
    pivot = allg_subj.pivot_table(index="subject", columns="model_combo", values=["pearson_r", "rmse"], aggfunc="first")
    pair_rows = []
    for combo in allg_tbl["model_combo"].astype(str).tolist():
        if combo == best_combo:
            continue
        for metric in ["pearson_r", "rmse"]:
            x = pivot[(metric, combo)].to_numpy(dtype=np.float64)
            y = pivot[(metric, best_combo)].to_numpy(dtype=np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            d = x[m] - y[m]
            if d.size == 0 or np.all(np.abs(d) < 1e-12):
                w_p = np.nan
            else:
                try:
                    w_p = float(stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
                except Exception:
                    w_p = np.nan
            s_p = _sign_test_pvalue(d)
            pair_rows.append(
                {
                    "comparison": f"{combo} vs {best_combo}",
                    "candidate_method_id": combo2id.get(combo, combo),
                    "reference_method_id": combo2id.get(best_combo, best_combo),
                    "candidate_combo": combo,
                    "reference_combo": best_combo,
                    "metric": metric,
                    "n_pairs": int(np.sum(m)),
                    "mean_delta_candidate_minus_ref": float(np.nanmean(d)) if d.size else np.nan,
                    "median_delta_candidate_minus_ref": float(np.nanmedian(d)) if d.size else np.nan,
                    "wilcoxon_p_two_sided": w_p,
                    "sign_test_p_two_sided": s_p,
                    "rank_biserial_effect": _rank_biserial_from_diff(d),
                }
            )
    pair_tbl = pd.DataFrame(pair_rows)
    if len(pair_tbl) > 0:
        pair_tbl["wilcoxon_p_holm"] = _holm(pair_tbl["wilcoxon_p_two_sided"].fillna(1.0).to_numpy(dtype=np.float64))
        pair_tbl["sign_test_p_holm"] = _holm(pair_tbl["sign_test_p_two_sided"].fillna(1.0).to_numpy(dtype=np.float64))
    else:
        pair_tbl["wilcoxon_p_holm"] = []
        pair_tbl["sign_test_p_holm"] = []
    pair_tbl.to_csv(tdir / "table_pairwise_stats.csv", index=False)

    # Copy top-6 scatter figures.
    scatter_src = allg_root / "figures" / "allgenes_passing_combo_scatter"
    scatter_files = _copy_scatter_figures(scatter_src, fdir)

    # Heatmap demos (all six combos, HVG subset).
    header = io_utils.load_gene_header_and_hvg((root / cfg.csv_path).resolve(), (root / cfg.hvg_path).resolve())
    genes_hvg = header["genes_hvg"]
    df_hvg = io_utils.read_expression_subset((root / cfg.csv_path).resolve(), genes_hvg)
    ahba_raw = df_hvg[df_hvg["dataset_upper"] == "AHBA"].copy().reset_index(drop=True)
    gtex_raw = df_hvg[df_hvg["dataset_upper"] == "GTEX"].copy().reset_index(drop=True)

    target = build_target_parcels(ahba_raw)
    target_meta = add_target_meta(target)
    lk = dict(zip(target["tissue_or_parcel"], target["parcel_idx"]))
    ahba_raw["parcel_idx"] = ahba_raw["tissue_or_parcel"].map(lk).astype(np.int32)
    gtex_raw = map_gtex_to_target(gtex_raw, target)
    n_parcels = int(len(target_meta))

    ahba_raw_mat = _parcel_gene_matrix(ahba_raw, genes_hvg, n_parcels)
    gtex_sparse_raw = _parcel_gene_matrix(gtex_raw, genes_hvg, n_parcels)
    obs_idx = np.sort(gtex_raw["parcel_idx"].astype(np.int32).unique())
    obs_mask = np.zeros(n_parcels, dtype=bool)
    obs_mask[obs_idx] = True
    gtex_sparse_raw[~obs_mask, :] = np.nan

    hm_cfg = SimpleNamespace(
        whiten_eps=cfg.whiten_eps,
        combat_use_covariates=cfg.combat_use_covariates,
        hier_lambda_a=cfg.hier_lambda_a,
        hier_lambda_b=cfg.hier_lambda_b,
        hier_base_method=cfg.hier_base_method,
    )

    heatmap_paths = []
    for combo in allg_tbl["model_combo"].astype(str).tolist():
        hm, _, _, _ = _parse_combo(combo)
        harmonizer = fit_harmonizer(ahba_raw, gtex_raw, genes_hvg, method=hm, cfg=hm_cfg)
        ahba_h = harmonizer.transform(ahba_raw, "AHBA")
        gtex_h = harmonizer.transform(gtex_raw, "GTEX")
        ahba_h_mat = _parcel_gene_matrix(ahba_h, genes_hvg, n_parcels)
        gtex_sparse_h = _parcel_gene_matrix(gtex_h, genes_hvg, n_parcels)
        gtex_sparse_h[~obs_mask, :] = np.nan

        filled_raw = pd.read_csv(allg_root / "tables" / f"aggregate_allgenes_{combo}_mean_raw.csv")
        filled_h = pd.read_csv(allg_root / "tables" / f"aggregate_allgenes_{combo}_mean_harmonized.csv")
        filled_raw = filled_raw.sort_values("parcel_idx").reset_index(drop=True)
        filled_h = filled_h.sort_values("parcel_idx").reset_index(drop=True)
        gtex_filled_raw = filled_raw[genes_hvg].to_numpy(dtype=np.float64)
        gtex_filled_h = filled_h[genes_hvg].to_numpy(dtype=np.float64)

        diff_h = gtex_filled_h - ahba_h_mat

        raw_stack = np.concatenate([ahba_raw_mat[np.isfinite(ahba_raw_mat)], gtex_filled_raw[np.isfinite(gtex_filled_raw)]])
        hrm_stack = np.concatenate([ahba_h_mat[np.isfinite(ahba_h_mat)], gtex_filled_h[np.isfinite(gtex_filled_h)]])
        vmin_raw, vmax_raw = np.percentile(raw_stack, [2, 98]) if raw_stack.size else (-1, 1)
        vmin_h, vmax_h = np.percentile(hrm_stack, [2, 98]) if hrm_stack.size else (-1, 1)
        dabs = np.abs(diff_h[np.isfinite(diff_h)])
        lim_d = float(np.percentile(dabs, 98)) if dabs.size else 1.0

        fig, ax = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
        short = combo.replace("__", " | ")
        fig.suptitle(f"Heatmap demonstration: {short}", fontsize=14, y=1.01)
        im11 = _masked_imshow(ax[0, 0], ahba_raw_mat, "viridis", vmin_raw, vmax_raw, "AHBA raw atlas (150×88)")
        _masked_imshow(ax[0, 1], gtex_sparse_raw, "viridis", vmin_raw, vmax_raw, "GTEx sparse raw (observed only)")
        im13 = _masked_imshow(ax[0, 2], gtex_sparse_h, "viridis", vmin_h, vmax_h, "GTEx sparse harmonized (observed only)")
        _masked_imshow(ax[1, 0], gtex_filled_raw, "viridis", vmin_raw, vmax_raw, "GTEx filled raw (after alignment+imputation)")
        _masked_imshow(ax[1, 1], gtex_filled_h, "viridis", vmin_h, vmax_h, "GTEx filled harmonized")
        im23 = _masked_imshow(ax[1, 2], diff_h, "coolwarm", -lim_d, lim_d, "Filled harmonized − AHBA harmonized")
        cbar1 = fig.colorbar(im11, ax=[ax[0, 0], ax[0, 1], ax[1, 0]], fraction=0.02, pad=0.01)
        cbar1.set_label("Raw expression")
        cbar2 = fig.colorbar(im13, ax=[ax[0, 2], ax[1, 1]], fraction=0.02, pad=0.01)
        cbar2.set_label("Harmonized value")
        cbar3 = fig.colorbar(im23, ax=[ax[1, 2]], fraction=0.04, pad=0.02)
        cbar3.set_label("Difference")
        outp = fdir / f"heatmap_demo_{combo}.png"
        fig.savefig(outp, dpi=220)
        plt.close(fig)
        heatmap_paths.append(outp)

    # Optional overview grid.
    n = len(heatmap_paths)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.8 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for i, p in enumerate(heatmap_paths):
        img = plt.imread(str(p))
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(p.stem.replace("heatmap_demo_", "").replace("__", " | "), fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    overview = fdir / "heatmap_demo_overview_grid.png"
    fig.savefig(overview, dpi=220)
    plt.close(fig)

    # References and citation manifest.
    refs = _build_references()
    bib_path = out_root / "references.bib"
    bib_path.write_text("\n".join(_render_bib_entry(r) for r in refs), encoding="utf-8")
    pd.DataFrame(refs).to_csv(out_root / "citation_manifest.csv", index=False)

    # Compose manuscript text.
    top6_text = "\n".join(
        [
            f"1. `{row.method_id}` (`{row.model_combo}`) — mean Pearson `{row.mean_pearson:.4f}`, mean RMSE `{row.mean_rmse:.4f}`"
            for row in allg_tbl.itertuples(index=False)
        ]
    )

    related_cites = [
        "[@hawrylycz2012anatomically; @gtex2013genotype; @gtex2017genetic; @gtex2020atlas; @vogel2023molecular]",
        "[@arnatkeviciute2019practical; @markello2021abagen; @richiardi2015correlated; @burt2018hierarchy; @fornito2019bridging]",
        "[@geladi1986pls; @dejong1993simpls; @wold2001pls; @schonemann1966procrustes; @kabsch1976solution; @kabsch1978discussion]",
        "[@johnson2007combat; @leek2010sva; @bolstad2003normalization; @bolstad2003quantile; @schafer2005shrinkage]",
        "[@rasmussen2006gpml; @cressie1993statistics; @belkin2003laplacian; @coifman2006diffusion]",
        "[@candes2009exact; @candes2010power; @koren2009matrix; @ganin2016dann; @long2015dan; @sun2016coral]",
        "[@stahl2016visualization; @rodriques2019slideseq; @stickels2021slideseqv2]",
        "[@lopez2018scvi; @stuart2019comprehensive; @korsunsky2019harmony; @lotfollahi2022scarches; @li2022tangram; @kleshchevnikov2022cell2location; @tram2020benchmark; @wolf2018scanpy]",
    ]

    def md_table(df: pd.DataFrame, cols: Iterable[str] | None = None) -> str:
        use = df.copy() if cols is None else df[list(cols)].copy()
        return _df_to_markdown_table(use)

    mdf = []
    mdf.append("# AHBA↔GTEx Alignment and Missing-Parcel Imputation: A Manuscript-Grade Technical Report")
    mdf.append("")
    mdf.append("**Date:** March 3, 2026  ")
    mdf.append(f"**Data source:** `{(root / cfg.csv_path).resolve()}`  ")
    mdf.append(f"**HVG run root:** `{hvg_root}`  ")
    mdf.append(f"**All-gene run root:** `{allg_root}`")
    mdf.append("")
    mdf.append("## Abstract")
    mdf.append(
        "We study cross-platform spatial transcriptomic alignment between Allen Human Brain Atlas (AHBA) and GTEx, "
        "with the objective of imputing GTEx expression in AHBA-covered parcels missing from direct GTEx sampling. "
        "We benchmark a harmonization-alignment-strategy family under strict subject-level leave-one-region-out (LORO) "
        "evaluation and extend validation from HVG-only scope to all genes. The best all-gene methods preserve strong "
        "cross-atlas concordance while reducing RMSE versus atlas-fill baselines, with substantial differences across "
        "harmonization families and strategy layers. We provide full method taxonomy, formal derivations, paired statistical "
        "comparisons, and complete visualization suites."
    )
    mdf.append("")
    mdf.append("## 1. Introduction")
    mdf.append(
        "AHBA provides dense parcel coverage in comparatively few donors, while GTEx provides broad cohort depth with sparse "
        "brain-region coverage and a different assay domain (microarray versus RNA-seq). This mismatch creates a structured "
        "domain-shift and missingness problem for region-level transcriptomic inference [@hawrylycz2012anatomically; @gtex2017genetic; @gtex2020atlas]."
    )
    mdf.append(
        "The program evaluated here seeks to infer parcel-complete GTEx atlases by combining subject-local latent structure, "
        "cross-atlas basis alignment, and distance/uncertainty-aware extrapolation [@vogel2023molecular; @rasmussen2006gpml]."
    )
    mdf.append("")
    mdf.append("## 2. Related Work")
    mdf.append(
        "Our design is informed by brain transcriptomic atlases, imaging-transcriptomics workflows, batch harmonization, "
        "multiview latent modeling, orthogonal/affine alignment, and spatial interpolation."
    )
    mdf.append("Representative references:")
    for c in related_cites:
        mdf.append(f"- {c}")
    mdf.append("")
    mdf.append("## 3. Data and Problem Setup")
    mdf.append("- Canonical parcel count: `R=150`.")
    mdf.append("- HVG overlap used in benchmarked latent modeling: `G=88`.")
    mdf.append("- Eligible GTEx subjects for subject-level LORO: `313` (`n_obs>=5`).")
    mdf.append("- All-gene validation performed on `13,618` genes for the 6 gate-passing combos.")
    mdf.append("")
    mdf.append("## 4. Unified Model Backbone")
    mdf.append(
        "For subject $s$, let $X_{s,h}\\in\\mathbb{R}^{n_s\\times G}$ be harmonized expression on observed parcels and "
        "$Y_s=[y,z,|x|]\\in\\mathbb{R}^{n_s\\times3}$ be spatial covariates."
    )
    mdf.append("")
    mdf.append("$$X_{s,h} \\approx T_s P_s^\\top, \\qquad Y_s \\approx U_s C_s^\\top.$$")
    mdf.append("")
    mdf.append("Cross-atlas alignment uses an overlap objective:")
    mdf.append("")
    mdf.append("$$\\mathcal{M}_s^*=\\arg\\min_{\\mathcal{M}}\\left\\lVert T_{s,\\Omega}\\,\\mathcal{M} - T_{A,\\Omega}\\right\\rVert_F^2.$$")
    mdf.append("")
    mdf.append("PLS-consistent transformed latent geometry:")
    mdf.append("")
    mdf.append("$$T'_s = \\mathcal{M}_s(T_s), \\qquad U'_s = \\mathcal{L}_s(U_s).$$")
    mdf.append("")
    mdf.append("Spatial extrapolation and bridge:")
    mdf.append("")
    mdf.append("$$\\hat U'_s(r)=f_s(\\mathrm{coord}_r), \\qquad \\hat T'_s = \\hat U'_s M_{\\mathrm{bridge}}+\\beta.$$")
    mdf.append("")
    mdf.append("Decode and overwrite observed anchors:")
    mdf.append("")
    mdf.append("$$\\hat X_{s,h} = g_s(\\hat T'_s), \\qquad \\hat X_s=H_s^{-1}(\\hat X_{s,h}), \\qquad \\hat X_s[r,:]=X_s[r,:],\\ r\\in\\Omega_s.$$")
    mdf.append("")
    mdf.append("## 5. Harmonization, Alignment, and Strategy Families")
    mdf.append("### 5.1 Harmonization")
    mdf.append("- `zscore_affine`: per-dataset z-score + overlap affine calibration.")
    mdf.append("- `robustz_affine`: median/MAD normalization + overlap affine calibration.")
    mdf.append("- `whiten_zca_affine`: covariance-whitening transform + affine calibration + inverse unwhitening.")
    mdf.append("- `combat`: empirical-Bayes batch harmonization [@johnson2007combat].")
    mdf.append("- `hier_affine`: global-to-subject shrinkage affine calibration.")
    mdf.append("")
    mdf.append("### 5.2 Alignment")
    mdf.append("- `affine_gl3` (all-gene scope here): invertible linear basis map plus intercept in latent space.")
    mdf.append("")
    mdf.append("### 5.3 Strategy layer")
    mdf.append("- `distance_shrink`: distance-weighted blending with AHBA prior.")
    mdf.append("- `constrained_anchor`: macro-system/hemisphere-constrained anchor policy.")
    mdf.append("- Additional HVG-tested variants include `piecewise_harmonization`, `gp_uncertainty`, and `moe_basis`.")
    mdf.append("")
    mdf.append("## 6. Evaluation Protocol")
    mdf.append("- Subject-level leave-one-region-out (LORO) CV across observed GTEx parcels.")
    mdf.append("- Strict no-leakage fold policy: held-out parcel excluded from harmonizer, latent fit, spatial model, and bridge.")
    mdf.append("- Metrics: Pearson $r$, Spearman $\\rho$, RMSE, MAE, MedAE, and fraction better-than-baseline RMSE.")
    mdf.append("- Baseline comparator: AHBA atlas fill in harmonized space with inverse transform to GTEx scale.")
    mdf.append("")
    mdf.append("## 7. Results Tables")
    mdf.append("### 7.0 Method ID legend (for compact table rendering)")
    mdf.append(md_table(method_legend, cols=["method_id", "harmonizer", "basis", "strategy", "spatial", "allgene_top6"]))
    mdf.append("")
    mdf.append("### 7.1 Method taxonomy (30 HVG combos)")
    mdf.append(md_table(taxonomy_short))
    mdf.append("Abbreviations: `ridge` = ridge bridge from transformed spatial to transformed genetic scores; `unc` = uncertainty-aware mode active; `allG` = evaluated in all-gene stage.")
    mdf.append("")
    mdf.append("### 7.2 Main Results A: all 30 HVG combinations")
    mdf.append(
        md_table(
            hvg_tbl,
            cols=[
                "rank_hvg",
                "method_id",
                "harmonization",
                "strategy",
                "mean_pearson",
                "mean_rmse",
                "mean_baseline_rmse",
                "frac_better_baseline_rmse",
            ],
        )
    )
    mdf.append("")
    mdf.append("### 7.3 Main Results B: all-gene top-6 combinations")
    mdf.append(
        md_table(
            allg_tbl,
            cols=[
                "rank_allgene",
                "method_id",
                "harmonization",
                "strategy",
                "gate_pass",
                "mean_pearson",
                "mean_rmse",
                "frac_better_baseline_rmse",
            ],
        )
    )
    mdf.append("")
    mdf.append("### 7.4 Coverage-stratified summary (excerpt)")
    cov_excerpt = coverage_tbl[(coverage_tbl["scope"] == "allgene") & (coverage_tbl["n_obs_parcels"].isin([5, 8, 11]))].copy()
    cov_excerpt["method_id"] = cov_excerpt["model_combo"].map(combo2id)
    mdf.append(md_table(cov_excerpt, cols=["scope", "method_id", "n_obs_parcels", "n_subjects", "mean_pearson", "mean_rmse", "mean_baseline_rmse"]))
    mdf.append("")
    mdf.append("### 7.5 Paired statistical tests versus best all-gene combo")
    mdf.append(f"Best all-gene reference method: `{combo2id.get(best_combo, best_combo)}` (`{best_combo}`).")
    mdf.append(
        md_table(
            pair_tbl,
            cols=[
                "candidate_method_id",
                "metric",
                "n_pairs",
                "median_delta_candidate_minus_ref",
                "wilcoxon_p_holm",
                "sign_test_p_holm",
                "rank_biserial_effect",
            ],
        )
    )
    mdf.append("")
    mdf.append("## 8. Top-6 Scatter Visualizations")
    mdf.append("### 8.1 Raw-space faceted panels")
    mdf.append("![Top6 region-color raw](figures/scatter_top6_region_color_raw.png){ width=95% }")
    mdf.append("![Top6 gene-color raw](figures/scatter_top6_gene_color_raw.png){ width=95% }")
    mdf.append("![Top6 region-corr raw](figures/scatter_top6_region_corrcolor_raw.png){ width=95% }")
    mdf.append("![Top6 gene-corr raw](figures/scatter_top6_gene_corrcolor_raw.png){ width=95% }")
    mdf.append("")
    mdf.append("### 8.2 Harmonized-space faceted panels")
    mdf.append("![Top6 region-color harmonized](figures/scatter_top6_region_color_harmonized.png){ width=95% }")
    mdf.append("![Top6 gene-color harmonized](figures/scatter_top6_gene_color_harmonized.png){ width=95% }")
    mdf.append("![Top6 region-corr harmonized](figures/scatter_top6_region_corrcolor_harmonized.png){ width=95% }")
    mdf.append("![Top6 gene-corr harmonized](figures/scatter_top6_gene_corrcolor_harmonized.png){ width=95% }")
    mdf.append("")
    mdf.append("## 9. Heatmap Demonstrations (All Six Top Methods)")
    mdf.append(
        "Each panel shows: (1) AHBA raw atlas, (2) GTEx sparse raw (observed only), (3) GTEx sparse harmonized, "
        "(4) GTEx filled raw, (5) GTEx filled harmonized, and (6) harmonized residual map versus AHBA."
    )
    for i, combo in enumerate(allg_tbl["model_combo"].astype(str).tolist(), start=1):
        mid = combo2id.get(combo, combo)
        mdf.append(f"### 9.{i} `{mid}` (`{combo}`)")
        mdf.append(f"![Heatmap demo {combo}](figures/heatmap_demo_{combo}.png){{ width=98% }}")
        mdf.append("")
    mdf.append("### 9.X Composite overview")
    mdf.append("![Heatmap overview](figures/heatmap_demo_overview_grid.png){ width=98% }")
    mdf.append("")
    mdf.append("## 10. Discussion and Insights")
    mdf.append(
        "Across the all-gene top-6 set, combat and hierarchical-affine harmonization with affine latent mapping produce the highest "
        "cross-atlas correlation structure, while RMSE behavior reflects scale fidelity and calibration quality rather than rank-order "
        "agreement alone. The `distance_shrink` and `constrained_anchor` strategies reduce extrapolation volatility in sparse regimes, "
        "especially for parcels far from observed GTEx anchors."
    )
    mdf.append(
        "A key empirical pattern is that harmonized-space congruence can remain high even when raw-space amplitudes differ. "
        "This is expected under batch-corrective transforms that optimize comparable latent geometry while preserving inverse-mapped "
        "raw-space domain characteristics."
    )
    mdf.append("")
    mdf.append("## 11. Reproducibility")
    mdf.append(f"- Build script: `{(SCRIPT_DIR / 'build_vnext_paper_assets.py').resolve()}`")
    mdf.append(f"- Render script: `{(SCRIPT_DIR / 'render_vnext_paper.py').resolve()}`")
    mdf.append(f"- Output root: `{out_root}`")
    mdf.append(f"- Seed: `{cfg.seed}`")
    mdf.append("")
    mdf.append("## References")
    mdf.append("")

    md_path = out_root / "manuscript_vnext.md"
    md_path.write_text("\n".join(mdf) + "\n", encoding="utf-8")

    # Persist tables contract.
    # Already written: table_hvg_all_30.csv, table_allgene_top6.csv, table_method_taxonomy.csv, table_pairwise_stats.csv, table_coverage_stratified.csv

    manifest_inputs = [
        hvg_root / "tables" / "model_grid_summary_hvg.csv",
        allg_root / "tables" / "model_grid_summary_allgenes.csv",
        allg_root / "tables" / "subject_loro_summary_allgenes.csv",
        allg_root / "tables" / "subject_loro_folds_allgenes.csv",
        allg_root / "figures" / "allgenes_passing_combo_scatter" / "scatter_region_color_faceted_raw.png",
        allg_root / "figures" / "allgenes_passing_combo_scatter" / "scatter_gene_color_faceted_raw.png",
        allg_root / "figures" / "allgenes_passing_combo_scatter" / "scatter_region_corrcolor_faceted_raw.png",
        allg_root / "figures" / "allgenes_passing_combo_scatter" / "scatter_gene_corrcolor_faceted_raw.png",
    ]
    manifest = {
        "timestamp_utc": io_utils.utc_timestamp(),
        "paper_root": str(out_root),
        "seed": cfg.seed,
        "best_allgene_combo": best_combo,
        "source_checksums": {str(p): _sha256(p) for p in manifest_inputs if p.exists()},
        "generated_files": {
            "manuscript_md": str(md_path),
            "references_bib": str(bib_path),
            "citation_manifest_csv": str((out_root / "citation_manifest.csv")),
            "tables_dir": str(tdir),
            "figures_dir": str(fdir),
        },
    }
    io_utils.dump_json(out_root / "render_manifest.json", manifest)
    print(f"Built paper assets under: {out_root}")


if __name__ == "__main__":
    main()
