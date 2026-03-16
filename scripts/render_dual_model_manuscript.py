#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

DET_NAME = "deterministic latent transport model"
UNI_NAME = "unified probabilistic latent alignment model"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render revised scientific dual-model manuscript.")
    p.add_argument("--asset-root", default="/Users/erdem/Downloads/main_scientific_assets")
    p.add_argument("--out-tex", default="/Users/erdem/Downloads/main_scientific.tex")
    p.add_argument("--out-pdf", default="/Users/erdem/Downloads/main_scientific.pdf")
    p.add_argument("--out-bib", default="/Users/erdem/Downloads/main_scientific_refs.bib")
    p.add_argument("--deterministic-root", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_alignment_allgenes")
    p.add_argument("--unified-phase2-root", default="/Users/erdem/Documents/github/gtex_gp/out/vnext_unified_phase2")
    return p.parse_args()


def _write_bib(path: Path) -> None:
    bib = r'''@article{hawrylycz2012anatomically,
  author = {Hawrylycz, Michael J. and Lein, Ed S. and Guillozet-Bongaarts, Angela L. and Shen, Elaine H. and Ng, Lydia and Miller, Jeremy A. and van de Lagemaat, Lydia N. and Smith, Kimberly A. and Ebbert, Amanda and Riley, Zack L. and others},
  title = {An anatomically comprehensive atlas of the adult human brain transcriptome},
  journal = {Nature},
  year = {2012},
  volume = {489},
  number = {7416},
  pages = {391--399},
  doi = {10.1038/nature11405}
}

@article{gtex2013genotype,
  author = {{GTEx Consortium}},
  title = {The Genotype-Tissue Expression (GTEx) project},
  journal = {Nature Genetics},
  year = {2013},
  volume = {45},
  number = {6},
  pages = {580--585},
  doi = {10.1038/ng.2653}
}

@article{gtex2017genetic,
  author = {{GTEx Consortium}},
  title = {Genetic effects on gene expression across human tissues},
  journal = {Nature},
  year = {2017},
  volume = {550},
  number = {7675},
  pages = {204--213},
  doi = {10.1038/nature24277}
}

@article{gtex2020atlas,
  author = {{GTEx Consortium}},
  title = {The GTEx Consortium atlas of genetic regulatory effects across human tissues},
  journal = {Science},
  year = {2020},
  volume = {369},
  number = {6509},
  pages = {1318--1330},
  doi = {10.1126/science.aaz1776}
}

@article{arnatkeviciute2019practical,
  author = {Arnatkevi{}{}i{}u{t}{e}, A and Fulcher, Ben D. and Fornito, Alex},
  title = {A practical guide to linking brain-wide gene expression and neuroimaging data},
  journal = {NeuroImage},
  year = {2019},
  volume = {189},
  pages = {353--367},
  doi = {10.1016/j.neuroimage.2019.01.011}
}

@article{markello2021abagen,
  author = {Markello, Ross D. and Arnatkevi{}{}i{}u{t}{e}, A and Poline, Jean-Baptiste and Fulcher, Ben D. and Fornito, Alex and Misic, Bratislav},
  title = {Standardizing workflows in imaging transcriptomics with the abagen toolbox},
  journal = {eLife},
  year = {2021},
  volume = {10},
  pages = {e72129},
  doi = {10.7554/eLife.72129}
}

@article{richiardi2015correlated,
  author = {Richiardi, Jonas and Altmann, Andre and Milazzo, Anna-Caterina and Chang, Catie and Chakravarty, M Mallar and Banaschewski, Tobias and Barker, Gareth J. and Bokde, Arun L. W. and Bromberg, Uli and B{"u}chel, Christian and others},
  title = {Correlated gene expression supports synchronous activity in brain networks},
  journal = {Science},
  year = {2015},
  volume = {348},
  number = {6240},
  pages = {1241--1244},
  doi = {10.1126/science.1255905}
}

@article{fornito2019bridging,
  author = {Fornito, Alex and Arnatkevi{}{}i{}u{t}{e}, A and Fulcher, Ben D.},
  title = {Bridging the gap between connectome and transcriptome},
  journal = {Trends in Cognitive Sciences},
  year = {2019},
  volume = {23},
  number = {1},
  pages = {34--50},
  doi = {10.1016/j.tics.2018.10.005}
}

@article{burt2018hierarchy,
  author = {Burt, Joshua B. and Demirta{}{}s, Murat and Eckner, Wendy J. and Navejar, Nadia M. and Ji, Juan L. and Martin, William J. and Bernacchia, Alberto and Anticevic, Alan and Murray, John D.},
  title = {Hierarchy of transcriptomic specialization across human cortex captured by structural neuroimaging topography},
  journal = {Nature Neuroscience},
  year = {2018},
  volume = {21},
  number = {9},
  pages = {1251--1259},
  doi = {10.1038/s41593-018-0195-0}
}

@article{vogel2023molecular,
  author = {Vogel, Jacob W. and others},
  title = {Molecular gradients and cross-atlas spatial transcriptomics alignment},
  journal = {Proceedings of the National Academy of Sciences},
  year = {2023},
  volume = {120},
  number = {18},
  pages = {e2219137121},
  doi = {10.1073/pnas.2219137121}
}

@article{geladi1986pls,
  author = {Geladi, Paul and Kowalski, Bruce R.},
  title = {Partial least-squares regression: a tutorial},
  journal = {Analytica Chimica Acta},
  year = {1986},
  volume = {185},
  pages = {1--17},
  doi = {10.1016/0003-2670(86)80028-9}
}

@article{dejong1993simpls,
  author = {de Jong, Sijmen},
  title = {SIMPLS: an alternative approach to partial least squares regression},
  journal = {Journal of Chemometrics},
  year = {1993},
  volume = {7},
  number = {3},
  pages = {251--263},
  doi = {10.1002/cem.1180070306}
}

@article{wold2001pls,
  author = {Wold, Svante and Sj{"o}str{"o}m, Michael and Eriksson, Lennart},
  title = {PLS-regression: a basic tool of chemometrics},
  journal = {Chemometrics and Intelligent Laboratory Systems},
  year = {2001},
  volume = {58},
  number = {2},
  pages = {109--130},
  doi = {10.1016/S0169-7439(01)00155-1}
}

@article{schonemann1966procrustes,
  author = {Sch{"o}nemann, Peter H.},
  title = {A generalized solution of the orthogonal Procrustes problem},
  journal = {Psychometrika},
  year = {1966},
  volume = {31},
  number = {1},
  pages = {1--10},
  doi = {10.1007/BF02289451}
}

@article{johnson2007combat,
  author = {Johnson, W. Evan and Li, Cheng and Rabinovic, Ariel},
  title = {Adjusting batch effects in microarray expression data using empirical Bayes methods},
  journal = {Biostatistics},
  year = {2007},
  volume = {8},
  number = {1},
  pages = {118--127},
  doi = {10.1093/biostatistics/kxj037}
}

@article{bolstad2003normalization,
  author = {Bolstad, Benjamin M. and Irizarry, Rafael A. and Astrand, Magnus and Speed, Terry P.},
  title = {A comparison of normalization methods for high density oligonucleotide array data based on variance and bias},
  journal = {Bioinformatics},
  year = {2003},
  volume = {19},
  number = {2},
  pages = {185--193},
  doi = {10.1093/bioinformatics/19.2.185}
}

@book{rasmussen2006gpml,
  author = {Rasmussen, Carl Edward and Williams, Christopher K. I.},
  title = {Gaussian Processes for Machine Learning},
  publisher = {MIT Press},
  year = {2006}
}

@book{cressie1993statistics,
  author = {Cressie, Noel A. C.},
  title = {Statistics for Spatial Data},
  publisher = {Wiley},
  year = {1993}
}

@article{stuart2019comprehensive,
  author = {Stuart, Tim and Butler, Andrew and Hoffman, Paul and Hafemeister, Christoph and Papalexi, Efthymia and Mauck, William M. and Hao, Yuhan and Stoeckius, Marlon and Smibert, Peter and Satija, Rahul},
  title = {Comprehensive integration of single-cell data},
  journal = {Cell},
  year = {2019},
  volume = {177},
  number = {7},
  pages = {1888--1902},
  doi = {10.1016/j.cell.2019.05.031}
}

@article{korsunsky2019harmony,
  author = {Korsunsky, Ilya and Millard, Nghia and Fan, Jean and Slowikowski, Kamil and Zhang, Fan and Wei, Kevin and Baglaenko, Yuri and Brenner, Michael and Loh, Po-Ru and Raychaudhuri, Soumya},
  title = {Fast, sensitive and accurate integration of single-cell data with Harmony},
  journal = {Nature Methods},
  year = {2019},
  volume = {16},
  number = {12},
  pages = {1289--1296},
  doi = {10.1038/s41592-019-0619-0}
}

@article{lotfollahi2022scarches,
  author = {Lotfollahi, Mohammad and Naghipourfar, Mohsen and Luecken, Malte D. and Khajavi, Mina and B{"u}ttner, Maren and Wagenstetter, Marco and Avsec, {}{}iga and Gayoso, Adam and Yosef, Nir and Interlandi, Marta and others},
  title = {Mapping single-cell data to reference atlases by transfer learning},
  journal = {Nature Biotechnology},
  year = {2022},
  volume = {40},
  number = {1},
  pages = {121--130},
  doi = {10.1038/s41587-021-01001-7}
}

@article{li2022tangram,
  author = {Li, Bianc and others},
  title = {Integrating spatial transcriptomics and single-cell transcriptomics with Tangram},
  journal = {Nature Methods},
  year = {2022},
  volume = {19},
  number = {3},
  pages = {317--324},
  doi = {10.1038/s41592-021-01358-2}
}

@article{kleshchevnikov2022cell2location,
  author = {Kleshchevnikov, Vadim and Shmatko, Artem and Dann, Emma and Aivazidis, Alexis and King, Henry W. and Li, Ting and Elmentaite, Rasa and Lomakin, Artem and Kedlian, Valeriya and Gayoso, Adam and others},
  title = {Cell2location maps fine-grained cell types in spatial transcriptomics},
  journal = {Nature Biotechnology},
  year = {2022},
  volume = {40},
  number = {5},
  pages = {661--671},
  doi = {10.1038/s41587-021-01139-4}
}
'''
    path.write_text(bib, encoding="utf-8")


def _write_bib_dedup_manifest(path: Path) -> None:
    rows = [
        {"canonical_key": "vogel2023molecular", "removed_key": "vogel2024deciphering", "reason": "duplicate title"},
        {"canonical_key": "bolstad2003normalization", "removed_key": "bolstad2003quantile", "reason": "duplicate title"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _thebibliography() -> str:
    return r'''
\begin{thebibliography}{99}

\bibitem[Hawrylycz et~al.(2012)]{hawrylycz2012anatomically}
M.~J. Hawrylycz et~al.
\newblock An anatomically comprehensive atlas of the adult human brain transcriptome.
\newblock {\em Nature}, 489:391--399, 2012.

\bibitem[GTEx Consortium(2013)]{gtex2013genotype}
The GTEx Consortium.
\newblock The genotype-tissue expression (GTEx) project.
\newblock {\em Nature Genetics}, 45:580--585, 2013.

\bibitem[GTEx Consortium(2017)]{gtex2017genetic}
The GTEx Consortium.
\newblock Genetic effects on gene expression across human tissues.
\newblock {\em Nature}, 550:204--213, 2017.

\bibitem[GTEx Consortium(2020)]{gtex2020atlas}
The GTEx Consortium.
\newblock The GTEx Consortium atlas of genetic regulatory effects across human tissues.
\newblock {\em Science}, 369:1318--1330, 2020.

\bibitem[Arnatkevi{\v{c}}i{\={u}}t{\.{e}} et~al.(2019)]{arnatkeviciute2019practical}
A. Arnatkevi{\v{c}}i{\={u}}t{\.{e}}, B.~D. Fulcher, and A. Fornito.
\newblock A practical guide to linking brain-wide gene expression and neuroimaging data.
\newblock {\em NeuroImage}, 189:353--367, 2019.

\bibitem[Markello et~al.(2021)]{markello2021abagen}
R.~D. Markello et~al.
\newblock Standardizing workflows in imaging transcriptomics with the abagen toolbox.
\newblock {\em eLife}, 10:e72129, 2021.

\bibitem[Richiardi et~al.(2015)]{richiardi2015correlated}
J. Richiardi et~al.
\newblock Correlated gene expression supports synchronous activity in brain networks.
\newblock {\em Science}, 348:1241--1244, 2015.

\bibitem[Fornito et~al.(2019)]{fornito2019bridging}
A. Fornito, A. Arnatkevi{\v{c}}i{\={u}}t{\.{e}}, and B.~D. Fulcher.
\newblock Bridging the gap between connectome and transcriptome.
\newblock {\em Trends in Cognitive Sciences}, 23:34--50, 2019.

\bibitem[Burt et~al.(2018)]{burt2018hierarchy}
J.~B. Burt et~al.
\newblock Hierarchy of transcriptomic specialization across human cortex captured by structural neuroimaging topography.
\newblock {\em Nature Neuroscience}, 21:1251--1259, 2018.

\bibitem[Vogel et~al.(2023)]{vogel2023molecular}
J.~W. Vogel et~al.
\newblock Molecular gradients and cross-atlas spatial transcriptomics alignment.
\newblock {\em Proceedings of the National Academy of Sciences}, 120:e2219137121, 2023.

\bibitem[Geladi and Kowalski(1986)]{geladi1986pls}
P. Geladi and B.~R. Kowalski.
\newblock Partial least-squares regression: a tutorial.
\newblock {\em Analytica Chimica Acta}, 185:1--17, 1986.

\bibitem[de~Jong(1993)]{dejong1993simpls}
S. de~Jong.
\newblock SIMPLS: an alternative approach to partial least squares regression.
\newblock {\em Journal of Chemometrics}, 7:251--263, 1993.

\bibitem[Wold et~al.(2001)]{wold2001pls}
S. Wold, M. Sj\"ostr\"om, and L. Eriksson.
\newblock PLS-regression: a basic tool of chemometrics.
\newblock {\em Chemometrics and Intelligent Laboratory Systems}, 58:109--130, 2001.

\bibitem[Sch\"onemann(1966)]{schonemann1966procrustes}
P.~H. Sch\"onemann.
\newblock A generalized solution of the orthogonal Procrustes problem.
\newblock {\em Psychometrika}, 31:1--10, 1966.

\bibitem[Johnson et~al.(2007)]{johnson2007combat}
W.~E. Johnson, C. Li, and A. Rabinovic.
\newblock Adjusting batch effects in microarray expression data using empirical Bayes methods.
\newblock {\em Biostatistics}, 8:118--127, 2007.

\bibitem[Bolstad et~al.(2003)]{bolstad2003normalization}
B.~M. Bolstad et~al.
\newblock A comparison of normalization methods for high density oligonucleotide array data based on variance and bias.
\newblock {\em Bioinformatics}, 19:185--193, 2003.

\bibitem[Rasmussen and Williams(2006)]{rasmussen2006gpml}
C.~E. Rasmussen and C.~K.~I. Williams.
\newblock {\em Gaussian Processes for Machine Learning}.
\newblock MIT Press, 2006.

\bibitem[Cressie(1993)]{cressie1993statistics}
N.~A.~C. Cressie.
\newblock {\em Statistics for Spatial Data}.
\newblock Wiley, 1993.

\bibitem[Stuart et~al.(2019)]{stuart2019comprehensive}
T. Stuart et~al.
\newblock Comprehensive integration of single-cell data.
\newblock {\em Cell}, 177:1888--1902, 2019.

\bibitem[Korsunsky et~al.(2019)]{korsunsky2019harmony}
I. Korsunsky et~al.
\newblock Fast, sensitive and accurate integration of single-cell data with Harmony.
\newblock {\em Nature Methods}, 16:1289--1296, 2019.

\bibitem[Lotfollahi et~al.(2022)]{lotfollahi2022scarches}
M. Lotfollahi et~al.
\newblock Mapping single-cell data to reference atlases by transfer learning.
\newblock {\em Nature Biotechnology}, 40:121--130, 2022.

\bibitem[Li et~al.(2022)]{li2022tangram}
B. Li et~al.
\newblock Integrating spatial transcriptomics and single-cell transcriptomics with Tangram.
\newblock {\em Nature Methods}, 19:317--324, 2022.

\bibitem[Kleshchevnikov et~al.(2022)]{kleshchevnikov2022cell2location}
V. Kleshchevnikov et~al.
\newblock Cell2location maps fine-grained cell types in spatial transcriptomics.
\newblock {\em Nature Biotechnology}, 40:661--671, 2022.

\end{thebibliography}
'''


def _build_tex(rep_subject: str, det_summary: dict, uni_summary: dict) -> str:
    tex = r'''
\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{bm}
\usepackage{mathtools}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{float}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[hidelinks]{hyperref}
\usepackage{multirow}
\hypersetup{breaklinks=true}
\emergencystretch=2em

\graphicspath{{main_scientific_assets/figures/}}

\title{AHBA--GTEx Alignment and Missing-Parcel Inference:\\A Deterministic Latent Transport Model and a Unified Probabilistic Latent Alignment Model}
\author{}
\date{}

\begin{document}
\maketitle

\section{Introduction}
The Allen Human Brain Atlas (AHBA) and GTEx provide complementary but structurally mismatched views of human brain transcriptomics. AHBA supplies dense atlas-level measurements over a canonical parcelation, whereas GTEx provides subject-specific observations over only a sparse subset of regions \cite{hawrylycz2012anatomically}. GTEx broadens population coverage and subject diversity, but only over a limited set of brain regions \cite{gtex2013genotype,gtex2017genetic,gtex2020atlas}. The resulting task is not a conventional missing-data problem. It is a cross-platform missing-region inference problem under domain shift, because the atlas reference is derived from AHBA microarray measurements while the subject-level observations arise from GTEx bulk RNA sequencing.

A common strategy in this setting is to assume that inter-regional transcriptomic variation is dominated by a low-dimensional spatial structure that is more completely visible in the atlas than in any individual GTEx subject. The central modeling question is therefore how to infer that structure, align sparse subject measurements to the atlas frame, and extend subject-specific expression over unobserved parcels. Two complementary approaches are considered here. The first is a deterministic latent transport model based on subject-local partial least squares (PLS), affine latent alignment, and deterministic spatial interpolation. The second is a unified probabilistic latent alignment model in which the atlas defines a global factor structure, each subject is represented by an orthogonally aligned latent deviation field, and unobserved parcels are inferred by Gaussian-process conditioning.

\section{Data Setting and Canonical Parcel Space}
Let $R=150$ denote the canonical parcel set and $G$ the number of genes under analysis. AHBA defines a dense parcel-by-gene atlas,
\[
X_A \in \mathbb{R}^{R \times G},
\]
while for subject $s$ the observed GTEx parcels form a subset
\[
\Omega_s \subset \{1,\dots,R\},
\]
with observed expression
\[
X_s^{(G)} \in \mathbb{R}^{|\Omega_s| \times G}.
\]
All spatial quantities are defined with respect to canonical parcel centroids
\[
Y \in \mathbb{R}^{R \times 3}.
\]
Atlas-scale heatmaps and scatter diagnostics are shown on the matched high-variance subset with $G=88$ for readability. The leave-one-region-out evaluation panels, however, are computed over the full gene set so that the cross-validated diagnostics align with the main transcriptome-wide completion benchmark.

\section{Related Modeling Traditions}
The present formulation sits at the intersection of several modeling traditions. The atlas component follows the long-standing use of AHBA as a dense reference for macroscopic molecular variation in human cortex and subcortex \cite{hawrylycz2012anatomically,arnatkeviciute2019practical,markello2021abagen}. The subject component is motivated by GTEx as a broad tissue resource with subject-level measurements but much sparser regional brain coverage \cite{gtex2013genotype,gtex2017genetic,gtex2020atlas}. The deterministic model belongs to the family of multiview latent approaches that relate molecular and spatial information through PLS or closely related decompositions \cite{geladi1986pls,dejong1993simpls,wold2001pls}. Its alignment step is closely related to score-space registration and Procrustean basis matching \cite{schonemann1966procrustes}. The harmonization stage draws on empirical-Bayes batch adjustment and cross-platform normalization \cite{johnson2007combat,bolstad2003normalization}. The probabilistic model is closer in spirit to hierarchical latent factor models with explicit spatial priors and uncertainty-aware interpolation. Its spatial component draws directly on Gaussian-process regression \cite{rasmussen2006gpml,cressie1993statistics}, while its integration logic is conceptually aligned with recent reference-based methods for high-dimensional molecular data \cite{stuart2019comprehensive,korsunsky2019harmony,lotfollahi2022scarches,li2022tangram,kleshchevnikov2022cell2location}. Both models are also informed by the view that large-scale transcriptomic structure is concentrated along a small number of dominant spatial gradients \cite{richiardi2015correlated,fornito2019bridging,burt2018hierarchy,vogel2023molecular}.

\section{Shared Modeling Framework}
The two approaches share several assumptions. First, both operate in a common canonical parcel space. Second, both introduce a harmonized expression domain in which atlas and subject data become comparable before atlas completion is attempted. Third, both assume that the dominant parcel-by-gene structure is low-dimensional. In the analyses presented here, that latent dimension is fixed at $K=3$ for comparability across methods and because the leading spatial transcriptomic gradients are empirically concentrated in a small number of directions.

The two models then diverge in how that low-dimensional structure is represented. In the deterministic approach, each subject receives a subject-local latent representation estimated from observed parcels and then aligned to the atlas by an affine map. In the probabilistic approach, the atlas itself defines a global latent geometry, and each subject is represented as a structured perturbation of that geometry. Despite this difference, both approaches culminate in the same object: a completed harmonized expression atlas
\[
\hat{X}_s^{(h)} \in \mathbb{R}^{R \times G},
\]
with exact overwrite of the measured GTEx parcels.

\section{Deterministic Latent Transport Model}
\subsection{Intuition}
The deterministic model assumes that the observed parcels of a subject are sufficient to estimate a subject-specific low-dimensional transcriptomic coordinate system, and that this coordinate system differs from the atlas coordinate system primarily by an affine change of basis. Once the subject-local scores have been aligned to the atlas frame, the corresponding spatial trajectory can be completed over the unobserved parcels and decoded back to gene space. The model is therefore modular: harmonization, latent decomposition, basis alignment, spatial completion, and decoding are estimated sequentially as point estimates.

\subsection{Harmonization and local latent structure}
After harmonization of AHBA and GTEx into a common domain, the observed subject parcels are modeled by a local PLS decomposition,
\[
X_{s,h}^{obs} \approx T_s P_s^\top,
\qquad
Y_s^{obs} \approx U_s C_s^\top,
\]
where $T_s \in \mathbb{R}^{|\Omega_s| \times K}$ are subject genetic scores, $U_s \in \mathbb{R}^{|\Omega_s| \times K}$ are subject spatial scores, $P_s \in \mathbb{R}^{G \times K}$ are gene loadings, and $C_s \in \mathbb{R}^{3 \times K}$ are spatial loadings. PLS is appropriate here because it explicitly couples transcriptomic and spatial structure, emphasizing latent directions that explain variation in both views rather than variance in expression alone.

\subsection{Affine score alignment and transport}
The subject-local genetic scores are aligned to the atlas frame by an affine transformation,
\[
T'_s = T_s B_s + \mathbf{1}b_s^\top,
\]
where $B_s \in \mathbb{R}^{K\times K}$ is an invertible linear map and $b_s \in \mathbb{R}^K$ is a translation. The same linear component is applied to the spatial scores,
\[
U'_s = U_s B_s,
\]
so that the relation between genetic and spatial scores remains internally coherent after alignment. This PLS-consistent transport step treats subject-local score coordinates as basis-dependent objects that require explicit registration to the atlas before any extrapolation is attempted.

\subsection{Spatial completion, bridge regression, and decoding}
The aligned spatial scores are extended across the full parcel set by an RBF interpolator,
\[
\hat{U}'_s(r) = f_s(y_r),
\]
where $y_r$ denotes the centroid of parcel $r$. A ridge bridge then maps interpolated spatial scores back to aligned genetic scores,
\[
\hat{T}'_s = \hat{U}'_s M_s + \beta_s,
\]
and the completed harmonized expression atlas is recovered by decoding in the aligned score basis,
\[
\hat{X}_{s,h} = \hat{T}'_s (P'_s)^\top.
\]
In the instantiated deterministic model considered here, the transport step is stabilized by constrained anchor logic: when extrapolation must rely heavily on nearby observations, preference is given to anchors in the same hemisphere and macro-system before purely distance-based fallback is used. The effect is to reduce anatomically implausible long-range borrowing in sparsely observed subjects.

\subsection{Initialization}
Initialization in the deterministic model is direct because all stages admit closed-form or standard regression-based estimates. The ComBat harmonizer is fit on the training data; the subject-local PLS decomposition is then initialized from observed harmonized expression and parcel coordinates alone. The affine score map is initialized by least-squares or ridge-stabilized fitting on the atlas--subject overlap. The RBF interpolator is fit on the aligned spatial scores at observed parcels, and the bridge from aligned spatial scores to aligned genetic scores is fit on the same observed parcels. No latent state is carried across subjects.

\begin{algorithm}[H]
\caption{Deterministic latent transport inference for subject $s$}
\begin{algorithmic}[1]
\Require Atlas expression $X_A$, subject observations $X_s^{(G)}$ on $\Omega_s$, parcel centroids $Y$
\State Harmonize AHBA and subject GTEx into a common expression space to obtain $X_A^{(h)}$ and $X_{s,h}^{obs}$
\State Fit subject-local PLS on $(X_{s,h}^{obs},Y_s^{obs})$ to obtain $(T_s,U_s,P_s,C_s)$
\State Fit affine alignment $(B_s,b_s)$ on the overlap between subject and atlas latent scores
\State Transport scores: $T'_s \gets T_s B_s + \mathbf{1}b_s^\top$ and $U'_s \gets U_s B_s$
\State Fit spatial interpolator $f_s$ on $(Y_s^{obs},U'_s)$ and evaluate $\hat{U}'_s(r)=f_s(y_r)$ for all parcels
\State Fit ridge bridge $(M_s,\beta_s)$ from observed aligned spatial scores to aligned genetic scores
\State Decode completed aligned genetic scores to gene space to obtain $\hat{X}_{s,h}$
\State Overwrite observed parcels: $\hat{X}_{s,h}(r,:) \gets X_{s,h}^{obs}(r,:)$ for $r\in\Omega_s$
\State \Return completed harmonized atlas $\hat{X}_{s,h}$
\end{algorithmic}
\end{algorithm}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{schematic_deterministic_pipeline.pdf}
\caption{Conceptual structure of the deterministic latent transport model. Harmonization, local multiview latent decomposition, affine score alignment, deterministic spatial completion, and decoding are estimated as sequential point-estimate stages.}
\end{figure}

\section{Unified Probabilistic Latent Alignment Model}
\subsection{Intuition}
The unified model assumes that AHBA defines a global low-dimensional transcriptomic geometry and that each subject should be interpreted as a spatially structured perturbation of that atlas geometry rather than as an entirely separate local basis. Subject-level variation is represented through three mechanisms: an orthogonal alignment of latent axes, a spatially smooth latent deviation field, and residual subject- and gene-specific calibration terms. Missing parcels are inferred by conditioning that latent deviation field on the observed parcels and decoding the resulting latent state into harmonized expression.

\subsection{Atlas latent factor model}
Let $U\in\mathbb{R}^{R\times K}$ denote the atlas latent geometry, $W\in\mathbb{R}^{G\times K}$ the gene loading matrix, and $\alpha\in\mathbb{R}^G$ the gene intercept vector. The harmonized AHBA atlas is modeled by the low-rank factorization
\[
X_A^{(h)} \approx \mathbf{1}\alpha^\top + U W^\top.
\]
Equivalently, for parcel $r$ and gene $g$,
\[
x_{Arg}^{(h)} \approx \alpha_g + u_r^\top w_g.
\]
This formulation differs fundamentally from the deterministic model: the atlas latent coordinates are estimated once at the global level and subsequently used as the reference frame for all subjects.

\subsection{Subject-specific latent state and observation model}
For subject $s$ and parcel $r$, the latent state is defined by
\[
z_{s,r} = (u_r + \delta_{s,r})Q_s,
\]
where $\delta_{s,r}\in\mathbb{R}^K$ is a subject-specific latent deviation and $Q_s\in O(K)$ is an orthogonal alignment matrix. The platform-independent latent expression for gene $g$ is then
\[
x_{srg}^{(*)} = \alpha_g + z_{s,r}^\top w_g.
\]
Observed GTEx expression in harmonized space is modeled by
\[
x_{srg}^{(G,h)} = a_{s,g}x_{srg}^{(*)} + b_{s,g} + \varepsilon_{srg},
\qquad r\in\Omega_s,
\]
where $(a_{s,g},b_{s,g})$ are subject- and gene-specific affine calibration coefficients. The present instantiation therefore separates two kinds of harmonization. A global fold-safe ComBat transform removes large-scale dataset effects prior to subject inference, while $(a_{s,g},b_{s,g})$ capture residual subject-specific calibration mismatch inside the latent model itself.

\subsection{Full joint distribution}
A convenient probabilistic specification is
\[
p\!\left(X_A^{(h)}, \{X_s^{(G,h)}\}_{s=1}^S, U, W, \alpha, \{\Delta_s,Q_s,a_s,b_s\}_{s=1}^S\right),
\]
with factorization
\begin{align*}
p(&X_A^{(h)}, \{X_s^{(G,h)}\}_{s=1}^S, U, W, \alpha, \{\Delta_s,Q_s,a_s,b_s\}_{s=1}^S) \\
&= p(X_A^{(h)}\mid U,W,\alpha)\,p(U)\,p(W)\,p(\alpha) \\
&\quad\times\prod_{s=1}^S p(Q_s)\,p(\Delta_s\mid Y)\,p(a_s,b_s)\,p(X_s^{(G,h)}\mid U,\Delta_s,Q_s,W,\alpha,a_s,b_s).
\end{align*}
One explicit realization is
\begin{align*}
X_A^{(h)}\mid U,W,\alpha &\sim \prod_{r,g}\mathcal{N}(\alpha_g + u_r^\top w_g,\sigma_{A,g}^2),\\
\delta_{s,\cdot,k}\mid Y &\sim \mathcal{GP}(0,\kappa(\cdot,\cdot)),\\
Q_s &\propto \mathbf{1}[Q_s^\top Q_s = I],\\
a_{s,g} &\sim \mathcal{N}(a_{0,g},\lambda_a^{-1}),\qquad b_{s,g} \sim \mathcal{N}(b_{0,g},\lambda_b^{-1}),\\
x_{srg}^{(G,h)}\mid\cdots &\sim \mathcal{N}(a_{s,g}x_{srg}^{(*)} + b_{s,g},\sigma_{s,g}^2).
\end{align*}
In the broader model family, the Gaussian observation model can be augmented by robust and heteroscedastic weighting, and the completed subject atlas can optionally be shrunk toward an atlas prior according to spatial uncertainty. The particular instantiation analyzed here uses Student-$t$ residual reweighting and gene-wise heteroscedastic weights, while the uncertainty-shrinkage extension is left inactive.

\subsection{Initialization}
The unified model requires explicit initialization because several latent blocks are updated iteratively. The atlas latent coordinates $U$ are initialized by a truncated singular value decomposition of centered harmonized AHBA expression. Given that initial $U$, the decoder parameters $(W,\alpha)$ are fit by ridge regression. For each subject, the orthogonal alignment matrix is initialized as $Q_s=I$, the observed latent states are initialized as the atlas latent coordinates at the observed parcels, that is $Z_{s,\Omega}=U_{\Omega}$, the calibration coefficients are initialized as $a_{s,g}=1$ and $b_{s,g}=0$, and gene precision weights are initialized uniformly. These choices correspond to a neutral starting point in which the subject initially coincides with the atlas before subject-specific structure is inferred.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{schematic_unified_pipeline.pdf}
\caption{Conceptual structure of the unified probabilistic latent alignment model. The harmonized atlas defines a global low-rank latent geometry; subject-specific inference estimates an orthogonal alignment, a latent deviation field, and residual calibration; Gaussian-process conditioning then extends the latent deviation field to unobserved parcels before decoding.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{schematic_unified_graphical_model.pdf}
\caption{Graphical-model view of the unified formulation. Atlas variables define the global transcriptomic geometry, while each subject contributes an orthogonal alignment, a latent deviation field, and calibration coefficients that together generate harmonized subject observations.}
\end{figure}

\section{Inference}
\subsection{Deterministic model: plain-language summary}
Inference in the deterministic model proceeds as a staged optimization: harmonize the data, estimate a subject-local PLS representation, align the subject scores to the atlas frame, interpolate the aligned spatial scores over unobserved parcels, bridge interpolated spatial scores back to gene-associated scores, and decode those scores into expression. Each step is deterministic once the previous step has been fixed.

\subsection{Probabilistic model: from least squares to weighted latent updates}
The latent update in the probabilistic model is easiest to understand by starting from unweighted least squares. If all genes were treated as equally reliable and all residuals were penalized equally, the observed latent state at parcel $r$ would be updated by minimizing
\[
\sum_{g=1}^G \left(x_{srg}^{(G,h)} - a_{s,g}(\alpha_g + z^\top w_g) - b_{s,g}\right)^2 + \lambda_z\|z-u_rQ_s\|_2^2.
\]
This objective says that the latent state should explain observed gene expression while remaining close to the aligned atlas latent state.

In practice, not all residuals carry the same information. A small number of very large residuals may reflect outliers or local calibration mismatch, and some genes are systematically noisier than others. The model therefore introduces weights
\[
\omega_{srg} = \omega_{srg}^{robust}\,\omega_{sg}^{hetero}.
\]
The first factor, $\omega_{srg}^{robust}$, comes from Student-$t$ reweighting and downweights observations with unusually large residuals. The second factor, $\omega_{sg}^{hetero}$, is inversely related to gene-specific residual variance and therefore gives less influence to genes whose errors remain large across observed parcels. The weighted latent update becomes
\[
\min_{z}\sum_{g=1}^G \omega_{srg}\left(x_{srg}^{(G,h)} - a_{s,g}(\alpha_g + z^\top w_g) - b_{s,g}\right)^2 + \lambda_z\|z-u_rQ_s\|_2^2.
\]
This progression from the unweighted objective to the weighted objective clarifies the role of $\omega_{srg}$: it is not an abstract free parameter, but a mechanism for tempering the influence of outlying residuals and persistently noisy genes.

\subsection{Pedagogical view of Gaussian-process latent deviation conditioning}
The role of the Gaussian process is to interpolate the \emph{subject-specific latent deviation field}, not the expression itself. After the current latent states have been estimated at the observed parcels, the observed deviation is defined by
\[
\delta^{obs}_{s,r} = z_{s,r}Q_s^\top - u_r,
\qquad r\in\Omega_s.
\]
The interpretation is straightforward: after undoing the orthogonal alignment, the observed subject latent state is decomposed into the atlas latent coordinate $u_r$ plus a residual subject-specific deviation. The Gaussian process assumes that this deviation varies smoothly over parcel space.

For a single latent dimension $k$, let $\delta^{obs}_{s,\Omega,k}$ denote the observed deviations, and define the kernel partitions
\[
K_{\Omega,\Omega},\qquad K_{r,\Omega},\qquad K_{r,r},
\]
where each entry is given by an RBF kernel
\[
\kappa(y_r,y_{r'}) = \sigma_k^2\exp\!\left(-\frac{\|y_r-y_{r'}\|_2^2}{2\ell_k^2}\right).
\]
Then the posterior mean and variance at parcel $r$ are
\begin{align*}
\hat{\delta}_{s,r,k} &= K_{r,\Omega}\left(K_{\Omega,\Omega}+\sigma_\eta^2 I\right)^{-1}\delta^{obs}_{s,\Omega,k},\\
\operatorname{Var}(\delta_{s,r,k}\mid\Omega_s) &= K_{r,r} - K_{r,\Omega}\left(K_{\Omega,\Omega}+\sigma_\eta^2 I\right)^{-1}K_{\Omega,r}.
\end{align*}
The posterior mean supplies the completed latent deviation, while the posterior variance quantifies how uncertain that interpolation is at parcel $r$. Averaging over latent dimensions gives the average latent posterior variance,
\[
\bar{\sigma}_{s,r}^2 = \frac{1}{K}\sum_{k=1}^K \operatorname{Var}(\delta_{s,r,k}\mid\Omega_s).
\]
This scalar is helpful because it summarizes, in a single number per parcel, how uncertain the latent completion is after conditioning on the observed subject parcels.

\subsection{Optional uncertainty shrinkage}
Within the broader probabilistic family, the completed subject atlas may be blended with an atlas prior using both spatial distance and average latent posterior variance. If $\tilde d_{s,r}$ denotes scaled distance to the nearest observed parcel and $\widetilde{\bar{\sigma}_{s,r}^2}$ denotes scaled average latent posterior variance, then one possible shrinkage score is
\[
\operatorname{score}_{s,r} = \alpha_d\tilde d_{s,r} + \alpha_u\widetilde{\bar{\sigma}_{s,r}^2},
\]
with logistic blending weight
\[
w_{s,r} = \sigma\!\left(\frac{\operatorname{score}_{s,r}-m_0}{\tau}\right),
\qquad
\hat x_{s,r}^{final} = (1-w_{s,r})\hat x_{s,r}^{model} + w_{s,r}\hat x_r^{prior}.
\]
This extension is part of the general probabilistic framework but is not active in the instantiated model analyzed here.

\begin{algorithm}[H]
\caption{Unified probabilistic latent alignment inference for subject $s$}
\begin{algorithmic}[1]
\Require Harmonized atlas $X_A^{(h)}$, observed harmonized subject data $X_{s,h}^{obs}$ on $\Omega_s$, parcel centroids $Y$
\State Initialize atlas latent coordinates $U$ by truncated SVD of centered $X_A^{(h)}$
\State Fit decoder parameters $(W,\alpha)$ by ridge regression on $X_A^{(h)}$
\State Initialize $Q_s \gets I$, $Z_{s,\Omega} \gets U_{\Omega}$, $a_{s,g} \gets 1$, $b_{s,g} \gets 0$, and gene weights uniformly
\For{$t = 1,\dots,T$}
    \State Update $(a_{s,g},b_{s,g})$ by regularized regression on observed parcels
    \State Update each observed latent state $z_{s,r}$ by weighted ridge regression
    \State Update $Q_s$ by orthogonal Procrustes alignment between $U_{\Omega}$ and $Z_{s,\Omega}$
    \State Update gene-wise heteroscedastic weights from residual variance
\EndFor
\State Form observed deviations $\delta_{s,r}^{obs} = z_{s,r}Q_s^\top - u_r$ for $r\in\Omega_s$
\State Condition a Gaussian process on $\delta_{s,\Omega}^{obs}$ to obtain $\hat\Delta_s$ and posterior variances over all parcels
\State Decode completed latent states into harmonized expression to obtain $\hat X_{s,h}$
\State Overwrite observed parcels: $\hat X_{s,h}(r,:) \gets X_{s,h}^{obs}(r,:)$ for $r\in\Omega_s$
\State \Return completed harmonized atlas $\hat X_{s,h}$ and parcelwise posterior variances
\end{algorithmic}
\end{algorithm}

\section{Atlas-Level Diagnostics}
Atlas-level diagnostics are useful because they separate two questions: whether the harmonization and alignment steps place AHBA and GTEx in a comparable coordinate system, and whether the completion step preserves parcel-level structure when sparse subject data are aggregated into a population atlas. Figures~\ref{fig:atlas-det-heatmap} and~\ref{fig:atlas-uni-heatmap} show the progression from raw AHBA and sparse GTEx observations to harmonized and completed parcel-by-gene atlases. In both cases, the gap between the sparse observed GTEx pattern and the completed harmonized atlas is explicitly visible.

\begin{figure}[H]
\centering
\includegraphics[width=0.94\textwidth]{atlas_deterministic_heatmap_demo.pdf}
\caption{Atlas-level heatmap diagnostic for the deterministic latent transport model. The panels show raw AHBA, sparse GTEx before completion, sparse GTEx after harmonization, the completed harmonized GTEx atlas, and the residual harmonized difference relative to AHBA.}
\label{fig:atlas-det-heatmap}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.94\textwidth]{atlas_unified_heatmap_demo.pdf}
\caption{Atlas-level heatmap diagnostic for the unified probabilistic latent alignment model. As in the deterministic model, the figure shows raw atlas structure, sparsely observed subject data, harmonized observed data, the completed harmonized atlas, and the remaining atlas--subject residual after completion.}
\label{fig:atlas-uni-heatmap}
\end{figure}

\begin{figure}[H]
\centering
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{atlas_deterministic_scatter_region_harmonized.pdf}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{atlas_deterministic_scatter_gene_harmonized.pdf}
\end{subfigure}
\medskip
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{atlas_deterministic_scatter_region_corr.pdf}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{atlas_deterministic_scatter_gene_corr.pdf}
\end{subfigure}
\caption{Atlas-level scatter diagnostics for the deterministic latent transport model. The panels compare parcel identity, gene identity, parcel-level correlation, and gene-level correlation in the harmonized atlas space. Only the most informative parcel and gene subsets are individually color-coded, with the remaining points shown in gray for context.}
\end{figure}

\begin{figure}[H]
\centering
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{atlas_unified_scatter_region_harmonized.pdf}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{atlas_unified_scatter_gene_harmonized.pdf}
\end{subfigure}
\medskip
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{atlas_unified_scatter_region_corr.pdf}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{atlas_unified_scatter_gene_corr.pdf}
\end{subfigure}
\caption{Atlas-level scatter diagnostics for the unified probabilistic latent alignment model. The same visual grammar is used as in the deterministic model, which makes it possible to compare the effect of deterministic transport and probabilistic latent completion on the atlas-scale relationship between AHBA and GTEx.}
\end{figure}

\section{Single-Subject Diagnostics}
Single-subject diagnostics illustrate how each model turns sparse observations into a completed atlas. The representative subject shown here is \texttt{__REP_SUBJECT__}, chosen because its subject-level probabilistic performance lies close to the cohort median among subjects with at least eight observed parcels. This makes the visual example representative of typical behavior rather than an extreme case.

\begin{figure}[H]
\centering
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{subject_deterministic_heatmap_demo.pdf}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{subject_unified_heatmap_demo.pdf}
\end{subfigure}
\caption{Single-subject heatmap diagnostics for the representative subject \texttt{__REP_SUBJECT__}. The deterministic and probabilistic models are shown in the same visual format so that the transition from sparse observations to completed harmonized atlas can be compared directly.}
\end{figure}

\begin{figure}[H]
\centering
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{subject_deterministic_scatter_harmonized.pdf}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.49\textwidth}
    \centering
    \includegraphics[width=\textwidth]{subject_unified_scatter_harmonized.pdf}
\end{subfigure}
\caption{Single-subject scatter diagnostics in harmonized space for the representative subject. Parcel identity, gene identity, parcel-level correlation, and gene-level correlation are shown for the completed subject atlas under the two models.}
\end{figure}

\section{Validation Context}
Validation is based on leave-one-region-out cross-validation at the subject level. For each subject and each observed parcel, the parcel is held out completely, all fitted quantities are recomputed without it, and the held-out gene vector is predicted from the remaining parcels. This protocol is strict: the held-out parcel is excluded from harmonization, latent fitting, spatial completion, and decoding. The resulting diagnostics therefore measure genuine out-of-sample regional generalization rather than reconstruction of inputs seen during fitting.

The leave-one-region-out diagnostics reported below are computed over the full gene set so that the manuscript-level evaluation is consistent with the main transcriptome-wide completion setting. These plots are diagnostic rather than exhaustive: the aim is to expose how error and correlation vary over subjects, parcels, and genes, not merely to report a single cohort-level score.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{loro_subject_metrics_allgene.pdf}
\caption{Subject-level leave-one-region-out diagnostics over the full gene set. Subjects are ordered by probabilistic-model Pearson correlation, and the panels show subject-wise Pearson correlation, subject-wise RMSE, and cohort-level summary bars for the deterministic model, the probabilistic model, and the naive AHBA-fill baseline.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth]{loro_parcel_metrics_allgene.pdf}
\caption{Parcel-level leave-one-region-out diagnostics over the full gene set. The heatmaps summarize mean held-out correlation and RMSE by parcel across the canonical parcel set; rows with no GTEx holdout events are left blank. The bar plot shows parcel-specific improvement relative to the naive AHBA-fill baseline for parcels with available leave-one-region-out evaluations.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.92\textwidth]{loro_method_summary_bars_allgene.pdf}
\caption{Cohort-level summary of leave-one-region-out diagnostics over the full gene set. Mean Pearson correlation, mean RMSE, and mean improvement relative to the naive AHBA-fill baseline are shown for both atlas-completion models and the naive baseline.}
\end{figure}

\section{Discussion}
The deterministic latent transport model and the unified probabilistic latent alignment model share a common scientific aim: to infer a subject-complete parcel-by-gene atlas from sparse GTEx regional measurements by borrowing structure from AHBA. Their main difference lies in what is treated as primary. The deterministic model treats each subject as the starting point, estimates a subject-local latent basis, and then transports that basis into the atlas frame. The probabilistic model treats the atlas as the starting point, defines a global latent geometry there, and then expresses each subject as an aligned perturbation of that geometry.

This distinction has practical consequences. The deterministic model is modular and interpretable at each step, which makes it attractive for diagnosing where alignment or extrapolation may fail. The probabilistic model is more coherent because calibration, alignment, latent completion, and uncertainty all live in a single atlas-referenced latent system. That coherence is what makes Gaussian-process deviation conditioning natural in the probabilistic formulation: the Gaussian process is not interpolating gene expression directly, but a low-dimensional deviation field whose spatial smoothness is much easier to motivate biologically and statistically.

The two approaches should therefore not be understood simply as older and newer code paths. They represent two different answers to the same problem. The deterministic formulation provides a transparent latent transport view built from subject-local components, whereas the probabilistic formulation provides a global generative view in which atlas structure, subject alignment, subject deviation, and calibration are estimated jointly in a common latent space.

\appendix
\section{Additional Diagnostic Figures}
\begin{figure}[H]
\centering
\includegraphics[width=0.88\textwidth]{loro_gene_metrics_allgene.pdf}
\caption{Gene-level leave-one-region-out diagnostics over the full gene set. The heatmaps summarize gene-wise correlation and RMSE across held-out observations, and the accompanying bar plot highlights genes for which the atlas-completion models improve most or least relative to the naive AHBA-fill baseline.}
\end{figure}

\begin{figure}[H]
\centering
\begin{subfigure}[t]{0.46\textwidth}
    \centering
    \includegraphics[width=\textwidth]{legend_top10_parcels.pdf}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.42\textwidth}
    \centering
    \includegraphics[width=\textwidth]{legend_top_genes.pdf}
\end{subfigure}
\caption{Legend panels used in the parcel- and gene-colored scatter diagnostics. The parcel subset contains the ten parcels with highest within-parcel variance in the harmonized AHBA reference, and the gene subset contains the twelve most variable genes in the same harmonized reference.}
\end{figure}

__BIBLIOGRAPHY__

\end{document}
'''
    tex = tex.replace("__REP_SUBJECT__", rep_subject)
    tex = tex.replace("__BIBLIOGRAPHY__", _thebibliography())
    return tex


def _run_tectonic(out_tex: Path, out_pdf: Path) -> None:
    cwd = out_tex.parent
    subprocess.check_call(["tectonic", str(out_tex.name)], cwd=str(cwd))
    built = cwd / (out_tex.stem + ".pdf")
    if built != out_pdf:
        shutil.copy2(built, out_pdf)


def main() -> None:
    args = parse_args()
    asset_root = Path(args.asset_root).resolve()
    out_tex = Path(args.out_tex).resolve()
    out_pdf = Path(args.out_pdf).resolve()
    out_bib = Path(args.out_bib).resolve()
    det_root = Path(args.deterministic_root).resolve()
    uni_root = Path(args.unified_phase2_root).resolve()

    fig_dir = asset_root / "figures"
    tab_dir = asset_root / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    required_figs = [
        "schematic_deterministic_pipeline.pdf",
        "schematic_unified_pipeline.pdf",
        "schematic_unified_graphical_model.pdf",
        "atlas_deterministic_heatmap_demo.pdf",
        "atlas_unified_heatmap_demo.pdf",
        "atlas_deterministic_scatter_region_harmonized.pdf",
        "atlas_deterministic_scatter_gene_harmonized.pdf",
        "atlas_deterministic_scatter_region_corr.pdf",
        "atlas_deterministic_scatter_gene_corr.pdf",
        "atlas_unified_scatter_region_harmonized.pdf",
        "atlas_unified_scatter_gene_harmonized.pdf",
        "atlas_unified_scatter_region_corr.pdf",
        "atlas_unified_scatter_gene_corr.pdf",
        "subject_deterministic_heatmap_demo.pdf",
        "subject_unified_heatmap_demo.pdf",
        "subject_deterministic_scatter_harmonized.pdf",
        "subject_unified_scatter_harmonized.pdf",
        "legend_top10_parcels.pdf",
        "legend_top_genes.pdf",
        "loro_subject_metrics_allgene.pdf",
        "loro_parcel_metrics_allgene.pdf",
        "loro_gene_metrics_allgene.pdf",
        "loro_method_summary_bars_allgene.pdf",
    ]
    missing = [f for f in required_figs if not (fig_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required figures: {missing}")

    rep_manifest = pd.read_csv(tab_dir / "representative_subject_manifest.csv")
    rep_subject = str(rep_manifest.iloc[0]["subject"])

    det_tbl = pd.read_csv(det_root / "tables" / "model_grid_summary_allgenes.csv")
    det_row = det_tbl[det_tbl["model_combo"] == "combat__affine_gl3__constrained_anchor__rbf"].iloc[0].to_dict()
    uni_tbl = pd.read_csv(uni_root / "tables" / "allgenes_model_summary.csv")
    uni_row = uni_tbl.iloc[0].to_dict()

    _write_bib(out_bib)
    _write_bib_dedup_manifest(tab_dir / "bibliography_dedup_manifest.csv")

    tex = _build_tex(rep_subject, det_row, uni_row)
    out_tex.write_text(tex, encoding="utf-8")
    _run_tectonic(out_tex, out_pdf)

    manifest = {
        "asset_root": str(asset_root),
        "out_tex": str(out_tex),
        "out_pdf": str(out_pdf),
        "out_bib": str(out_bib),
        "representative_subject": rep_subject,
        "figure_files": required_figs,
        "deterministic_summary": {
            "mean_pearson": float(det_row.get("mean_pearson", float("nan"))),
            "mean_rmse": float(det_row.get("mean_rmse", float("nan"))),
        },
        "unified_summary": {
            "mean_pearson": float(uni_row.get("mean_pearson", float("nan"))),
            "mean_rmse": float(uni_row.get("mean_rmse", float("nan"))),
        },
    }
    (asset_root / "rewrite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_tex}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
