"""Lithuanian interim report formatted per VU GMC thesis guidelines (2025 edition).

Output: private/results/ataskaita_LT.pdf (figures as vector PDF in private/results/M9_lt_fig/).
Run:    uv run --frozen --with reportlab --with pyphen --with pdfrw python scripts/m9_report_lt.py

GMC rules applied: A4; margins top/bottom 2 cm, left 2.5 cm, right 1.5 cm; body Serif 12 pt,
1.5 line spacing, justified, 1 cm first-line indent; headings Sans Serif bold 17/14/12 pt with
6 pt spacing; captions single-spaced, figures captioned below and tables above, no final
period; figure text Sans Serif 9-15 pt at final size; decimal comma; Lithuanian quotes;
non-breaking space between number and unit; page numbers from page 2 in the bottom margin;
author-year (APA) citations.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RUNS = Path(".local/runs")
M9 = RUNS / "20261001T181436.048902Z-m9-5a678459"
FEAT = RUNS / "20261001T175036.684944Z-features-7f6e6304"
IMGS = RUNS / "20261001T124005.643225Z-call-images-c60840b7"
CTRL = RUNS / "20261001T122013.275165Z-controls-c4cf4161"
ATTR = RUNS / "20261001T181732.397771Z-m9-attribution-bcb0fb9a"
FIG = Path("private/results/M9_lt_fig")
PDF = Path("private/results/ataskaita_LT.pdf")

AUTHOR = "Saulius Beconis"
SUPERVISOR = "Osvaldas Rukšėnas"
TEXT_W_CM = 21.0 - 2.5 - 1.5  # 17 cm text block
IN = 1 / 2.54
LIT = {"E2L1": "#0072B2", "E2L2": "#D55E00", "E2L3": "#009E73"}  # validated (CVD ΔE ≥ 11)
GREY, INK, MUTED = "#8c8c8c", "#222222", "#666666"
ACCENT = "#7A1F45"  # VU template accent (title, headings)
NB = " "  # non-breaking space

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#444444", "axes.labelcolor": INK,
    "xtick.color": "#444444", "ytick.color": "#444444", "pdf.fonttype": 42, "lines.linewidth": 1.6,
    "axes.unicode_minus": True, "savefig.bbox": None, "pdf.compression": 0,
})  # fmt: skip


# ------------------------------------------------------------------ number formatting


def num(x: float, d: int = 2, sign: bool = False) -> str:
    """Lithuanian decimal comma and true minus sign."""
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−").replace(".", ",")


def ci(lo: float, hi: float, d: int = 2) -> str:
    return f"[{num(lo, d)}; {num(hi, d)}]"


def comma_axis(ax: Any, which: str = "both") -> None:
    def f(x: float, _: Any) -> str:
        s = f"{x:g}"
        return s.replace("-", "−").replace(".", ",")

    if which in ("both", "y"):
        ax.yaxis.set_major_formatter(FuncFormatter(f))
    if which in ("both", "x"):
        ax.xaxis.set_major_formatter(FuncFormatter(f))


def panel(ax: Any, letter: str) -> None:
    from matplotlib.transforms import ScaledTranslation

    off = ScaledTranslation(-30 / 72, 6 / 72, ax.figure.dpi_scale_trans)
    ax.text(
        0,
        1,
        letter,
        transform=ax.transAxes + off,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save(fig: Any, name: str) -> Path:
    path = FIG / f"{name}.pdf"
    fig.savefig(path)
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ data


def per_pup_mae() -> pd.DataFrame:
    """Leave-one-litter-out per-pup MAE: classical vs classical + six shape features (decision 0019)."""
    from rat_usv.evaluation.controls import feature_sets
    from rat_usv.evaluation.m4 import fit_predict
    from rat_usv.evaluation.m9 import load_days

    day, baseline = load_days(CTRL, FEAT)
    classical = feature_sets(day, [])["classical"]
    pred = {k: np.full(len(day), np.nan) for k in ("c", "cs")}
    for lit in sorted(day["litter_id"].unique()):
        test = (day["litter_id"] == lit).to_numpy()
        pred["c"][test] = fit_predict(day[~test], day[test], classical)
        pred["cs"][test] = fit_predict(day[~test], day[test], baseline)
    out = pd.DataFrame({"animal_id": day["animal_id"], "litter_id": day["litter_id"]})
    for k, v in pred.items():
        out[k] = np.abs(v - day["age"].to_numpy())
    return out.groupby(["animal_id", "litter_id"])[["c", "cs"]].mean().reset_index()


# ------------------------------------------------------------------ figures


def fig_design() -> Path:
    fig, ax = plt.subplots(figsize=(TEXT_W_CM * IN, 7.4 * IN))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    rows = [("E2L1", ("E2L2", "E2L3")), ("E2L2", ("E2L1", "E2L3")), ("E2L3", ("E2L1", "E2L2"))]
    heads = [
        (2.6, "Atradimas\n(dvi vados)"),
        (8.5, "Užrakinimas"),
        (14.35, "Patvirtinimas\n(nematyta vada)"),
    ]
    for x, t in heads:
        ax.text(x, 6.85, t, ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
    for i, (held, disc) in enumerate(rows):
        y = 5.35 - i * 1.75

        def box(x0: float, w: float, text: str, fc: str, ec: str, y: float = y) -> None:
            ax.add_patch(FancyBboxPatch((x0, y - 0.6), w, 1.2, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        fc=fc, ec=ec, lw=0.9))  # fmt: skip
            ax.text(x0 + w / 2, y, text, ha="center", va="center", fontsize=9.5, color=INK)

        box(
            0.1,
            5.0,
            f"{disc[0]} + {disc[1]}:\npožymių atranka ir\nkrypties prognozė",
            "#eef2f7",
            "#9fb1c9",
        )
        box(
            6.15,
            4.7,
            "prognozės failas\nužrakinamas SHA-256\nkontroline suma",
            "#f6f0f3",
            "#c9a3b5",
        )
        box(11.95, 4.8, f"{held}:\ntikrinama tik\nužrakinta prognozė", "#edf6f1", "#9cc8b0")
        for x0, x1 in ((5.15, 6.1), (10.9, 11.9)):
            ax.annotate(
                "",
                xy=(x1, y),
                xytext=(x0, y),
                arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 0.9},
            )
    ax.text(8.5, 0.1, "Radinys patvirtintas, jei pasitvirtina ≥ 2 iš 3 nematytų vadų su prognozuota kryptimi (FDR q = 0,05)",
            ha="center", va="bottom", fontsize=9.5, color=MUTED)  # fmt: skip
    return save(fig, "1_schema")


def fig_gain(mae: pd.DataFrame, vae_mean: float, both_mean: float) -> Path:
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(TEXT_W_CM * IN, 6.4 * IN), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    labels = ["Klasikiniai", "+ VAE", "+ 6 formos\npožymiai", "+ forma\n+ VAE"]
    vals = [mae["c"].mean(), vae_mean, mae["cs"].mean(), both_mean]
    colors = [GREY, "#9b8ec4", "#0072B2", "#3d4f8f"]
    x = np.arange(len(vals))
    a.bar(x, vals, color=colors, width=0.62, edgecolor="white", linewidth=1.5)
    for xi, v in zip(x, vals, strict=True):
        a.text(xi, v + 0.06, num(v), ha="center", va="bottom", fontsize=9.5, color=INK)
    a.set_xticks(x, labels)
    a.set_ylabel("Amžiaus paklaida (d)")
    a.set_ylim(0, 3.9)
    comma_axis(a, "y")
    a.tick_params(axis="x", length=0)
    panel(a, "A")
    for _, r in mae.iterrows():
        b.plot([0, 1], [r["c"], r["cs"]], color=LIT[r["litter_id"]], alpha=0.55, lw=1.0)
    for lit, g in mae.groupby("litter_id"):
        b.scatter(
            [0] * len(g), g["c"], s=14, color=LIT[lit], zorder=3, edgecolor="white", linewidth=0.5
        )
        b.scatter(
            [1] * len(g),
            g["cs"],
            s=14,
            color=LIT[lit],
            zorder=3,
            edgecolor="white",
            linewidth=0.5,
            label=lit,
        )
    b.set_xticks([0, 1], ["Klasikiniai", "+ 6 formos\npožymiai"])
    b.set_xlim(-0.3, 1.3)
    b.set_ylabel("Jauniklio paklaida (d)")
    comma_axis(b, "y")
    b.tick_params(axis="x", length=0)
    b.legend(frameon=False, loc="upper right", handletextpad=0.2)
    panel(b, "B")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.2, top=0.88, wspace=0.34)
    return save(fig, "2_paklaida")


def fig_modes() -> Path:
    s = json.loads((M9 / "m9_confirmation.json").read_text())
    var = s["fpca"]["E2L2"]["f0"]["var"]
    titles = ["Kryptis", "Arka ↔ U forma", "Banguotumas"]
    with np.load(M9 / "m9_bases_holdout_E2L2.npz") as z:
        mean, modes, sd = z["f0_mean"], z["f0_modes"], z["f0_sd"]
    tau = np.linspace(0, 100, mean.size)
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_W_CM * IN, 10.6 * IN), sharex=True)
    for j in range(3):
        ax = axes[0, j]
        ax.axhline(0, color="#dddddd", lw=0.7)
        ax.plot(tau, mean + 2 * sd[j] * modes[j], color="#B2182B", label="+2 SD")
        ax.plot(tau, mean - 2 * sd[j] * modes[j], color="#2166AC", label="−2 SD")
        ax.set_title(f"{titles[j]} ({num(100 * var[j], 0)}{NB}%)", fontsize=10.5)
        comma_axis(ax)
        b = axes[1, j]
        b.axhline(0, color="#dddddd", lw=0.7)
        for rot, c in LIT.items():
            with np.load(M9 / f"m9_bases_holdout_{rot}.npz") as z:
                b.plot(tau, z["f0_modes"][j], color=c, lw=1.4, label=f"be {rot}")
        b.set_xlabel("Signalo trukmė (%)")
        comma_axis(b)
    axes[0, 0].set_ylabel("f0 nuokrypis (kHz)")
    axes[1, 0].set_ylabel("Komponentės svoris")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        h,
        lab,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.55, 1.0),
        handlelength=1.6,
    )
    h, lab = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        h,
        lab,
        loc="center right",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.995, 0.505),
        handlelength=1.6,
    )
    panel(axes[0, 0], "A")
    panel(axes[1, 0], "B")
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.85, wspace=0.3, hspace=0.5)
    return save(fig, "3_formos")


def trend_fig(cols: list[tuple[str, str]], name: str, ncols: int) -> Path:
    day = pd.read_parquet(M9 / "m9_day_features_holdout_E2L2.parquet")
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(TEXT_W_CM * IN, (6.4 if nrows == 1 else 5.3) * nrows * IN),
        squeeze=False,
    )
    for k, (ax, (col, lab)) in enumerate(zip(axes.ravel(), cols, strict=False)):
        d = day.dropna(subset=[col])
        for _, g in d.groupby("animal_id"):
            g = g.sort_values("age")
            ax.plot(g["age"], g[col], color=LIT[g["litter_id"].iloc[0]], alpha=0.18, lw=0.6)
        for lit, g in d.groupby("litter_id"):
            m = g.groupby("age")[col].median().rolling(3, center=True, min_periods=1).mean()
            ax.plot(m.index, m.to_numpy(), color=LIT[lit], lw=2.0, label=lit)
        ax.set_xticks([1, 7, 14, 21])
        ax.set_xlabel("Amžius (P)")
        ax.set_ylabel(lab)
        lo, hi = np.nanpercentile(d[col], [1, 99])
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        comma_axis(ax)
        panel(ax, "ABCDEF"[k])
    for ax in axes.ravel()[len(cols) :]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, fontsize=9, handlelength=1.2, loc="best")
    fig.subplots_adjust(left=0.1 if ncols < 3 else 0.085, right=0.99, bottom=0.21 if nrows == 1 else 0.1,
                        top=0.88 if nrows == 1 else 0.93, wspace=0.42 if ncols > 2 else 0.3, hspace=0.55)  # fmt: skip
    return save(fig, name)


def fig_examples() -> Path:
    from rat_usv.evaluation.m9 import load_calls, project

    calls, curves = load_calls(FEAT)
    idx = pd.read_parquet(IMGS / "calls_index.parquet").reset_index()
    imgs = np.load(IMGS / "calls_images.npy", mmap_mode="r")
    with np.load(M9 / "m9_bases_holdout_E2L2.npz") as z:
        basis = {"mean": z["f0_mean"], "modes": z["f0_modes"]}
    calls["score"] = project(curves["f0"][calls["row"].to_numpy()].astype(float), basis)[:, 0]
    pos = idx.merge(calls[["wav_path", "call_index", "score"]], on=["wav_path", "call_index"])
    pos = pos[pos["litter_id"] == "E2L2"]
    q = pos["score"].quantile([0.05, 0.15, 0.85, 0.95]).to_numpy()
    rows = [("A", "Krintantys", pos[pos["score"].between(q[0], q[1])].sample(6, random_state=0)),
            ("B", "Kylantys", pos[pos["score"].between(q[2], q[3])].sample(6, random_state=0))]  # fmt: skip
    fig, axes = plt.subplots(2, 6, figsize=(TEXT_W_CM * IN, 7.0 * IN))
    for r, (letter, lab, sel) in enumerate(rows):
        sel = sel.sort_values("postnatal_age_days")
        for c, (_, row) in enumerate(sel.iterrows()):
            ax = axes[r, c]
            ax.imshow(
                imgs[int(row["index"])],
                cmap="magma",
                aspect="auto",
                extent=(0, 1, 18, 100),
                interpolation="bilinear",
            )
            ax.set_xticks([])
            ax.set_yticks([20, 60, 100] if c == 0 else [])
            ax.set_title(f"P{int(row['postnatal_age_days'])}", fontsize=9.5)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(f"{lab}\nf (kHz)")
        panel(axes[r, 0], letter)
    fig.subplots_adjust(left=0.1, right=0.995, bottom=0.03, top=0.9, wspace=0.06, hspace=0.32)
    return save(fig, "5_pavyzdziai")


def fig_occlusion() -> Path:
    t = pd.read_parquet(ATTR / "m9_occlusion.parquet")
    t = t.assign(
        abs_delta=t["delta"].abs(),
        band=pd.cut(t["age"], [0, 7, 14, 21], labels=["P1–7", "P8–14", "P15–21"]),
    )
    order = [
        "fund_start",
        "fund_middle",
        "fund_end",
        "harm_start",
        "harm_middle",
        "harm_end",
        "outside",
    ]
    names = [
        "f0: pradžia",
        "f0: vidurys",
        "f0: pabaiga",
        "2f0: pradžia",
        "2f0: vidurys",
        "2f0: pabaiga",
        "Kita sritis",
    ]
    g = t.pivot_table(
        index="region", columns="band", values="abs_delta", aggfunc="mean", observed=True
    ).reindex(order)
    fig, ax = plt.subplots(figsize=(10.5 * IN, 6.3 * IN))
    im = ax.imshow(g.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
    for (i, j), v in np.ndenumerate(g.to_numpy()):
        ax.text(
            j, i, num(v), ha="center", va="center", fontsize=9, color="white" if v > 0.45 else INK
        )
    ax.set_xticks(range(3), list(g.columns))
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("|Δ| liekamojo amžiaus įverčio (d)")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    fig.subplots_adjust(left=0.27, right=0.88, bottom=0.1, top=0.97)
    return save(fig, "7_uzdengimas")


# ------------------------------------------------------------------ document


def build(figs: dict[str, Path], mae: pd.DataFrame, vae_mean: float, both_mean: float) -> None:
    from pdfrw import PdfReader
    from pdfrw.buildxobj import pagexobj
    from pdfrw.toreportlab import makerl
    from reportlab import rl_config
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        BaseDocTemplate,
        CondPageBreak,
        Flowable,
        Frame,
        KeepTogether,
        NextPageTemplate,
        PageBreak,  # fmt: skip
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.tableofcontents import TableOfContents

    fd = Path("/usr/share/fonts/liberation")
    for fam, base in (("Serif", "LiberationSerif"), ("Sans", "LiberationSans")):
        for sty, suf in (("", "Regular"), ("-B", "Bold"), ("-I", "Italic"), ("-BI", "BoldItalic")):
            pdfmetrics.registerFont(TTFont(f"{fam}{sty}", str(fd / f"{base}-{suf}.ttf")))
        pdfmetrics.registerFontFamily(
            fam, normal=fam, bold=f"{fam}-B", italic=f"{fam}-I", boldItalic=f"{fam}-BI"
        )
    rl_config.canvas_basefontname = "Serif"  # no unembedded Helvetica
    accent = colors.HexColor(ACCENT)
    hy = {"hyphenationLang": "lt", "embeddedHyphenation": 1, "uriWasteReduce": 0.3}
    body = ParagraphStyle("body", fontName="Serif", fontSize=12, leading=18, alignment=TA_JUSTIFY,
                          firstLineIndent=1 * cm, spaceAfter=0, **hy)  # fmt: skip
    body_ni = ParagraphStyle("body_ni", parent=body, firstLineIndent=0)
    li = ParagraphStyle(
        "li",
        parent=body,
        firstLineIndent=0,
        leftIndent=1.0 * cm,
        bulletIndent=0.45 * cm,
        bulletFontName="Serif",
        bulletFontSize=12,
    )
    cap = ParagraphStyle("cap", fontName="Serif", fontSize=11, leading=13.2, alignment=TA_JUSTIFY,
                         leftIndent=0, spaceBefore=4, spaceAfter=12, **hy)  # fmt: skip
    tcap = ParagraphStyle("tcap", parent=cap, spaceBefore=10, spaceAfter=6)
    cell = ParagraphStyle("cell", fontName="Serif", fontSize=10.5, leading=12.6, alignment=TA_LEFT)
    cellc = ParagraphStyle("cellc", parent=cell, alignment=TA_CENTER)
    h1 = ParagraphStyle(
        "h1",
        fontName="Sans-B",
        fontSize=17,
        leading=20.4,
        spaceBefore=6,
        spaceAfter=12,
        textColor=accent,
    )
    h2 = ParagraphStyle(
        "h2",
        fontName="Sans-B",
        fontSize=14,
        leading=16.8,
        spaceBefore=14,
        spaceAfter=6,
        textColor=accent,
    )
    ref = ParagraphStyle(
        "ref", parent=body, firstLineIndent=-1 * cm, leftIndent=1 * cm, alignment=TA_LEFT
    )

    class Doc(BaseDocTemplate):
        def afterFlowable(self, f: Any) -> None:
            if isinstance(f, Paragraph) and getattr(f, "_toc", None) is not None:
                level, text = f._toc
                key = f"h{id(f)}"
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (level, text, self.page, key))

    def on_page(canv: Any, doc: Any) -> None:
        canv.saveState()
        canv.setFont("Sans", 10)
        canv.setFillColor(accent)
        canv.drawRightString(A4[0] - 1.5 * cm, 1.1 * cm, str(doc.page))
        canv.restoreState()

    doc = Doc(str(PDF), pagesize=A4, leftMargin=2.5 * cm, rightMargin=1.5 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
              title="Žiurkių jauniklių ultragarsinių signalų forma ir postnatalinė raida", author=AUTHOR,
              subject="Tarpinė tyrimo ataskaita", lang="lt-LT")  # fmt: skip
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)  # fmt: skip
    doc.addPageTemplates(
        [PageTemplate("title", [frame]), PageTemplate("main", [frame], onPage=on_page)]
    )

    class PdfFig(Flowable):
        def __init__(self, path: Path, width_cm: float = TEXT_W_CM) -> None:
            super().__init__()
            self.x = pagexobj(PdfReader(str(path)).pages[0])
            w, h = (
                float(self.x.BBox[2]) - float(self.x.BBox[0]),
                float(self.x.BBox[3]) - float(self.x.BBox[1]),
            )
            self.s = width_cm * cm / w
            self.width, self.height = width_cm * cm, h * self.s
            self.hAlign = "CENTER"

        def wrap(self, aw: float, ah: float) -> tuple[float, float]:
            return self.width, self.height

        def draw(self) -> None:
            rl = makerl(self.canv, self.x)
            self.canv.saveState()
            self.canv.scale(self.s, self.s)
            self.canv.doForm(rl)
            self.canv.restoreState()

    def H(text: str, level: int, num_: str = "", new_page: bool = False) -> list[Any]:
        full = f"{num_} {text}".strip()
        p = Paragraph(full, h1 if level == 0 else h2)
        p._toc = (level, full)  # type: ignore[attr-defined]
        return ([PageBreak()] if new_page else [CondPageBreak(4 * cm)]) + [p]

    def P(t: str, indent: bool = True) -> Paragraph:
        return Paragraph(t, body if indent else body_ni)

    def bullets(items: list[str], numbered: bool = False) -> list[Paragraph]:
        return [
            Paragraph(t, li, bulletText=(f"{i}." if numbered else "•"))
            for i, t in enumerate(items, 1)
        ]

    def figure(key: str, n: int, title: str, text: str, width: float = TEXT_W_CM) -> KeepTogether:
        c = Paragraph(
            f"<b>{n} pav.</b> {title}. {text}" if text else f"<b>{n} pav.</b> {title}", cap
        )
        return KeepTogether([Spacer(1, 6), PdfFig(figs[key], width), c])

    def table(
        n: int, title: str, rows: list[list[Any]], widths: list[float], header_rows: int = 1
    ) -> KeepTogether:
        data = [[Paragraph(str(c), cellc if j else cell) if not isinstance(c, Paragraph) else c for j, c in enumerate(r)]
                for r in rows]  # fmt: skip
        t = Table(data, colWidths=[w * cm for w in widths], repeatRows=header_rows, hAlign="CENTER")
        t.setStyle(TableStyle([
            ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black), ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.5, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))  # fmt: skip
        return KeepTogether([Paragraph(f"<b>{n} lentelė.</b> {title}", tcap), t, Spacer(1, 12)])

    s = json.loads((M9 / "m9_confirmation.json").read_text())
    conf = s["confirmation"]["features"]
    gains = s["confirmation"]["gains"]
    var = s["fpca"]["E2L2"]["f0"]["var"]
    bio = json.loads((M9 / "m9_biology.json").read_text())
    ident = pd.DataFrame(bio["identity_by_band"]["rows"])
    attr = json.loads((ATTR / "m9_attribution_summary.json").read_text())
    n_conf = sum(v["confirmed"] for v in conf.values())
    n_all3 = sum(v["rotations_confirmed"] == 3 for v in conf.values())
    gain_c = mae["c"].mean() - mae["cs"].mean()
    improved = int((mae["c"] > mae["cs"]).sum())

    # ---------------------------------------------------------- title page (Sans Serif; Appendix A layout)
    tp = lambda size, bold=False, color=colors.black, space=0, align=TA_CENTER: ParagraphStyle(  # noqa: E731
        f"tp{size}{bold}", fontName="Sans-B" if bold else "Sans", fontSize=size, leading=size * 1.25,
        alignment=align, textColor=color, spaceAfter=space)  # fmt: skip
    story: list[Any] = [
        Spacer(1, 3.2 * cm),
        Paragraph("VILNIAUS UNIVERSITETAS", tp(14)),
        Paragraph("GYVYBĖS MOKSLŲ CENTRAS", tp(14)),
        Spacer(1, 2.6 * cm),
        Paragraph(AUTHOR, tp(14, True)),
        Spacer(1, 1.1 * cm),
        Paragraph(
            "Žiurkių jauniklių ultragarsinių signalų forma kaip postnatalinės raidos žymuo",
            tp(17, True, accent),
        ),
        Spacer(1, 0.9 * cm),
        Paragraph("Tarpinė bakalauro baigiamojo darbo tyrimo ataskaita", tp(12, True)),
        Spacer(1, 3.2 * cm),
    ]
    right = ParagraphStyle("r", fontName="Sans", fontSize=12, leading=16, leftIndent=8.8 * cm)
    right_i = ParagraphStyle("ri", parent=right, fontName="Sans-I", spaceBefore=12)
    story += [Paragraph("Darbo vadovas", right_i), Paragraph(SUPERVISOR, right),
              Paragraph("Darbas atliktas", right_i), Paragraph("GMC Biomokslų instituto", right),
              Paragraph("Neurobiologijos ir biofizikos katedroje", right),
              Spacer(1, 4.0 * cm), Paragraph("Vilnius 2026", tp(12)), NextPageTemplate("main"), PageBreak()]  # fmt: skip

    # ---------------------------------------------------------- contents
    toc = TableOfContents(dotsMinLevel=0)
    toc.levelStyles = [
        ParagraphStyle("t0", fontName="Serif-B", fontSize=12, leading=15, spaceBefore=5),
        ParagraphStyle("t1", fontName="Serif", fontSize=12, leading=15, leftIndent=0.8 * cm),
    ]
    p = Paragraph("Turinys", h1)
    story += [p, toc]

    # ---------------------------------------------------------- introduction
    story += H("Įvadas", 0, new_page=True)
    story += [
        P("Žiurkių jauniklių ultragarsinės vokalizacijos (UGV) dažnai naudojamos kaip raidos ir elgsenos rodiklis. "
          "Įprastoje analizėje signalai sutraukiami į kelis suvestinius dydžius: signalų skaičių ir dažnį, trukmę, "
          "vidutinį dažnį, dažnių juostos plotį ar nuolydį. Tačiau kiekvienas 30–60" + NB + "ms trukmės signalas turi ir "
          "sudėtingesnę struktūrą: per jį kinta pagrindinis dažnis (f0), garsumas, harmonikų stiprumas ir tonalumas. "
          "Šiame darbe tikriname hipotezę, kad ši daugiamatė struktūra atspindi postnatalinę raidą išsamiau nei "
          "klasikiniai rodikliai."),
        P("Amžių naudojame kaip patikrinimo skalę, o ne kaip galutinį tikslą: jei požymiai neša raidos informaciją, "
          "iš jų turi būti galima nustatyti jauniklių, kurių analizė nematė, amžių. Pagrindinis dėmesys skiriamas "
          "tam, <i>kas</i> signaluose keičiasi ir ar tie pokyčiai atsikartoja nepriklausomose vadose."),
        P("<b>Darbo tikslas</b> – įvertinti, ar daugiamatė žiurkių jauniklių UGV struktūra atspindi postnatalinę raidą "
          "(P1–P21) išsamiau nei klasikiniai suvestiniai rodikliai.", indent=False),
        Spacer(1, 4),
        P("<b>Darbo uždaviniai:</b>", indent=False),
        *bullets([
            "Sukurti atkuriamą signalų aptikimo, kontūro išskyrimo ir klasikinių požymių grandinę.",
            "Įvertinti, ar išmokta reprezentacija ir iš anksto aprašyti formos požymiai prideda amžiaus "
            "informacijos, kurios nėra klasikiniuose požymiuose.",
            "Duomenimis grįstai išskirti signalų formas bei kitas savybes ir patikrinti jų raidos kryptis "
            "nematytose vadose.",
            "Įvertinti signalų įvairovės, lyties ir individualių skirtumų kitimą su amžiumi.",
        ], numbered=True),
    ]  # fmt: skip

    # ---------------------------------------------------------- methods
    story += H("Medžiagos ir metodai", 0, "1.", new_page=True)
    story += H("Duomenys", 1, "1.1.")
    story += [
        P(
            "Analizuoti 28 žiurkių jauniklių (3 vados: E2L1 – 10, E2L2 – 4, E2L3 – 14 jauniklių; 17 patelių ir "
            "11 patinų) izoliacijos metu įrašyti signalai nuo P1 iki P21. Iš viso 574 įrašai (apie 5"
            + NB
            + "min kiekvienas, "
            "diskretizavimo dažnis 214"
            + NB
            + "285"
            + NB
            + "Hz). Amžius skaičiuotas dienomis nuo užrašytos gimimo datos (P0). "
            "Analizės vienetas yra jauniklis: duomenys niekada nebuvo dalijami atsitiktinai pagal signalus."
        ),
    ]
    story += H("Signalų aptikimas ir požymiai", 1, "1.2.")
    story += [
        P(
            "Signalai aptikti skaidriu energijos detektoriumi 18–100"
            + NB
            + "kHz juostoje, normalizavus kiekvieno dažnio "
            "kanalo foną. Rankiniu būdu patikrintame imties įraše detektoriaus tikslumas buvo apie 0,93 (signalams su "
            "SNR ≥ 14"
            + NB
            + "dB – apie 0,99), jautrumas – apie 0,99. Aptikta 42"
            + NB
            + "391 signalų, iš jų 38"
            + NB
            + "170 patikimų "
            "(SNR ≥ 14" + NB + "dB); jie sudarė 494 jauniklio dienas su ne mažiau kaip 5 signalais."
        ),
        P(
            "Kiekvienam signalui išskirtas pagrindinio dažnio kontūras (su oktavos klaidų korekcija) ir apskaičiuoti "
            "klasikiniai požymiai: trukmė, f0 mediana, minimumas ir maksimumas, juostos plotis, nuolydis, vingiuotumas, "
            "harmonikų dalis, tonalumas, signalų dažnis ir intervalai tarp jų. Prieš skaičiavimus aprašyti šeši formos "
            "požymiai: pradžios ir pabaigos dažnio poslinkiai, U indeksas, išlinkimas ir dvi smulkios dažnio moduliacijos "
            "charakteristikos. Be to, kiekvienas signalas aprašytas keturiomis 32 taškų kreivėmis: f0, garsumo gaubtine, "
            "harmonikų ir f0 santykiu bei tonalumu."
        ),
    ]
    story += H("Reprezentacijos ir statistinė analizė", 1, "1.3.")
    story += [
        P("Kontūrų formos duomenimis grįstai išskirtos funkcine pagrindinių komponenčių analize (fPCA; Ramsay & "
          "Silverman, 2005), paliekant komponentes, kurios kartu paaiškina ≥ 90" + NB + "% variacijos. Palyginimui naudota "
          "spektrogramų variacinio autoenkoderio (VAE) reprezentacija (Goffinet ir kt., 2021), apmokyta tik "
          "mokymo vadų signalais."),
        P("Amžius nustatytas keteros (ridge) regresija, kurios hiperparametrai parinkti vidiniu kryžminiu patikrinimu "
          "tik su mokymo jaunikliais. Pagrindinis padalijimas – „palik vieną vadą“: modelis mokomas dviem vadomis ir "
          "tikrinamas trečiąja. Rodiklis – vidutinė absoliuti amžiaus paklaida jaunikliui, o jos skirtumo 95" + NB + "% "
          "pasikliautinasis intervalas (PI) gautas jauniklių lygmens savirankos (bootstrap) metodu. Amžiaus tendencijos "
          "vertintos mišriaisiais modeliais su atsitiktiniu jauniklio poslinkiu."),
    ]  # fmt: skip
    story += H("Rotacinis atradimo ir patvirtinimo planas", 1, "1.4.")
    story += [
        P("Naujiems (M9) požymiams taikytas rotacinis atradimo ir patvirtinimo planas (1" + NB + "pav.). Kiekvienai vadai "
          "požymiai ir jų kitimo kryptis atrinkti tik iš kitų dviejų vadų. Prognozės užrašytos į failą, užrakintos "
          "SHA-256 kontroline suma ir tik tada patikrintos nematytoje vadoje. Daugybiniams palyginimams taikyta "
          "Benjamini ir Hochbergo (1995) korekcija (q = 0,05). Radinys laikytas patvirtintu, jei pasitvirtino bent "
          "dviejose iš trijų nematytų vadų su prognozuota kryptimi."),
        figure("design", 1, "Rotacinis atradimo ir patvirtinimo planas",
               "Kiekvienoje eilutėje viena vada laikoma nematyta: požymiai atrenkami kitose dviejose, prognozė "
               "užrakinama ir tik tada tikrinama"),
    ]  # fmt: skip

    # ---------------------------------------------------------- results
    story += H("Rezultatai ir jų aptarimas", 0, "2.", new_page=True)
    story += H(
        "Signalo forma prideda informacijos, kurios nėra klasikiniuose požymiuose", 1, "2.1."
    )
    story += [
        P(f"Šeši iš anksto aprašyti formos požymiai sumažino nematytų vadų jauniklių amžiaus paklaidą nuo "
          f"{num(mae['c'].mean())} iki {num(mae['cs'].mean())} d (skirtumas {num(gain_c)} d, 95{NB}% PI {ci(0.657, 0.933)}); "
          f"paklaida sumažėjo {improved} iš 28 jauniklių (2{NB}pav.). VAE reprezentacija klasikinius požymius pagerino "
          f"panašiai (vidutiniškai iki {num(vae_mean)} d per tris atsitiktines sėklas), tačiau šeši formos požymiai "
          f"paaiškino apie 76{NB}% (74–78{NB}%) jos pranašumo. Taigi didžioji dalis išmoktos reprezentacijos "
          f"pranašumo nėra „juodoji dėžė“: ją galima įvardyti kaip signalo kontūro formą."),
        figure("gain", 2, "Amžiaus nustatymo paklaida nematytose vadose",
               "(A) Vidutinė jauniklio paklaida naudojant klasikinius požymius ir juos papildžius VAE reprezentacija, "
               "šešiais formos požymiais arba abiem (VAE – trijų sėklų vidurkis). (B) Kiekvieno jauniklio paklaida be "
               "formos požymių ir su jais; spalva žymi vadą"),
        P("Formos požymių kryptys sutapo su iš anksto suformuluota prognoze: su amžiumi stiprėjo pradžios ir pabaigos "
          "dažnio poslinkiai (U forma) ir smulki dažnio moduliacija. Kadangi ši hipotezė buvo suformuluota stebint tą "
          "pačią kohortą, šis rezultatas yra suderinamas su hipoteze, bet nėra nepriklausomas jos patvirtinimas."),
    ]  # fmt: skip
    story += H("Signalų formų rinkinys yra stabilus", 1, "2.2.")
    story += [
        P(f"Funkcinė PCA išskyrė šešias kontūro komponentes, kurios kartu paaiškino {num(100 * sum(var), 0)}{NB}% "
          f"variacijos. Pirmosios trys aprašo dažnio kitimo kryptį ({num(100 * var[0], 0)}{NB}%), arkos ir U formos "
          f"priešpriešą ({num(100 * var[1], 0)}{NB}%) ir banguotumą ({num(100 * var[2], 0)}{NB}%; 3{NB}pav.{NB}A). "
          "Kiekvieną kartą išmetus po vieną vadą, komponentės išliko beveik identiškos (3" + NB + "pav." + NB + "B), todėl jos "
          "atspindi visos kohortos, o ne vienos vados savybes."),
        figure("modes", 3, "Duomenimis išskirtos f0 kontūro formos",
               "(A) Trys pagrindinės fPCA komponentės: vidutinis kontūras ±2 standartiniai nuokrypiai (SD) "
               "komponentės kryptimi; skliaustuose – paaiškinta variacijos dalis. (B) Tos pačios komponentės, "
               "apskaičiuotos kiekvieną kartą be vienos vados"),
    ]  # fmt: skip
    story += H("Raidos pokyčiai pasitvirtino nematytose vadose", 1, "2.3.")
    story += [
        P(f"Iš 31 kandidatinio požymio {n_conf} pasitvirtino bent dviejose iš trijų nematytų vadų, iš jų {n_all3} – "
          "visose trijose (1" + NB + "lentelė). Su amžiumi signalai labiau kilo dažniu (pabaiga aukštesnė už pradžią), rečiau "
          "buvo arkos ir dažniau U formos, garsumo maksimumas slinko į signalo pabaigą, gilėjo garsumo moduliacija ir "
          "dažnėjo staigūs dažnio šuoliai (4" + NB + "pav.). Arkos komponentės mažėjimas glaudžiai susijęs su anksčiau "
          "aprašytu išlinkimo požymiu (Spearmano ρ = −0,95), todėl duomenimis grįstas metodas nepriklausomai "
          "atkartojo U formos kryptį."),
    ]  # fmt: skip
    t1_feats = [
        ("f0m_P1_med", "f0 kryptis (kylantys signalai)"), ("f0m_P2_med", "Arkos komponentė"),
        ("f0m_P3_med", "Banguotumo komponentė"), ("env_rise_frac_med", "Garsumo maksimumo vieta"),
        ("env_fall_frac_med", "Garsumo slopimo dalis"), ("env_am_depth_db_med", "Garsumo moduliacijos gylis"),
        ("share_jump", "Signalų su dažnio šuoliu dalis"), ("share_biphonation", "Bifonacijos tipo signalų dalis"),
        ("harmonic_ratio_db_med", "Harmonikų ir f0 santykis"), ("tonality_middle_med", "Tonalumas (vidurys)"),
        ("var_f0m_P1", "f0 krypties įvairovė"), ("var_log_duration", "Trukmės įvairovė"),
        ("var_within_family_f0m_P1", "f0 krypties įvairovė tipo viduje"),
    ]  # fmt: skip
    rows: list[list[Any]] = [["Požymis", "E2L1", "E2L2", "E2L3", "Vados"]]
    for f, lab in t1_feats:
        v = conf[f]
        cells = []
        for lit in ("E2L1", "E2L2", "E2L3"):
            r = v["per_rotation"].get(lit)
            if r is None:
                cells.append("–")
            else:
                txt = num(10 * r["slope"], 2, sign=True)
                cells.append(txt if r["confirmed"] else f"({txt})")
        rows.append([lab, *cells, f"{v['rotations_confirmed']} iš 3"])
    story += [table(1, "Užrakintų prognozių patikrinimas nematytose vadose. Vertės – amžiaus tendencija nematytoje vadoje "
                    "(standartizuoto požymio pokytis per 10 dienų); skliaustuose – nepatvirtinta (nereikšminga po FDR "
                    "korekcijos arba ne ta kryptimi). Atrinkti 13 iš 31 tikrinto požymio", rows,
                    [6.6, 2.4, 2.4, 2.4, 2.2])]  # fmt: skip
    story += [
        P("Harmonikų ir f0 santykis kito netiesiškai: jis mažėjo iki maždaug P12, o vėliau iš dalies atsistatė. "
          "Tonalumo sumažėjimas pasitvirtino tik dviejose vadose. Bifonacijos tipo signalai buvo reti, o jų "
          "detektorius dar nepatikrintas rankiniu žymėjimu, todėl šį rezultatą vertiname atsargiai."),
        figure("trends", 4, "Patvirtintų požymių kitimas su amžiumi",
               "(A) f0 krypties komponentė, (B) arkos komponentė, (C) garsumo maksimumo vieta (signalo trukmės dalis), "
               "(D) garsumo moduliacijos gylis. Plonos linijos – atskiri jaunikliai, storos – vados mediana "
               "(slankusis trijų dienų vidurkis)"),
        P("Šiuos pokyčius iliustruoja tipiški tikri signalai (5" + NB + "pav.): jaunesnių jauniklių signaluose dažniau "
          "pasitaiko krintantys ar plokšti kontūrai, o vyresnių – kylantys. Pavyzdžiai parinkti iš 5–15 ir 85–95 "
          "procentilių intervalų, kad atspindėtų tipiškus, o ne kraštutinius signalus."),
        figure("examples", 5, "Tipiški signalai su skirtinga f0 kitimo kryptimi (vada E2L2)",
               "(A) Krintantys signalai (f0 krypties komponentės 5–15 procentilis). (B) Kylantys signalai "
               "(85–95 procentilis). Spektrogramos 18–100" + NB + "kHz juostoje; virš kiekvienos – amžius"),
    ]  # fmt: skip
    story += H("Signalų įvairovė su amžiumi didėja", 1, "2.4.")
    story += [
        P("Tikėjomės, kad brendimas mažins signalų įvairovę (stereotipiškumo hipotezė). Rezultatai rodo priešingai: "
          "visose trijose vadose su amžiumi didėjo f0 komponenčių ir trukmės įvairovė (robastinis SD jauniklio dienoje), "
          "o dviejose vadose ji didėjo ir to paties tipo signalų viduje (6" + NB + "pav.). Vadinasi, įvairovės augimo "
          "negalima paaiškinti vien kintančiu signalų tipų mišiniu. Trukmės įvairovė buvo didžiausia apie P11–15. "
          "Taigi pirmosiomis trimis savaitėmis jauniklių signalų repertuaras plečiasi, o ne standartizuojasi."),
        figure("variability", 6, "Signalų įvairovės kitimas su amžiumi",
               "(A) f0 krypties komponentės, (B) trukmės logaritmo ir (C) f0 krypties komponentės to paties tipo "
               "signalų viduje įvairovė (robastinis SD). Linijų reikšmės kaip 4" + NB + "pav."),
    ]  # fmt: skip
    story += H("Ar nauji požymiai prideda amžiaus informacijos", 1, "2.5.")
    g_rows: list[list[Any]] = [["Požymių grupė", "E2L1", "E2L2", "E2L3", "Vados"]]
    g_names = {"f0_modes": "f0 formos komponentės", "env_modes": "Garsumo gaubtinės komponentės",
               "beyond": "Garsumas, harmonikos, tonalumas, šuoliai", "variability": "Signalų įvairovė",
               "all_selected": "Visi atrinkti požymiai"}  # fmt: skip
    for b, lab in g_names.items():
        cells = []
        for lit in ("E2L1", "E2L2", "E2L3"):
            v = gains[b]["per_rotation"][lit]
            pred = "+" if v["predicted_direction"] > 0 else "−"
            txt = (
                f"{num(v['mean_mae_difference_days'], 2, sign=True)}<br/>{ci(*v['ci95'])} [{pred}]"
            )
            cells.append(f"<b>{txt}</b>" if v["confirmed"] else txt)
        g_rows.append([lab, *cells, f"{gains[b]['rotations_confirmed']} iš 3"])
    story += [
        P("Griežčiausias testas – ar požymių grupė sumažina amžiaus paklaidą nematytoje vadoje, palyginti su geriausiu "
          "interpretuojamu baziniu lygiu (klasikiniai ir šeši formos požymiai). Nė viena grupė šio testo nepraėjo "
          "(2" + NB + "lentelė). f0 formos komponentės papildomos informacijos nesuteikė nė vienoje vadoje: šeši iš anksto "
          "aprašyti formos požymiai jau apima su amžiumi susijusią kontūro formą. Garsumo, harmonikų ir tonalumo "
          "požymiai paklaidą reikšmingai sumažino dviejose vadose (E2L1 ir E2L3, pastarojoje – 13 iš 14 jauniklių), "
          "tačiau E2L3 atveju užrakinta prognozė buvo neigiama. Pagal iš anksto nustatytas taisykles šis rezultatas "
          "nepatvirtintas ir laikomas tik kandidatu tolesniam tikrinimui."),
        table(2, "Amžiaus paklaidos sumažėjimas (d) nematytoje vadoje, papildžius bazinį lygį (klasikiniai ir šeši "
                 "formos požymiai). Teigiama vertė reiškia mažesnę paklaidą; žemiau – 95" + NB + "% PI ir laužtiniuose "
                 "skliaustuose užrakinta krypties prognozė; paryškinta – patvirtinta", g_rows, [5.6, 3.0, 3.0, 3.0, 1.8]),
    ]  # fmt: skip
    story += H("Likusi informacija, lytis ir individualumas", 1, "2.6.")
    rho = attr["rho_drop_when_occluded"]
    story += [
        P("Žvalgomojoje analizėje tirta, kurios spektrogramos sritys neša VAE amžiaus informaciją, kurios nepaaiškina "
          "klasikiniai ir formos požymiai. Uždengus f0 juostą, liekamasis įvertis kito labiausiai, tačiau daugiausia "
          f"informacijos prarasta uždengus antrąją harmoniką (koreliacijos sumažėjimas {num(rho['harm_middle'], 3)} "
          f"signalo viduryje, palyginti su ≤ {num(max(rho['fund_start'], rho['fund_middle'], rho['fund_end']), 3)} "
          "f0 juostoje). Harmonikos įtaka didžiausia P1–7 ir išnyksta iki P15–21 (7" + NB + "pav.), nes didėjant f0 "
          "antroji harmonika išeina už 100" + NB + "kHz analizės juostos ribų. Dalis šio efekto todėl yra techninė."),
        P("Lyties ir amžiaus sąveika nė vienam iš patvirtintų požymių nebuvo reikšminga po FDR korekcijos "
          f"(mažiausia p = {num(min(r['p'] for r in bio['sex_by_age']['features'].values()), 3)}). Individualūs "
          "jauniklių „balsai“ atpažįstami silpnai, bet geriau nei atsitiktinai: nematytą dieną teisingai atpažinto "
          f"jauniklio dalis didelėse vadose buvo {num(ident[ident['pups'] > 5]['balanced_accuracy'].min())}–"
          f"{num(ident[ident['pups'] > 5]['balanced_accuracy'].max())}, kai sumaišytų žymių atveju – "
          f"{num(ident[ident['pups'] > 5]['null_mean'].min())}–{num(ident[ident['pups'] > 5]['null_mean'].max())}. "
          "Aiškaus individualumo stiprėjimo su amžiumi nenustatyta."),
        figure("occlusion", 7, "Spektrogramos sričių uždengimo poveikis liekamajam amžiaus įverčiui",
               "Vidutinis absoliutus įverčio pokytis (d) uždengus f0 arba antrosios harmonikos (2f0) juostą signalo "
               "pradžioje, viduryje ar pabaigoje, pagal amžiaus grupes. Žvalgomoji analizė, VAE sėkla 0",
               width=10.5),
    ]  # fmt: skip
    story += [
        Spacer(1, 6),
        P("<b>Ribotumai.</b> Visi rezultatai gauti vienoje kohortoje (trys vados; E2L2 – tik 4 jaunikliai), todėl "
          "rotacijos nėra visiškai nepriklausomos. Dauguma požymių kinta su amžiumi, todėl patvirtinta tendencija yra "
          "palyginti žemas kartelės aukštis; griežtesni yra informacijos prieaugio testai. Netiesinių reiškinių "
          "detektoriai nepatikrinti rankiniu žymėjimu. Rezultatai aprašo akustinį fenotipą ir nėra tiesioginis "
          "nervinių mechanizmų įrodymas."),
    ]  # fmt: skip

    # ---------------------------------------------------------- conclusions
    story += H("Išvados", 0, new_page=True)
    story += bullets([
        f"Šeši iš anksto aprašyti signalo formos požymiai sumažina nematytų vadų jauniklių amžiaus paklaidą nuo "
        f"{num(mae['c'].mean())} iki {num(mae['cs'].mean())} d ({improved} iš 28 jauniklių) ir paaiškina apie 76{NB}% "
        "išmoktos VAE reprezentacijos pranašumo prieš klasikinius požymius.",
        f"Duomenimis išskirtos šešios f0 kontūro formos paaiškina {num(100 * sum(var), 0)}{NB}% kontūrų variacijos ir "
        "išlieka beveik identiškos nepriklausomai nuo to, kuri vada neįtraukta.",
        f"{n_conf} iš 31 naujo požymio raidos kryptis pasitvirtino nematytose vadose ({n_all3} – visose trijose): su "
        "amžiumi signalai labiau kyla dažniu, įgauna U formą, garsumo maksimumas slenka į pabaigą, gilėja garsumo "
        "moduliacija ir dažnėja dažnio šuoliai.",
        "Signalų įvairovė su amžiumi didėja, taip pat ir to paties tipo signalų viduje; stereotipiškumo hipotezė P1–P21 "
        "laikotarpiu nepasitvirtino.",
        "Nė viena nauja požymių grupė patikimai nepagerino amžiaus nustatymo, palyginti su klasikiniais ir šešiais "
        "formos požymiais; garsumo, harmonikų ir tonalumo požymiai yra kandidatas tolesniam tikrinimui.",
    ], numbered=True)  # fmt: skip

    story += H("Rekomendacijos ir tolesni tyrimai", 0, new_page=True)
    story += [
        P("Amžius patvirtina, kad metodas fiksuoja biologiškai prasmingus pokyčius, tačiau įdomiausias klausimas – ar "
          "signalo formos požymiai atskleidžia <b>farmakologinio ar kitokio poveikio</b> pėdsakus ten, kur klasikiniai "
          "rodikliai skirtumų nerado. Tokiu atveju stiprinamosios raidos skalė leistų įvertinti, ar paveikti jaunikliai "
          "akustiškai atrodo jaunesni ar vyresni už savo tikrąjį amžių."),
        P("Visa analizės grandinė (signalų aptikimas, kontūrai, formos požymiai ir raidos modelis) yra užfiksuota, todėl "
          "ją galima pritaikyti naujiems įrašams nieko neperderinant, o hipotezes užregistruoti prieš analizuojant "
          "duomenis. Tam reikia originalių WAV įrašų ir informacijos apie grupę, amžių ir vadą. Jei katedroje yra "
          "anksčiau atliktų poveikio tyrimų įrašų, jie būtų idealus nepriklausomas metodo patikrinimas, o jų klasikinė "
          "analizė – tiesioginis palyginimo taškas."),
        P("Taip pat rekomenduojama rankiniu žymėjimu patikrinti bifonacijos ir subharmonikų detektorius bei, esant "
          "galimybei, analizuoti platesnę dažnių juostą."),
    ]  # fmt: skip

    story += H("Literatūros sąrašas", 0, new_page=True)
    refs = [
        "Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach "
        "to multiple testing. <i>Journal of the Royal Statistical Society: Series B (Methodological)</i>, 57(1), 289–300.",
        "Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021). Low-dimensional learned feature spaces quantify "
        "individual and group differences in vocal repertoires. <i>eLife</i>, 10, e67855.",
        "Ramsay, J. O., & Silverman, B. W. (2005). <i>Functional data analysis</i> (2nd ed.). Springer.",
    ]
    story += [Paragraph(r, ref) for r in refs]
    from functools import partial

    from reportlab.pdfgen.canvas import Canvas

    doc.multiBuild(story, canvasmaker=partial(Canvas, initialFontName="Serif", initialFontSize=12))
    del NextPageTemplate, Flowable


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    mae = per_pup_mae()
    assert abs(mae["c"].mean() - 3.423) < 0.01 and abs(mae["cs"].mean() - 2.629) < 0.01, mae[
        ["c", "cs"]
    ].mean()
    table = json.loads((RUNS / "m4_overnight_table.json").read_text())
    vae = [r for r in table if r["rung"] == "vae"]
    vae_mean = float(np.mean([r["mae_combined"] for r in vae]))
    shape = json.loads(next(RUNS.glob("*shape-test-fd6e9c44/shape_test_summary.json")).read_text())[
        "test_a"
    ]
    both_mean = float(
        mae["cs"].mean()
        - np.mean(
            [v["vae_gain_over_classical_shape"] for v in shape["A3_vae_gain_absorbed"].values()]
        )
    )
    figs = {
        "design": fig_design(),
        "gain": fig_gain(mae, vae_mean, both_mean),
        "modes": fig_modes(),
        "trends": trend_fig([("f0m_P1_med", "f0 kryptis (kHz)"), ("f0m_P2_med", "Arkos komponentė (kHz)"),
                             ("env_rise_frac_med", "Maksimumo vieta (dalis)"), ("env_am_depth_db_med", "Moduliacija (dB)")],
                            "4_raida", ncols=2),
        "examples": fig_examples(),
        "variability": trend_fig([("var_f0m_P1", "Krypties įvairovė (kHz)"), ("var_log_duration", "Trukmės įvairovė"),
                                  ("var_within_family_f0m_P1", "Įvairovė tipo viduje (kHz)")], "6_ivairove", ncols=3),
        "occlusion": fig_occlusion(),
    }  # fmt: skip
    build(figs, mae, vae_mean, both_mean)
    print(PDF)


if __name__ == "__main__":
    main()
