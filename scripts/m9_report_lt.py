"""Lithuanian results update for the supervisor (results only, no title page).

Typography follows the VU GMC thesis guidelines (fonts, captions, numbering, figures); the document
structure does not.

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

    off = ScaledTranslation(-40 / 72, 8 / 72, ax.figure.dpi_scale_trans)
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
    a.set_ylabel("Vidutinė amžiaus paklaida (d)")
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
        b.set_xlabel("Vokalizacijos trukmė (%)")
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
    rows = [("A", "Krintančios", pos[pos["score"].between(q[0], q[1])].sample(6, random_state=0)),
            ("B", "Kylančios", pos[pos["score"].between(q[2], q[3])].sample(6, random_state=0))]  # fmt: skip
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
        "Likusi sritis",
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
    cb.set_label("Įverčio pokytis |Δ| (d)")
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
        PageBreak,  # fmt: skip
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

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
              title="Žiurkių jauniklių ultragarsinių vokalizacijų forma ir postnatalinė raida", author=AUTHOR,
              subject="Tarpiniai rezultatai", lang="lt-LT")  # fmt: skip
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)  # fmt: skip
    doc.addPageTemplates([PageTemplate("main", [frame], onPage=on_page)])

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

    # ---------------------------------------------------------- header (no title page: a short update)
    sub = ParagraphStyle("sub", fontName="Sans", fontSize=11, leading=14, textColor=colors.HexColor(MUTED),
                         spaceAfter=10)  # fmt: skip
    who = "visiems 28 jaunikliams" if improved == 28 else f"{improved} iš 28 jauniklių"
    story: list[Any] = [
        Paragraph("Žiurkių jauniklių ultragarsinių vokalizacijų forma ir raida: naujausi rezultatai", h1),
        Paragraph(f"{AUTHOR} · tarpiniai rezultatai · 2026 m. lapkritis", sub),
        P("Analizuotos 28 jauniklių iš trijų vadų (E2L1, E2L2, E2L3) vokalizacijos, įrašytos P1–P21 dienomis. Visi "
          "rezultatai tikrinti <b>testinėje vadoje</b> – tokioje, kurios duomenys nebuvo naudoti nei modeliui mokyti, "
          "nei požymiams atrinkti.", indent=False),
        Spacer(1, 4),
        P("<b>Sąvokos</b>", indent=False),
        *bullets([
            "<b>Klasikiniai požymiai</b> – įprasti suvestiniai dydžiai: vokalizacijų skaičius ir dažnumas, trukmė, "
            "vidutinis, mažiausias ir didžiausias dažnis, dažnių juostos plotis, nuolydis ir kt.",
            "<b>f0</b> – pagrindinis dažnis; jo kitimas per vokalizaciją vadinamas <b>kontūru</b>.",
            "<b>Šeši formos požymiai</b> – iš anksto apibrėžti kontūro formos matai: dažnio poslinkis vokalizacijos "
            "pradžioje ir pabaigoje, U indeksas, išlinkimas ir du smulkios dažnio moduliacijos matai.",
            "<b>VAE</b> (variacinis autoenkoderis) – neuroninis tinklas, kuris be jokių išankstinių nurodymų išmoksta "
            "kiekvienos vokalizacijos spektrogramą aprašyti nedideliu skaičiumi kintamųjų. Tai „juodosios dėžės“ "
            "palyginimo taškas: jis parodo, kiek amžiaus informacijos apskritai galima išgauti iš spektrogramos, "
            "bet nepasako, kokia ji.",
            "<b>fPCA</b> (funkcinė pagrindinių komponenčių analizė) – metodas, kuris iš pačių duomenų išskiria "
            "pagrindinius kontūro formų tipus (komponentes).",
        ]),
    ]  # fmt: skip
    story += H("Svarbiausia", 1)
    story += bullets([
        f"Šeši vokalizacijos formos požymiai sumažina testinių vadų jauniklių amžiaus nustatymo paklaidą nuo "
        f"{num(mae['c'].mean())} iki {num(mae['cs'].mean())} d ({who}) ir paaiškina apie 76{NB}% VAE "
        "reprezentacijos pranašumo prieš klasikinius požymius.",
        f"Duomenimis išskirtos šešios f0 kontūro formos paaiškina {num(100 * sum(var), 0)}{NB}% kontūrų dispersijos ir "
        "išlieka beveik identiškos, nesvarbu, kuri vada neįtraukta.",
        f"{n_conf} iš 31 naujo požymio amžiaus tendencija pasikartojo bent dviejose testinėse vadose ({n_all3} – "
        "visose trijose): su amžiumi daugėja kylančio dažnio ir U formos vokalizacijų, amplitudės maksimumas slenka "
        "pabaigos link, gilėja amplitudės moduliacija ir dažnėja dažnio šuoliai.",
        "Vokalizacijų įvairovė su amžiumi didėja, taip pat ir to paties tipo vokalizacijų viduje: P1–P21 laikotarpiu vokalizacijos "
        "netampa vienodesnės.",
        "Nė viena nauja požymių grupė patikimai nepagerino amžiaus nustatymo, palyginti su klasikiniais ir šešiais "
        "formos požymiais; amplitudės, harmonikų ir tonalumo požymiai lieka kandidatu tolesniam tikrinimui.",
    ], numbered=True)  # fmt: skip

    # ---------------------------------------------------------- results
    story += H(
        "Vokalizacijos forma suteikia informacijos, kurios neturi klasikiniai požymiai", 1, "1."
    )
    story += [
        P(f"Šeši iš anksto apibrėžti formos požymiai sumažino testinių vadų jauniklių amžiaus nustatymo paklaidą nuo "
          f"{num(mae['c'].mean())} iki {num(mae['cs'].mean())} d (skirtumas {num(gain_c)} d, 95{NB}% pasikliautinasis intervalas (PI) {ci(0.657, 0.933)}); "
          f"paklaida sumažėjo {who} (1{NB}pav.). Papildžius klasikinius požymius VAE reprezentacija, paklaida sumažėjo "
          f"panašiai (vidutiniškai iki {num(vae_mean)} d; trijų nepriklausomai apmokytų VAE modelių vidurkis), tačiau šeši formos požymiai "
          f"paaiškino apie 76{NB}% (74–78{NB}%) šio pagerėjimo. Vadinasi, didžioji dalis to, ką prideda išmokta "
          f"reprezentacija, nėra „juodoji dėžė“ – tai daugiausia vokalizacijos kontūro forma."),
        figure("gain", 1, "Amžiaus nustatymo paklaida testinėse vadose",
               "(A) Vidutinė jauniklio paklaida, naudojant klasikinius požymius ir juos papildžius VAE reprezentacija, "
               "šešiais formos požymiais arba abiem (VAE – trijų atskirai apmokytų modelių vidurkis). (B) Kiekvieno jauniklio paklaida be "
               "formos požymių ir su jais; spalva žymi vadą"),
        P("Formos požymių kryptys sutapo su iš anksto suformuluota prognoze: su amžiumi didėjo dažnio poslinkiai "
          "vokalizacijos pradžioje ir pabaigoje (U forma) ir stiprėjo smulki dažnio moduliacija. Tačiau hipotezė buvo "
          "suformuluota tiriant tą pačią kohortą, todėl šis rezultatas jai neprieštarauja, bet nėra nepriklausomas "
          "jos patvirtinimas."),
    ]  # fmt: skip
    story += H("Vokalizacijų formų rinkinys yra stabilus", 1, "2.")
    story += [
        P(f"fPCA išskyrė šešias kontūro komponentes, kurios kartu paaiškino {num(100 * sum(var), 0)}{NB}% "
          f"dispersijos. Pirmosios trys aprašo dažnio kitimo kryptį ({num(100 * var[0], 0)}{NB}%), skirtumą tarp arkos "
          f"ir U formos ({num(100 * var[1], 0)}{NB}%) ir banguotumą ({num(100 * var[2], 0)}{NB}%; 2{NB}pav.{NB}A). "
          "Pakartojus analizę be kiekvienos iš vadų, komponentės išliko beveik identiškos (2" + NB + "pav." + NB + "B), "
          "todėl jos atspindi visos kohortos, o ne atskiros vados savybes."),
        figure("modes", 2, "Duomenimis išskirtos f0 kontūro formos",
               "(A) Trys pagrindinės fPCA komponentės: vidutinis kontūras, pridėjus ir atėmus du komponentės "
               "standartinius nuokrypius (SD); skliaustuose – paaiškinta dispersijos dalis. (B) Tos pačios komponentės, "
               "apskaičiuotos neįtraukus vienos iš vadų"),
    ]  # fmt: skip
    story += H("Raidos pokyčiai pasikartojo testinėse vadose", 1, "3.")
    story += [
        P(f"Iš 31 kandidatinio požymio {n_conf} pasitvirtino bent dviejose iš trijų testinių vadų, iš jų {n_all3} – "
          "visose trijose (1" + NB + "lentelė). Kiekvienai vadai požymiai ir tikėtina jų kitimo kryptis buvo atrinkti tik "
          "iš kitų dviejų vadų, prognozės užfiksuotos ir tik tada patikrintos (taikant kelių palyginimų korekciją, FDR q = 0,05). Su amžiumi daugėjo vokalizacijų, kurių dažnis kyla (pabaiga aukštesnė už "
          "pradžią), mažėjo arkos ir daugėjo U formos kontūrų, amplitudės maksimumas slinko vokalizacijos pabaigos link, "
          "gilėjo amplitudės moduliacija ir dažnėjo staigūs dažnio šuoliai (3" + NB + "pav.). Arkos komponentė glaudžiai "
          "susijusi su anksčiau apibrėžtu išlinkimo požymiu (Spearmano ρ = −0,95), todėl duomenimis grįstas metodas, "
          "„nežinodamas“ hipotezės, rado tą pačią U formos kryptį."),
    ]  # fmt: skip
    t1_feats = [
        ("f0m_P1_med", "f0 kryptis (kylančios vokalizacijos)"), ("f0m_P2_med", "Arkos komponentė"),
        ("f0m_P3_med", "Banguotumo komponentė"), ("env_rise_frac_med", "Amplitudės maksimumo vieta"),
        ("env_fall_frac_med", "Amplitudės slopimo trukmė"), ("env_am_depth_db_med", "Amplitudės moduliacijos gylis"),
        ("share_jump", "Vokalizacijų su dažnio šuoliu dalis"), ("share_biphonation", "Bifonacijos tipo vokalizacijų dalis"),
        ("harmonic_ratio_db_med", "Harmonikų ir f0 santykis"), ("tonality_middle_med", "Tonalumas vokalizacijos viduryje"),
        ("var_f0m_P1", "f0 krypties įvairovė"), ("var_log_duration", "Trukmės įvairovė"),
        ("var_within_family_f0m_P1", "f0 krypties įvairovė to paties tipo vokalizacijų viduje"),
    ]  # fmt: skip
    rows: list[list[Any]] = [["Požymis", "E2L1", "E2L2", "E2L3", "Patvirtinta"]]
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
    story += [table(1, "Iš anksto užfiksuotų prognozių patikrinimas testinėse vadose. Stulpeliai – testinė vada; vertės – "
                    "požymio pokytis per 10 dienų standartinių nuokrypių vienetais; skliaustuose – nepatvirtinta (nereikšminga po FDR "
                    "korekcijos arba priešinga kryptimi). Pateikti 13 iš 31 tikrinto požymio", rows,
                    [6.6, 2.3, 2.3, 2.3, 2.5])]  # fmt: skip
    story += [
        P("Harmonikų ir f0 santykis kito netiesiškai: jis mažėjo iki maždaug P12, o vėliau vėl šiek tiek padidėjo. "
          "Tonalumo sumažėjimas pasitvirtino tik dviejose vadose. Bifonacijos tipo vokalizacijos buvo retos, o jų "
          "detektorius dar nepatikrintas rankiniu žymėjimu, todėl šį rezultatą reikia vertinti atsargiai."),
        figure("trends", 3, "Patvirtintų požymių kitimas su amžiumi",
               "(A) f0 krypties komponentė, (B) arkos komponentė, (C) amplitudės maksimumo vieta (vokalizacijos trukmės "
               "dalis), (D) amplitudės moduliacijos gylis. Plonos linijos – atskiri jaunikliai, storos – vados mediana "
               "(slankusis trijų dienų vidurkis)"),
        P("Šiuos pokyčius iliustruoja realūs vokalizacijų pavyzdžiai (4" + NB + "pav.): jaunesnių jauniklių vokalizacijų kontūrai "
          "dažniau krintantys ar plokšti, vyresnių – kylantys. Pavyzdžiai parinkti iš 5–15 ir 85–95 procentilių "
          "intervalų, kad atspindėtų tipiškus, o ne kraštutinius atvejus."),
        figure("examples", 4, "Tipiškos vokalizacijos su skirtinga f0 kitimo kryptimi (vada E2L2)",
               "(A) Krintančios vokalizacijos (f0 krypties komponentės 5–15 procentilis). (B) Kylančios vokalizacijos "
               "(85–95 procentilis). Spektrogramos 18–100" + NB + "kHz juostoje; virš kiekvienos nurodytas "
               "jauniklio amžius"),
    ]  # fmt: skip
    story += H("Vokalizacijų įvairovė su amžiumi didėja", 1, "4.")
    story += [
        P("Tikėtasi, kad jaunikliams augant vokalizacijos taps vienodesnės (stereotipiškumo hipotezė). Rezultatai rodo "
          "priešingai: visose trijose vadose su amžiumi didėjo f0 komponenčių ir trukmės įvairovė (robastinis "
          "standartinis nuokrypis per jauniklio dieną), o dviejose vadose ji didėjo ir to paties tipo vokalizacijų viduje "
          "(5" + NB + "pav.). Vadinasi, įvairovės augimo negalima paaiškinti vien besikeičiančiu vokalizacijų tipų santykiu. "
          "Trukmės įvairovė buvo didžiausia apie P11–15. Taigi per pirmąsias tris savaites jauniklių vokalizacijų "
          "repertuaras plečiasi, o ne siaurėja."),
        figure("variability", 5, "Vokalizacijų įvairovės kitimas su amžiumi",
               "(A) f0 krypties komponentės, (B) trukmės logaritmo ir (C) f0 krypties komponentės to paties tipo "
               "vokalizacijų viduje įvairovė (robastinis SD). Linijos – kaip 3" + NB + "pav."),
    ]  # fmt: skip
    story += H("Papildoma naujų požymių informacija nepatvirtinta", 1, "5.")
    g_rows: list[list[Any]] = [["Požymių grupė", "E2L1", "E2L2", "E2L3", "Patvirtinta"]]
    g_names = {"f0_modes": "f0 formos komponentės", "env_modes": "Amplitudės gaubtinės komponentės",
               "beyond": "Amplitudė, harmonikos, tonalumas, šuoliai", "variability": "Vokalizacijų įvairovė",
               "all_selected": "Visi atrinkti požymiai"}  # fmt: skip
    for b, lab in g_names.items():
        cells = []
        for lit in ("E2L1", "E2L2", "E2L3"):
            v = gains[b]["per_rotation"][lit]
            pred = "+" if v["predicted_direction"] > 0 else "−"
            txt = (
                f"{num(v['mean_mae_difference_days'], 2, sign=True)}<br/>{ci(*v['ci95'])} ({pred})"
            )
            cells.append(f"<b>{txt}</b>" if v["confirmed"] else txt)
        g_rows.append([lab, *cells, f"{gains[b]['rotations_confirmed']} iš 3"])
    story += [
        P("Griežčiausias patikrinimas – ar požymių grupė sumažina amžiaus paklaidą testinėje vadoje, palyginti su "
          "stipriausiu interpretuojamu baziniu modeliu (klasikiniai ir šeši formos požymiai). Nė viena grupė šio "
          "patikrinimo neišlaikė (2" + NB + "lentelė). f0 formos komponentės papildomos informacijos nesuteikė nė "
          "vienoje vadoje: šeši iš anksto apibrėžti formos požymiai jau apima su amžiumi susijusią kontūro formą. "
          "Amplitudės, harmonikų ir tonalumo požymiai paklaidą reikšmingai sumažino dviejose vadose (E2L1 ir E2L3, "
          "pastarojoje – 13 iš 14 jauniklių), tačiau E2L3 atveju iš anksto užfiksuota prognozė buvo priešinga. Pagal "
          "iš anksto nustatytas taisykles toks rezultatas nelaikomas patvirtintu – tai tik kandidatas tolesniam "
          "tikrinimui su naujais duomenimis."),
        table(2, "Amžiaus paklaidos sumažėjimas (d) testinėje vadoje, bazinį modelį (klasikiniai ir šeši formos "
                 "požymiai) papildžius požymių grupe. Teigiama vertė reiškia mažesnę paklaidą; po ja – 95" + NB + "% PI, "
                 "skliaustuose – iš anksto prognozuota kryptis; paryškinta – patvirtinta", g_rows,
                 [5.2, 3.0, 3.0, 3.0, 2.4]),
    ]  # fmt: skip
    story += H("Liekamoji informacija, lytis ir individualumas", 1, "6.")
    rho = attr["rho_drop_when_occluded"]
    story += [
        P("Žvalgomojoje analizėje tirta, kuriose spektrogramos srityse slypi papildoma VAE amžiaus informacija. "
          "Tam kiekvienai vokalizacijai apskaičiuotas "
          "liekamasis amžiaus įvertis – VAE amžiaus prognozės dalis, kurios nepaaiškina klasikiniai ir formos "
          "požymiai, – ir stebėta, kaip jis keičiasi uždengus skirtingas spektrogramos sritis. Uždengus f0 juostą, "
          "įvertis kito labiausiai, tačiau daugiausia informacijos prarasta uždengus antrąją harmoniką: įverčio "
          "koreliacija su tikruoju liekamuoju amžiumi sumažėjo "
          f"{num(rho['harm_middle'], 3)} vokalizacijos viduryje, o uždengus bet kurią f0 juostos dalį – ne daugiau kaip "
          f"{num(max(rho['fund_start'], rho['fund_middle'], rho['fund_end']), 3)}. Harmonikos įtaka didžiausia P1–7, o "
          "P15–21 ji beveik išnyksta (6" + NB + "pav.). Tikėtina, kad taip yra todėl, jog didėjant f0 antroji harmonika "
          "dažniausiai patenka už 100" + NB + "kHz analizės juostos ribų. Taigi dalis šio efekto gali būti techninė."),
        P("Lyties ir amžiaus sąveika nė vienam iš patvirtintų požymių nebuvo reikšminga po FDR korekcijos "
          f"(mažiausia nekoreguota p = {num(min(r['p'] for r in bio['sex_by_age']['features'].values()), 3)}). "
          "Individualūs jauniklių „balsai“ atpažįstami silpnai, bet geriau nei atsitiktinai: kitą dieną įrašytų "
          "vokalizacijų subalansuotas atpažinimo tikslumas didelėse vadose siekė "
          f"{num(ident[ident['pups'] > 5]['balanced_accuracy'].min())}–"
          f"{num(ident[ident['pups'] > 5]['balanced_accuracy'].max())}, o atsitiktinai sumaišius, kuriam jaunikliui priklauso vokalizacijos, – "
          f"{num(ident[ident['pups'] > 5]['null_mean'].min())}–{num(ident[ident['pups'] > 5]['null_mean'].max())}. "
          "Nenustatyta, kad individualumas su amžiumi aiškiai stiprėtų. Kadangi tą pačią dieną jauniklis ir įrašymo "
          "sesija sutampa, šis rezultatas yra preliminarus."),
        figure("occlusion", 6, "Spektrogramos sričių uždengimo poveikis liekamajam amžiaus įverčiui",
               "Vidutinis absoliutus įverčio pokytis (d), uždengus f0 arba antrosios harmonikos (2f0) juostą vokalizacijos "
               "pradžios, vidurio ar pabaigos trečdalyje; stulpeliai – amžiaus grupės. Žvalgomoji analizė (vienas "
               "VAE modelis)",
               width=10.5),
    ]  # fmt: skip
    story += [
        Spacer(1, 6),
        P("<b>Ribotumai.</b> Visi rezultatai gauti vienoje kohortoje (trys vados; E2L2 – tik 4 jaunikliai). Trys "
          "patikrinimai nėra visiškai nepriklausomi, nes kiekviena vada du kartus naudota požymių atrankai. Kadangi su amžiumi "
          "kinta dauguma požymių, patvirtinta amžiaus tendencija yra gana žema kartelė; griežtesnis yra papildomos "
          "informacijos patikrinimas (5 skyrelis). Bifonacijos, subharmonikų ir dažnio šuolių detektoriai nepatikrinti rankiniu žymėjimu. "
          "Rezultatai aprašo akustinius pokyčius, tačiau nieko nesako apie juos lemiančius mechanizmus."),
    ]  # fmt: skip

    story += H("Kas toliau", 1)
    story += [
        P("Labai laukiame naujų duomenų. Visa analizės grandinė (vokalizacijų aptikimas, kontūrai, formos požymiai ir "
          "raidos modelis) yra užfiksuota, todėl naujus įrašus bus galima išanalizuoti iš karto ir nieko nekeičiant. "
          "Nauja kohorta leistų nepriklausomai patikrinti šiuos rezultatus, o kandidatus (amplitudės, harmonikų ir "
          "tonalumo požymius) – iš anksto užregistruotu testu."),
        P("Kol kas planuojame rankiniu žymėjimu patikrinti bifonacijos ir subharmonikų detektorius ir įvertinti, "
          "ar verta vokalizacijas analizuoti platesnėje dažnių juostoje."),
    ]  # fmt: skip

    from functools import partial

    from reportlab.pdfgen.canvas import Canvas

    doc.multiBuild(story, canvasmaker=partial(Canvas, initialFontName="Serif", initialFontSize=12))


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
        "gain": fig_gain(mae, vae_mean, both_mean),
        "modes": fig_modes(),
        "trends": trend_fig([("f0m_P1_med", "f0 kryptis (kHz)"), ("f0m_P2_med", "Arkos komponentė (kHz)"),
                             ("env_rise_frac_med", "Ampl. maksimumo vieta"), ("env_am_depth_db_med", "AM gylis (dB)")],
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
