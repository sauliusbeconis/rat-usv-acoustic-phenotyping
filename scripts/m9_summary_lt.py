"""Short Lithuanian summary for the supervisor: figures + PDF (private/results/santrauka_LT.pdf).

Run: uv run --frozen --with reportlab python scripts/m9_summary_lt.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path(".local/runs")
M9 = RUNS / "20261001T181436.048902Z-m9-5a678459"
FEAT = RUNS / "20261001T175036.684944Z-features-7f6e6304"
IMGS = RUNS / "20261001T124005.643225Z-call-images-c60840b7"
OUT = Path("private/results/M9_lt")
PDF = Path("private/results/santrauka_LT.pdf")
LIT = {"E2L1": "#2a6fdb", "E2L2": "#d1495b", "E2L3": "#2a9d5c"}
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9})


def save(fig: plt.Figure, name: str) -> Path:
    fig.tight_layout()
    path = OUT / name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_gain() -> Path:
    labels = [
        "Klasikiniai\npožymiai",
        "Klasikiniai +\n6 formos požymiai",
        "Klasikiniai +\nVAE (išmoktas)",
    ]
    vals = [3.42, 2.63, 2.69]
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    bars = ax.bar(labels, vals, color=["#9aa0a6", "#2a6fdb", "#7b5ea7"], width=0.6)
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center")
    ax.set_ylabel("Amžiaus paklaida, dienos\n(vidurkis per jauniklį)")
    ax.set_ylim(0, 3.9)
    ax.set_title("Nematytos vados amžiaus nustatymas iš balso", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "1_paklaida.png")


def fig_modes() -> Path:
    s = json.loads((M9 / "m9_confirmation.json").read_text())
    titles = ["Kryptis: kylantis ↔ krintantis", "Arka ↔ U forma", "Banguotumas"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8))
    with np.load(M9 / "m9_bases_holdout_E2L2.npz") as z:
        mean, modes, sd = z["f0_mean"], z["f0_modes"], z["f0_sd"]
    var = s["fpca"]["E2L2"]["f0"]["var"]
    tau = np.linspace(0, 100, mean.size)
    for j in range(3):
        ax = axes[0, j]
        ax.plot(tau, mean + 2 * sd[j] * modes[j], color="#d1495b")
        ax.plot(tau, mean - 2 * sd[j] * modes[j], color="#2a6fdb")
        ax.set_title(f"{titles[j]} ({100 * var[j]:.0f} %)", fontsize=9)
        ax.set_ylabel("f0 nuokrypis, kHz" if j == 0 else "")
        for rot, c in LIT.items():
            with np.load(M9 / f"m9_bases_holdout_{rot}.npz") as z:
                axes[1, j].plot(tau, z["f0_modes"][j], color=c, lw=1.2, label=f"be {rot}")
        axes[1, j].set_xlabel("% signalo trukmės")
    axes[1, 0].set_ylabel("formos komponentė")
    axes[1, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Duomenų išskirtos signalo formos (viršuje); jos beveik vienodos, kad ir kurią vadą išmestume (apačioje)",
                 fontsize=9)  # fmt: skip
    return save(fig, "2_formos.png")


def fig_trends(cols: list[tuple[str, str]], name: str, title: str) -> Path:
    day = pd.read_parquet(M9 / "m9_day_features_holdout_E2L2.parquet")
    fig, axes = plt.subplots(1, len(cols), figsize=(2.0 * len(cols) + 0.6, 2.6), squeeze=False)
    for ax, (col, lab) in zip(axes[0], cols, strict=True):
        d = day.dropna(subset=[col])
        for _, g in d.groupby("animal_id"):
            g = g.sort_values("age")
            ax.plot(g["age"], g[col], color=LIT[g["litter_id"].iloc[0]], alpha=0.2, lw=0.6)
        for lit, g in d.groupby("litter_id"):
            m = g.groupby("age")[col].median()
            ax.plot(m.index, m.to_numpy(), color=LIT[lit], lw=2, label=lit)
        ax.set_title(lab, fontsize=8.5)
        ax.set_xlabel("postnatalinė diena")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=9.5)
    return save(fig, name)


def fig_examples() -> Path:
    from rat_usv.evaluation.m9 import load_calls, project

    calls, curves = load_calls(FEAT)
    idx = pd.read_parquet(IMGS / "calls_index.parquet").reset_index()
    imgs = np.load(IMGS / "calls_images.npy", mmap_mode="r")
    with np.load(M9 / "m9_bases_holdout_E2L2.npz") as z:
        score = project(
            curves["f0"][calls["row"].to_numpy()].astype(float),
            {"mean": z["f0_mean"], "modes": z["f0_modes"]},
        )[:, 0]
    calls["score"] = score
    pos = idx.merge(calls[["wav_path", "call_index", "score"]], on=["wav_path", "call_index"])
    pos = pos[pos["litter_id"] == "E2L2"]
    q = pos["score"].quantile([0.05, 0.15, 0.85, 0.95]).to_numpy()
    rows = [("Krintantys", pos[pos["score"].between(q[0], q[1])].sample(6, random_state=0)),
            ("Kylantys", pos[pos["score"].between(q[2], q[3])].sample(6, random_state=0))]  # fmt: skip
    fig, axes = plt.subplots(2, 6, figsize=(7.2, 2.8))
    for r, (lab, sel) in enumerate(rows):
        for c, (_, row) in enumerate(sel.iterrows()):
            axes[r, c].imshow(imgs[int(row["index"])], cmap="magma", aspect="auto")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].set_title(f"P{int(row['postnatal_age_days'])}", fontsize=7)
        axes[r, 0].set_ylabel(lab, fontsize=8)
    fig.suptitle("Tikri signalai (spektrogramos): tipiški krintantys ir kylantys", fontsize=9.5)
    return save(fig, "5_pavyzdziai.png")


def build_pdf(figs: dict[str, Path]) -> None:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer

    font_dir = Path("/usr/share/fonts/liberation")
    pdfmetrics.registerFont(TTFont("LS", str(font_dir / "LiberationSans-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("LSB", str(font_dir / "LiberationSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("LSI", str(font_dir / "LiberationSans-Italic.ttf")))
    pdfmetrics.registerFontFamily("LS", normal="LS", bold="LSB", italic="LSI", boldItalic="LSB")
    body = ParagraphStyle("b", fontName="LS", fontSize=10, leading=14, spaceAfter=5)
    small = ParagraphStyle(
        "s", parent=body, fontSize=8.5, leading=11, textColor=HexColor("#555555")
    )
    h1 = ParagraphStyle("h1", fontName="LSB", fontSize=16, leading=20, spaceAfter=4)
    h2 = ParagraphStyle("h2", fontName="LSB", fontSize=12, leading=15, spaceBefore=10, spaceAfter=4,
                        textColor=HexColor("#1f3d7a"))  # fmt: skip
    box = ParagraphStyle("box", parent=body, backColor=HexColor("#eef3fb"), borderPadding=8,
                         borderColor=HexColor("#c9d6ee"), borderWidth=0.6, spaceBefore=14, spaceAfter=10)  # fmt: skip

    def fig(key: str, width_cm: float = 16.5) -> Image:
        from PIL import Image as PILImage

        w, h = PILImage.open(figs[key]).size
        return Image(str(figs[key]), width=width_cm * cm, height=width_cm * cm * h / w)

    def bullets(items: list[str]) -> list[Paragraph]:
        return [
            Paragraph(
                f"•&nbsp;&nbsp;{t}",
                ParagraphStyle("li", parent=body, leftIndent=12, firstLineIndent=-9),
            )
            for t in items
        ]

    S: list = [
        Paragraph(
            "Žiurkių jauniklių ultragarsiniai signalai: daugiau informacijos nei klasikiniuose rodikliuose",
            h1,
        ),
        Paragraph(
            "Bakalauro darbo tarpinių rezultatų santrauka · Saulius Beconis · 2026-10-02", small
        ),
        Paragraph(
            "<b>Trumpai.</b> Iš 28 jauniklių (3 vados, P1–P21, 38 170 patikimų signalų) įrašų radome, kad "
            "<b>signalo forma</b> – kaip kinta dažnis ir garsumas per vieną 30–60 ms signalą – neša raidos informaciją, "
            "kurios klasikiniai rodikliai (trukmė, vidutinis dažnis, juostos plotis, nuolydis) nepagauna. "
            "Pagrindiniai radiniai buvo <b>iš anksto užfiksuoti ir patikrinti vadose, kurių analizė nematė</b>.",
            box,
        ),
        Paragraph("1. Forma prideda informacijos, kurios klasikiniai rodikliai neturi", h2),
        Paragraph(
            "Amžių naudojame kaip patikrinimo „liniuotę“: jei požymiai neša raidos informaciją, iš jų turi būti galima "
            "nustatyti nematytos vados jauniklių amžių. Šeši iš anksto aprašyti formos požymiai (pradžios ir pabaigos "
            "dažnio poslinkiai, išlinkimas, smulki dažnio moduliacija) sumažino paklaidą nuo 3,42 iki 2,63 dienos; "
            "pagerėjo <b>visiems 28 iš 28 jauniklių</b>. Jie paaiškina apie <b>76 %</b> to pranašumo, kurį duoda "
            "neuroniniu tinklu išmokta reprezentacija (VAE), todėl rezultatas yra ne „juodoji dėžė“, o įvardijamos savybės.",
            body,
        ),
        fig("gain", 11),
        Paragraph("2. Signalų formų „žodynas“ yra stabilus", h2),
        Paragraph(
            "Leidome duomenims patiems išskirti pagrindines kontūro formas (funkcinė PCA). Šešios formos paaiškina "
            "91 % kontūrų įvairovės ir išeina <b>beveik identiškos, kad ir kurią vadą išmestume</b> – tai jauniklių, "
            "o ne vienos vados savybė.",
            body,
        ),
        fig("modes"),
        Paragraph("3. Kaip keičiasi signalai augant (patvirtinta nematytose vadose)", h2),
        Paragraph(
            "Saugiklis: kiekvienai vadai požymius ir jų kitimo kryptį pasirinkome tik iš kitų dviejų vadų, "
            "prognozę užrakinome (SHA-256), ir tik tada tikrinome nematytą vadą. Radinys laikomas patvirtintu, jei "
            "pasitvirtina ≥ 2 iš 3 vadų po daugybinių palyginimų korekcijos (FDR 5 %). Visose trijose vadose:",
            body,
        ),
        *bullets(
            [
                "signalai labiau <b>kyla</b> dažniu (pabaiga aukštesnė už pradžią);",
                "mažiau <b>arkos</b> formų, daugiau <b>U</b> formų (tai nepriklausomai atkartoja anksčiau iškeltą hipotezę);",
                "<b>garsumo maksimumas</b> slenka į signalo pabaigą; gilėja garsumo moduliacija;",
                "dažnėja staigūs dažnio <b>šuoliai</b>.",
            ]
        ),
        fig("trends"),
        fig("examples", 15),
        Paragraph("4. Netikėta: augant signalai tampa įvairesni, ne stereotipiškesni", h2),
        Paragraph(
            "Tikėjomės, kad brendimas „sustandartins“ signalus. Visose trijose vadose matėme priešingai: "
            "signalų formų ir trukmių <b>įvairovė auga</b> nuo P1 iki P21, ir ne tik dėl kintančio signalų tipų "
            "mišinio – ji auga ir to paties tipo signalų viduje. Repertuaras šiuo laikotarpiu plečiasi.",
            body,
        ),
        fig("variability", 13),
        Paragraph("5. Ko (dar) neparodėme – sąžiningai", h2),
        *bullets(
            [
                "Nauji, plačiau išskirti požymiai (garsumo eiga, harmonikos, tonalumas) amžiaus nustatymą pagerino dviejose "
                "vadose iš trijų, bet vienoje – priešingai užrakintai prognozei, todėl pagal mūsų taisykles tai "
                "<b>kandidatas, ne įrodymas</b>.",
                "Lyties skirtumų raidoje neradome. Individualūs „balsai“ atpažįstami silpnai.",
                "Visa tai – viena kohorta; nepriklausomo patikrinimo su kitais duomenimis dar nėra.",
                "Kalbame apie akustinį fenotipą, ne apie konkrečius nervinius mechanizmus.",
            ]
        ),
        Paragraph("Kitas žingsnis: poveikio tyrimai", h2),
        Paragraph(
            "Amžius tik patikrina, kad metodas veikia. Įdomiausias klausimas – ar formos požymiai pastebi "
            "<b>farmakologinio ar kitokio poveikio</b> pėdsaką ten, kur klasikiniai rodikliai skirtumų nerado. "
            "Visa grandinė (signalų aptikimas, kontūrai, formos požymiai) jau užfiksuota, todėl ją galima pritaikyti "
            "naujiems įrašams nieko neperderinant, o hipotezes užregistruoti prieš žiūrint į duomenis. "
            "Tam reikėtų tik originalių <b>WAV įrašų</b> ir informacijos apie grupę, amžių bei vadą. "
            "Jei katedroje yra tokių anksčiau atliktų tyrimų įrašų, tai būtų idealus nepriklausomas patikrinimas, "
            "o klasikinė jų analizė – tiesioginis palyginimo taškas.",
            box,
        ),
        Paragraph(
            "Metodai ir visi skaičiai: sprendimų užrašai 0013–0021 projekto saugykloje; kiekvienas paleidimas turi "
            "užfiksuotą kodo versiją, duomenų padalijimą ir sėklą.",
            small,
        ),
    ]
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="USV tarpiniai rezultatai", author="Saulius Beconis")  # fmt: skip
    del KeepTogether, Spacer
    doc.build(S)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {
        "gain": fig_gain(),
        "modes": fig_modes(),
        "trends": fig_trends(
            [("f0m_P1_med", "Kylančio dažnio\nkomponentė ↑"), ("f0m_P2_med", "Arkos komponentė ↓\n(daugiau U formų)"),
             ("env_rise_frac_med", "Garsumo maksimumo\nvieta ↑"), ("env_am_depth_db_med", "Garsumo moduliacijos\ngylis, dB ↑")],
            "3_raida.png", "Pokyčiai su amžiumi (plonos linijos – jaunikliai, storos – vados mediana)"),
        "variability": fig_trends(
            [("var_f0m_P1", "Kontūro krypties\nįvairovė ↑"), ("var_log_duration", "Trukmės\nįvairovė ↑"),
             ("var_within_family_f0m_P1", "Įvairovė to paties\ntipo viduje ↑")],
            "4_ivairove.png", "Signalų įvairovė (robastinis SD per jauniklio dieną)"),
        "examples": fig_examples(),
    }  # fmt: skip
    build_pdf(figs)
    print(PDF)
