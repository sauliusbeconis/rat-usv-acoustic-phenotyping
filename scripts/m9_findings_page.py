"""Build the private M9 results page (private/results/M9_findings.html) from private/results/M9/*.png."""

import base64
from pathlib import Path

FIG = Path("private/results/M9")
OUT = Path("private/results/M9_findings.html")


def img(name: str, alt: str) -> str:
    data = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'


BADGE = {
    "yes": '<span class="badge yes">Confirmed in held-out litters</span>',
    "no": '<span class="badge no">Not confirmed</span>',
    "contra": '<span class="badge contra">Confirmed, but opposite to the hypothesis</span>',
    "explore": '<span class="badge explore">Exploratory</span>',
}

sections = [
    (
        "1. Calls have a small, stable shape vocabulary",
        "yes",
        """<p><b>What we did.</b> We took every confident call's pitch contour (f0), stretched it to the same length,
        and let the data choose the main ways contours differ (functional PCA). We did this three times, each time
        <i>without</i> one litter.</p>
        <p><b>What we found.</b> Six shapes explain 91% of contour variation, and they come out <b>almost identical
        whichever litter is left out</b> (bottom row: three coloured lines lie on top of each other). The two biggest are
        <b>sweep direction</b> (up vs down, 50%) and <b>arch vs U</b> (20%).</p>
        <p><b>Why it matters.</b> The shape vocabulary of rat-pup calls is a property of the animals, not of one litter.
        It is the data-driven vocabulary we needed instead of hand-picked shapes.</p>""",
        [
            (
                "modes_f0.png",
                "f0 shape modes: top = mean ± 2 SD of each mode; bottom = the same mode refitted without each litter",
            )
        ],
    ),
    (
        "2. As pups grow, calls sweep upward, lose their arch and peak later 🎉",
        "yes",
        """<p><b>The test.</b> For each litter we (1) picked age-related features using only the other two litters,
        (2) wrote the predicted direction into a file and locked it with a fingerprint (SHA-256), and only then
        (3) checked the unseen litter. A finding counts only if it holds in at least 2 of the 3 unseen litters after correction for
        multiple testing.</p>
        <p><b>What held up, in all three litters:</b></p>
        <ul>
          <li><b>Upward sweeps increase</b> (f0 mode P1 ↑): older pups' calls end higher than they start.</li>
          <li><b>Fewer arches, more U shapes</b> (f0 mode P2 ↓). This is a data-driven replication of H-M7a's U-shape
          direction, found without being told to look for it (ρ = −0.95 with the hand-made curvature measure).</li>
          <li><b>Loudness peaks later in the call</b> (envelope rise ↑, fall ↓, envelope mode P1 ↑).</li>
          <li><b>Deeper loudness wobble</b> (amplitude-modulation depth ↑) and <b>more frequency jumps</b>.</li>
          <li>Slightly more <b>biphonation-like</b> calls (rare; this detector is new and not hand-checked).</li>
        </ul>
        <p>In 2 of 3 litters: <b>harmonics weaken relative to the fundamental</b> (not linear: lowest around P12) and
        <b>tonality drops</b>.</p>
        <p><b>Why it's exciting.</b> These are locked predictions that came true on pups the analysis had never seen. Development changes
        not just <i>what pitch</i> a pup calls at, but <i>how the call moves and swells over its 30–60 ms</i>, and it does
        so the same way in every litter.</p>""",
        [
            (
                "confirmation.png",
                "Every candidate feature × held-out litter: colour = age slope in the unseen litter, ✓ = locked prediction confirmed, ★ = confirmed in ≥ 2 of 3",
            ),
            (
                "trends_shape.png",
                "Confirmed features by age: thin lines = individual pups, thick = litter median",
            ),
            (
                "examples_f0_P1.png",
                "Real calls: low vs high sweep-direction score (more upward at the bottom)",
            ),
            ("examples_f0_P2.png", "Real calls: low (U/edge-sweep) vs high (arched) f0 mode P2"),
            (
                "examples_env_P1.png",
                "Real calls: low vs high envelope mode P1 (loudness building toward the end)",
            ),
        ],
    ),
    (
        "3. Surprise: calls get more variable with age, not more stereotyped",
        "contra",
        """<p><b>The idea.</b> As the vocal system matures, calls might become more stereotyped, with call-to-call spread
        shrinking.</p>
        <p><b>What we found: the opposite, in all three litters.</b> The spread of call shapes (all three main f0 modes)
        and of call durations <b>grows</b> from P1 to P21. It also grows <i>within</i> call types (2 of 3 litters),
        so it is not just a change in the mix of call types.</p>
        <p><b>Plain reading.</b> In the first three weeks, pups don't settle into a fixed call. Their repertoire
        <b>diversifies</b>. This is a clear, replicated, directional result. It rejects the simple stereotypy idea for this age
        range (duration spread peaks mid-period, around P11–15).</p>""",
        [("trends_variability.png", "Call-to-call variability per pup-day (robust SD) by age")],
    ),
    (
        "4. Does any of this add age information beyond what we already had?",
        "no",
        """<p><b>The strict test.</b> Does adding each feature block reduce the age-prediction error on the unseen litter, compared
        with the best interpretable baseline (classical features + the six H-M7a shape measures)?</p>
        <ul>
          <li><b>f0 shape modes: no gain in any litter.</b> This is good news for interpretability: the six
          pre-registered shape measures already capture the age-relevant contour shape.</li>
          <li><b>Beyond-contour features</b> (envelope timing, AM depth, harmonics, tonality profile, jumps):
          they helped clearly in <b>two</b> litters (E2L1 +0.18 days, E2L3 +0.25 days with 13/14 pups improving). But for E2L3 the
          locked prediction said "won't help", so under our rules it counts as <b>1 of 3: not confirmed</b>. It is a
          promising candidate for a pre-registered test on new data, not a finding.</li>
          <li>Envelope modes, variability and all features together: not confirmed.</li>
        </ul>
        <p>Honest bottom line: M9 explains <i>how</i> calls change with age much better. It does not yet show a new
        source of age information beyond classical + shape features.</p>""",
        [
            (
                "gains.png",
                "Age error saved per pup vs the classical + H-M7a baseline, per feature block and held-out litter",
            )
        ],
    ),
    (
        "5. Where does the VAE's leftover advantage live? Partly in the harmonic",
        "explore",
        """<p>We trained a small decoder to predict the part of age that classical + shape features miss, from the VAE's
        view of each call. We then blanked parts of each call's spectrogram, following its own pitch track.</p>
        <ul>
          <li>Blanking the <b>fundamental</b> changes the score most, but blanking the <b>harmonic band</b> removes the most
          <i>information</i> (the score's correlation with the true leftover age drops ~3–4× more).</li>
          <li>The harmonic matters mostly in <b>P1–7</b> calls and fades by P15–21. By then pitch is so high that the harmonic sits above our
          100 kHz analysis band, so part of this is a recording-band effect.</li>
          <li>Walking through the VAE's latent space along this direction produced almost no visible change (looks like an
          artifact). We report this as an uninformative result.</li>
        </ul>""",
        [
            (
                "m9_occlusion_regions.png",
                "Mean change in the leftover-age score when each region is blanked, by age band",
            ),
            (
                "m9_traversal_E2L1.png",
                "Latent traversal (uninformative: differences sit in a small fixed blob)",
            ),
        ],
    ),
    (
        "6. Sex and individual voices",
        "explore",
        """<ul>
          <li><b>Sex:</b> no confirmed feature develops differently in males and females after correction (28 pups).</li>
          <li><b>Individual voices:</b> which pup made a call can be guessed on a day the model never saw, above
          shuffled labels (5 of 6 cells in the large litters, p &lt; 0.05). The signal is weak (e.g. 17% vs 9%), and it does
          <b>not</b> clearly strengthen with age. Pup and recording session are tangled within a day.</li>
        </ul>""",
        [
            (
                "identity.png",
                "Which-pup accuracy on an unseen day, by litter and age band (solid = real, dashed = shuffled)",
            )
        ],
    ),
]

css = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--muted:#5d5d63;--card:#fff;--line:#e6e2da;--yes:#1f7a4d;--no:#8a8a8a;--contra:#b35c00;--explore:#3b5bab}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececef;--muted:#a3a3ab;--card:#1d1d21;--line:#2e2e34}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:2rem;margin:.2em 0}h2{font-size:1.35rem;margin:0 0 .4em}
.lede{color:var(--muted);font-size:1.05rem}
.tldr{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:24px 0}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:22px 0}
.badge{display:inline-block;font-size:.78rem;font-weight:600;padding:2px 10px;border-radius:999px;color:#fff;margin-bottom:10px}
.yes{background:var(--yes)}.no{background:var(--no)}.contra{background:var(--contra)}.explore{background:var(--explore)}
figure{margin:16px 0}img{max-width:100%;height:auto;border-radius:6px;background:#fff}
figcaption{color:var(--muted);font-size:.88rem}
.small{color:var(--muted);font-size:.85rem}
code{font-size:.85em}
"""

tldr = """<div class="tldr"><b>In one minute.</b>
<ul>
<li>Rat-pup calls have a <b>stable shape vocabulary</b>, and the way it shifts with age <b>replicates in every unseen litter</b>.
With age, calls sweep upward more, lose their arch, swell later and wobble more. 🎉</li>
<li><b>Surprise:</b> calls become <b>more variable</b> with age, not more stereotyped. The repertoire diversifies through P21.</li>
<li>None of the new features <b>reliably adds age information</b> beyond classical + the six H-M7a shape measures. Those six
already capture the contour; beyond-contour features are a promising but unconfirmed candidate.</li>
<li>Exploratory: the VAE's remaining edge sits partly in the <b>harmonic</b> of young pups' calls.</li>
</ul></div>"""

body = "".join(
    f"<section><h2>{t}</h2>{BADGE[b]}{text}{''.join(img(f, a) for f, a in figs)}</section>"
    for t, b, text, figs in sections
)
limits = """<section><h2>Limits, honestly</h2><ul>
<li>One cohort, three litters; E2L2 has only 4 pups. Each litter is in the discovery set twice, so the rotations are not independent.</li>
<li>Almost everything changes with age, so "age trend confirmed" is a low bar. The block gain tests are the strict ones, and none passed.</li>
<li>The biphonation and subharmonic detectors are new and not hand-validated. The harmonic is only visible while 2·f0 &lt; 100 kHz.</li>
<li>Descriptive only: these are acoustic changes, not evidence of a specific neural mechanism.</li>
</ul>
<p class="small">Runs: curves <code>20261001T175036.684944Z-features-7f6e6304</code>; M9 <code>20261001T181436.048902Z-m9-5a678459</code>
(locks E2L1 bd79dbb4…, E2L2 5ba276e3…, E2L3 9c09ec26…); attribution <code>20261001T181732.397771Z-m9-attribution-bcb0fb9a</code>.
Methods, deviations and full numbers: <code>docs/decisions/0021-m9-results.md</code>. Private: contains cohort spectrograms.</p></section>"""

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>M9 Findings</title><style>{css}</style></head><body><main>
<p class="small">M9 · data-driven discovery of developmental call structure · 2026-10-01</p>
<h1>How rat-pup calls change over the first three weeks</h1>
<p class="lede">28 pups, 3 litters, 38,170 confident calls, 574 recordings (494 pup-days with ≥ 5 calls). Every claim marked "confirmed" was predicted from two litters,
locked, and then checked on a third litter the analysis had never seen.</p>
{tldr}{body}{limits}</main></body></html>"""
OUT.write_text(html)
print(OUT, len(html) // 1024, "KiB")
