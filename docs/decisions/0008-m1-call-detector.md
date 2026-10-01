# 0008 — M1 call detector

Date: 2026-10-01. Status: implemented; hand-check pending.

## Decision
Use a transparent, in-repo spectrogram detector (`src/rat_usv/signal/detect.py`, parameters in `configs/detection/default.yaml`) instead of DeepSqueak, USVSEG or ContourUSV.

## Method
1. Hann STFT, 512 samples (2.4 ms) with a 128-sample hop (0.6 ms) at 214285 Hz; band 18–100 kHz.
2. Power is smoothed over 3 frequency bins × 5 frames, converted to dB, and whitened by subtracting each frequency bin's median over the 30 s chunk.
3. A frame is a candidate when its peak exceeds 6 dB and it is **tonal**: no more than 4 kHz of the band lies within 10 dB of the frame peak. Broadband frames (≥ 30% of bins above 6 dB) are rejected as clicks.
4. Candidate frames are merged across gaps of 8 ms or less. A segment becomes a call if it reaches 10 dB, lasts 4–1500 ms, and its loudest frame is at 25 kHz or higher.
5. Per call: start, end, duration, peak/min/max/mean/start/end frequency of the peak contour, peak and mean SNR, and the share of rejected frames.

## Why
- DeepSqueak and USVSEG are MATLAB tools, and ContourUSV's code was not confirmed. An in-repo detector is reproducible, testable and fast (~20 s per 5-min file per core).
- Each rule is visible and was motivated by inspected data. Unsmoothed single frames left calls indistinguishable from noise maxima. Low-frequency transients near the band edge, not broadband clicks, made up most false detections. 2020 calls rarely go below 25 kHz (1.6% of 34,457 call minima).

## Tuning and validation
- Tuned against the 2020 Avisoft SASLab exports on 36 recordings (seed 1, 4 per litter × age band; listed in each detect run). All other recordings are held out from tuning.
- The 2020 exports are a reference, not ground truth: the analyst's selection rules are unknown, and faint calls were sometimes skipped. On the tuning set, agreement was recall 0.98 and precision 0.81. Of the 40 missed 2020 calls, 7 were merged with a neighbour and 33 were very short (mostly under 10 ms; three had zero length).
- True precision and recall come from the owner's blind hand-check (`usv detect-review` → `review.html` → `usv detect-score`), which samples detector-only, Avisoft-only, agreed and empty events by litter × age band from non-tuning recordings.

## Results (full run 2026-10-01, 574 recordings)
- 42,391 calls; 5,740 background windows (10 × 0.5 s per recording, ≥ 0.2 s from any call).
- Agreement with 2020 Avisoft on the 535 non-tuning recordings: recall 0.988, precision 0.806 (P1–P7 0.881, P8–P14 0.837, P15–P21 0.669). This matches the tuning set (recall 0.981, precision 0.809), so there is no sign of overfitting.
- External sanity check (mouseTube Rat 16, adult, two human annotators): recall 0.90 / 0.93, precision 0.65 / 0.58; the two humans agree at recall 0.84 and precision 0.97. Not used for tuning.
- Calls per recording (median by age): every day has calls; only 2 recordings have zero. Medians: P1–P4 26–61, a dip at P5–P7 (16, 19, 7.5), a peak at P11–P16 (75–153), then a decline to P21 (12). Recordings with fewer than 10 calls cluster at P5–P7 and P21. The extra (non-Avisoft) detections are most common at P15–P21, so the hand-check decides how much of the late-day signal is real.

## Known limits
Very short calls (< ~8 ms) are under-detected; adjacent calls closer than 8 ms are merged; calls entirely below 25 kHz are dropped. Harmonics are not separated from fundamentals (the contour follows the loudest band). The USCMed raw-audio benchmark is not yet run.
