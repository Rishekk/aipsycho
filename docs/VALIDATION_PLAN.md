# Validation Plan

**Status:** draft for supervisor/team review. Thresholds below are *proposals* to agree on
**before** any data is collected, then committed unchanged (so results can't move the goalposts).

## 1. Question this plan answers

The current system maps acted *emotion* labels onto five "psychological state" names
(`EMOTION_TO_STATE`). The reported numbers (face 67% on FER-2013, speech 55% on RAVDESS)
only show that the models recognise **posed expressions in the datasets they were trained on**.
They say nothing about whether the output tracks a real change in a person's stress, anxiety or
frustration. This plan tests exactly that, in three stages that each end in a go/no-go decision.

| # | Hypothesis | Where it is tested |
|---|-----------|--------------------|
| H1 | Each agent beats trivial baselines on **unseen subjects** | Stage A |
| H2 | Each agent still works on a **different corpus** (not just a held-out split of its training set) | Stage A |
| H3 | Agent output moves with **independently measured** stress, within a person | Stage B |
| H4 | Fusion beats the best single modality | Stage C |

## 2. Stage A - offline, subject-independent (about 1 week)

Already done on this branch: subject-independent RAVDESS split (actors 1-16 / 17-20 / 21-24),
separate val and test splits for face data, `evaluate_model.py` reports on the *test* split.

To do:

1. **Retrain and re-evaluate** face and speech with the new splits. Replace the old
   `results/metrics/*.json` (they come from the leaky split). Record seed, split, git commit
   and the exact command alongside each result.
2. **Baselines** on the same test set: majority class, class-prior sampling, and (speech) a
   logistic regression on simple prosody features (mean/variance of pitch, energy, speaking rate).
   A neural model that cannot beat these is not learning affect.
3. **Cross-corpus tests** (train on one corpus, test on another, no fine-tuning):
   - Face: train FER-2013 -> test RAF-DB (and AffectNet if obtainable).
   - Speech: train RAVDESS -> test CREMA-D and/or IEMOCAP.
4. **Metrics:** macro-F1 and per-class recall (not accuracy alone; classes are imbalanced),
   confusion matrix, expected calibration error, and 95% bootstrap CIs resampled **by subject**,
   not by clip.
5. **Body agent:** has no evaluation at all and `record_body_data.py` stores no subject id.
   Either add subject ids and evaluate leave-subjects-out, or drop the body agent from claims.

**Stage A gate (proposed):** an agent proceeds only if its cross-corpus macro-F1 is clearly above
chance (0.20) *and* above the baseline in step 2, with the CI lower bound above the baseline.
Proposed bar: cross-corpus macro-F1 >= 0.35. An agent that fails is reported as a negative
result and excluded from Stage C.

## 3. Stage B - own data with independent ground truth (about 2 weeks)

Emotion labels from the face or voice cannot validate themselves. Ground truth must come from
somewhere else.

**Participants:** 20-30 adults, recruited with informed consent. Get institutional/supervisor
ethics sign-off first. This is a pilot; with this n expect wide CIs and report them.

**Protocol (about 15 min each, fixed order, same room and device for everyone):**

| Block | Task | Expected arousal |
|-------|------|------------------|
| 1 | Seated rest, read a neutral passage aloud | baseline |
| 2 | Timed mental arithmetic with mild time pressure (TSST-style, no deception) | stress |
| 3 | Impromptu 2-minute talk to the camera | stress / anxiety |
| 4 | Watch a pleasant video clip, then describe it | recovery / positive |

**Ground truth (any one is enough; two is better):**
- Self-report after each block: a short state-anxiety scale (STAI short form) plus a 0-100 "how
  stressed do you feel right now" slider.
- Heart rate / HRV from a chest strap or smartwatch, time-synchronised with the recording.
- Do **not** use outside annotators guessing the emotion from the video: that re-creates the
  original problem (labels derived from the same cues).

**Recording:** 30 fps video + audio with timestamps. Store only what you need; extract features
and delete raw video/audio after processing unless participants separately consented to keep it.

**Analysis (leave-subjects-out; all comparisons within person):**
1. Per participant, compare the model's mean "stress-like" score (define in advance, e.g.
   P(stressed) + P(anxious)) in blocks 2-3 versus block 1. Report the within-person AUC and the
   paired effect size across participants.
2. Correlate model score with self-report and with HR/HRV (Spearman), within person.
3. Compare against cheap baselines that need no emotion model: speaking rate, loudness, pitch
   variance, head-motion energy, HR alone. If a trivial baseline matches the model, the model
   adds nothing.
4. Report results per modality and, where sample size allows, by gender and age band. Do not
   claim fairness from groups with only a handful of people.

**Stage B gate (proposed):** the best modality shows a within-person AUC (stress blocks vs
baseline) >= 0.70 with a 95% CI lower bound > 0.5, **and** beats the best trivial baseline.
If nothing passes, stop building a product on this signal and write up the negative result.

## 4. Stage C - does fusion help? (about 1 week)

Needs the Stage B data (the repo has no labelled multimodal dataset). Compare, on identical
leave-subjects-out folds: each modality alone, each pair, all three with equal-weight late fusion,
and the attention fusion model. Use a paired test across participants (Wilcoxon signed-rank or
paired bootstrap).

**Stage C gate (proposed):** keep fusion only if it beats the best single modality by >= 0.03 AUC
with a CI excluding zero. Otherwise ship the best single modality; fewer sensors is also less
intrusive.

## 5. What each outcome means for the project

| Result | Direction |
|--------|-----------|
| Voice passes, face/body do not | Voice-first product (single microphone; simplest, most defensible) |
| Face and/or body pass | Self-use "delivery coach" with opt-in and cue-level feedback |
| Nothing beats trivial baselines | Treat as an engineering + negative-result report; do not market as stress/anxiety detection |

## 6. Threats to validity (keep in the final report)

- **Acted vs spontaneous:** RAVDESS/FER-2013 are posed; real expression is subtler and varies
  by person and culture.
- **Induced stress is not clinical anxiety.** Findings do not transfer to patients or to
  high-stakes settings.
- **Small pilot:** report CIs; avoid subgroup claims you cannot support.
- **Device and room effects:** one webcam/microphone setup. Test at least one other.
- **Leakage checklist before reporting any number:** no subject in two splits; test split never
  used for model selection, threshold tuning or fusion weights; preprocessing statistics fit on
  train only; duplicate/near-duplicate clips removed across splits.

## 7. Ethics and legal checklist

- Written informed consent covering what is recorded, how long it is kept and who sees it; the
  right to withdraw and have data deleted.
- Check obligations under India's DPDP Act for biometric/personal data (and any other
  jurisdiction where participants live). Seek advice; this plan is not legal advice.
- Never deploy to evaluate employees, students or job candidates. In the EU, inferring emotions
  in workplaces and educational institutions is prohibited by the AI Act, with narrow medical and
  safety exceptions; other emotion-recognition uses are high-risk.
- Do not describe outputs as stress/anxiety detection, and do not give health or coping advice,
  unless Stage B supports it and a qualified professional has reviewed the claim.

## 8. Work still to build

- `scripts/evaluate_model.py`: add `--split`, bootstrap-by-subject CIs, calibration, JSON with
  seed/commit/split metadata.
- CREMA-D / IEMOCAP loaders for cross-corpus speech tests; RAF-DB wiring for face.
- Baseline models (majority, prosody logistic regression).
- A small recording tool for the Stage B protocol (synchronised video, audio, timestamps, and
  self-report capture), and an analysis notebook for the within-person statistics.
- Subject ids in `record_body_data.py`.

## 9. Timeline

| Week | Work |
|------|------|
| 1 | Stage A: retrain, baselines, cross-corpus; agree and commit the gates above |
| 2 | Ethics sign-off; build recording tool; pilot with 2-3 people |
| 3 | Data collection (Stage B) |
| 4 | Stage B/C analysis, go/no-go decision, short written report |
