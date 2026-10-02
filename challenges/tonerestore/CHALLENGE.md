Overview
Predict the annotated pitch-contour markers in a recorded utterance and locate them within its supplied tone-free IPA transcription. Each output marker gives a character position and a sequence of Chao pitch levels, restoring the tone information that the phonetic scaffold omits.
The speech comes from brief elicited words and phrases in an endangered language of Northeast India. The task is audio-conditioned structured sequence restoration: the supplied IPA gives segmental content, while the recording provides pitch evidence. Gold is the released phonetic transcription, not measured fundamental frequency, a probability of alternative pronunciations or a forced alignment in seconds. Some source strings place a marker before trailing consonants or combining diacritics; reproduce the annotated insertion point without moving it to an assumed syllable boundary.
The competition environment provides access to a single NVIDIA A10G GPU. The entire pipeline must finish within 1.5 hours, including data loading, training or adaptation, inference, decoding, validation, and submission generation.
Dataset
Supplied files are `train.csv`, `test.csv`, `sample_submission.csv`, and the WAVs referenced under `audio/`. There are 1838 labeled training rows and 403 prediction-input rows. Training contains 575 distinct tone-free forms from 34 recording-list prefixes; prediction rows contain 132 forms from 8 prefixes. Whole connected families are held together using recording-list prefix, repeated gloss, repeated tone-free form and identical PCM. There are only 5 training and 7 held-out connected groups; correlated repetitions are not independent speakers. Speaker identities are unavailable, so speaker-disjoint evaluation is not claimed.
Audio is mono 16-bit PCM WAV at 16,000 or 22,050 Hz. The observed duration range is 0.2231 to 4.7716 seconds. Tone-free strings have 1 to 28 Unicode code points. Reference sequences contain 1 to 9 markers. Utterances without marked tones are not supplied as negative examples; absence of a marker in those records cannot be interpreted as flat pitch.
Training columns are `id`, `audio_path`, `phones`, `tones`. Prediction-input columns are `id`, `audio_path`, `phones`. Sample submission columns are `id`, `tones`, in that order.

```
id: string; opaque identifier.
audio_path: string; WAV path relative to the supplied directory.
phones: string; tone-free IPA in Unicode NFD form, including original spaces and combining marks.
tones: JSON array of [offset, contour] pairs, ordered by increasing offset.
offset: integer in 1..128; insert a tone marker after this many Unicode code points in phones.
contour: string of 1..8 digits from 1,2,3,4,5, representing low through high Chao pitch levels.
```

Offsets count code points, not bytes, displayed glyphs, phonemes, samples or milliseconds. Combining diacritics and tie bars each count separately. Do not normalize the scaffold to NFC before counting. Consecutive Chao letters at one insertion point form a single contour string. The phonetic span associated with a marker runs from the previous marker's insertion point (or zero for the first) up to its offset; these are textual spans, not independently certified syllables. Trailing unmarked text remains in the scaffold.
The complete level vocabulary is 1=low, 2=mid-low, 3=mid, 4=mid-high, 5=high. Contours observed in these artifacts are listed with training and prediction counts below; multi-marker runs such as 5331 are preserved as released, not interpreted as probability mixtures.

```
31: 1945 / 442
3131: 59 / 9
3151: 3 / 0
3153: 39 / 3
51: 516 / 107
53: 2301 / 433
5331: 107 / 15
5351: 10 / 3
5353: 70 / 6
```

A training example uses audio `audio/case_00112893a175ac0af7f8a607.wav`:

```
phones: ʔinəɕikfɯ
tones: [[4,"3131"],[6,"53"],[9,"53"]]
code_points: ["ʔ", "i", "n", "ə", "ɕ", "i", "k", "f", "ɯ"]
```

Submission Format
Write `./working/submission.csv` with exactly `id`, `tones`, in this order, and one row per prediction-input ID. Rows may be reordered. Missing, extra, duplicate or invalid IDs, duplicate columns, reordered columns and unexpected columns are rejected. IDs must be nonempty strings of at most 80 characters without surrounding whitespace.
Use a JSON list of at most 32 markers, with strictly increasing integer offsets and the contour grammar above. Boolean, floating-point and quoted offsets are invalid. The whole cell is limited to 4,096 characters. The empty list is allowed as abstention and scores zero. Malformed, out-of-range, duplicated-position or oversized sequences score zero for that row. Offsets should refer to the supplied string; impossible positions cannot match reference markers.

```
id,tones
case_00112893a175ac0af7f8a607,"[[4,""3131""],[6,""53""],[9,""53""]]"
```

Evaluation
Minimum score: 0. Maximum score: 1. Higher is better. The anchored-contour score combines marker recovery, contour-sequence agreement and complete restoration. For each row, P and T are the predicted and reference ordered marker lists. A is twice the number of exact matching offset/contour pairs divided by the sum of their lengths. C is one minus the Levenshtein edit distance between the ordered contour-string lists, divided by the larger list length. Each entire contour string is one edit token; substitution, insertion and deletion cost one. E is one for identical complete marker lists and zero otherwise. The row score is (6A + 3C + E) / 10, and the final score is the arithmetic mean across rows. Reference lists are nonempty.

```
A = 2 * |set(P) intersect set(T)| / (|P| + |T|)
Cp = [contour for offset,contour in P]
Ct = [contour for offset,contour in T]
C = 1 - Levenshtein(Cp,Ct) / max(|Cp|,|Ct|)
E = 1 if P == T else 0
row_score = (6*A + 3*C + E) / 10
score = mean(row_score)
```

Exact anchored contours receive 60 percent because wrong attachment changes the transcription. Contour-sequence agreement receives 30 percent to give partial credit when pitch patterns are recovered but attached at wrong boundaries. Complete restoration receives 10 percent. Unsupported additions reduce marker precision and incur sequence-edit costs; simply adding many possible contours cannot represent uncertainty.
What Not To Use
Do not match recordings or transcriptions to external source releases, recover labels from source filenames or IDs, or adapt to evaluation feedback. Training on supplied examples and use of general pretrained acoustic models are permitted. The phonetic scaffold is intended evidence, but its lexical identity must not be used to retrieve source annotations.
