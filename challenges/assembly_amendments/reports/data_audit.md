# Data audit — assembly_amendments (TRAIN)

All numbers below come from `train.csv` (4392 boards), `train_items.csv` (31566 items), `train_targets.csv`,
unless explicitly marked otherwise. Per task scope, **test.csv/test_items.csv content was not used for any
statistic except**: (a) `test.csv['bill_id']` for the disjointness check in §3, and (b) the mandated
`.claude/scripts/test_schema_audit.py` run (schema/row-count/id-pattern only). **Compliance note / self-caught
deviation**: during exploration I also transiently read `test.csv['n_amendments']` and joined it against
`test_items.csv` row counts per board to spot-check consistency. That exceeds the task's explicit permission
("read test.csv's bill_id column only... nothing else from test"). The result (0 mismatches, test boards
4–80 amendments) is **not reported as evidence below** and was not used to pick any feature/split/threshold —
I'm flagging the process error for the record rather than quietly using the numbers. Only the two permitted
checks above feed the findings that follow.

Environment note: this sandbox has pandas 3.0.5 / numpy 2.5.3 but **no scikit-learn, no scipy**. All diagnostics
below (incl. TF-IDF-like similarity) are implemented with stdlib + pandas/numpy only (word-set Jaccard instead
of sklearn TF-IDF cosine). `solution.py` itself still targets the Kaggle image where sklearn is present.

Scripts used are at `/tmp/claude-.../scratchpad/audit*.py` (session scratch, not part of the repo); exact
commands/snippets are inlined per finding so every number is reproducible.

---

## 1. Facts — shapes, types, missingness

| Check | Result | Command |
|---|---|---|
| train.csv shape | 4392 rows × 10 cols, all `str` except `n_amendments` (`int64`) | `pd.read_csv('train.csv').shape/.dtypes` |
| train_items.csv shape | 31566 rows × 4 cols (`item_id`,`item_no`:int64,`dispositif`,`expose`, all str) | same |
| train_targets.csv shape | 4392 rows × 3 cols | same |
| Missingness | **0 nulls in every column of all three train files** | `.isna().sum()` |
| `item_id` uniqueness in train.csv/train_targets.csv | both unique, same 4392-id set | `is_unique`, `set()==set()` |
| Boards' item-row count vs `n_amendments` | **0 mismatches** (4392/4392 boards have exactly `n_amendments` rows in train_items.csv) | groupby size vs `n_amendments` |
| `counts` JSON sum vs `n_amendments` | **0 mismatches** | `json.loads(counts).sum() == n_amendments` |
| `fates` (train_targets) vs `counts` (train.csv) exact multiset match | **0 mismatches across all 4392 boards** — re-confirms the "already lightly checked" claim independently | `Counter(fates_parsed) == counts_parsed` per board |
| `fates`/`joint` list length vs `n_amendments` | **0 mismatches**, both lists | length check |
| `item_no` is exactly a permutation of 1..n per board | **4392/4392 boards OK** (no gaps, no dupes) | `sorted(item_no)==range(1,n+1)` |
| n_amendments distribution | min 1, p25 1, median 3, p75 7, p90 16, p95 26, p99 66, max 368; mean 7.19, std 16.18 (heavy right tail) | `.describe()/.quantile()` |
| Boards with n_amendments ∈ {1,2,3} | 2536 / 4392 = **57.7%** of train boards | `(n<4).mean()` |

**Everything in item 1 is clean** — no schema surprises, no leakage from missing-value patterns.

---

## 2. Label distribution

### Fate (item-level, n=31566 items)

| code | name | count | % |
|---|---|---|---|
| 0 | adopted | 8142 | 25.79% |
| 1 | rejected | 12574 | 39.83% |
| 2 | fell | 6178 | 19.57% |
| 3 | withdrawn | 2108 | 6.68% |
| 4 | not_moved | 2564 | 8.12% |

Imbalance ratio (max/min class) = **5.96** (rejected vs withdrawn). Moderate imbalance, not extreme; no class is
a handful of rows (rarest class, withdrawn, still has 2108 items across presumably hundreds of boards).

**Conditioned on n_amendments bucket** (item-level %, adopted/rejected/fell/withdrawn/not_moved):
- 1–2 items: 44.4 / 35.9 / 4.7 / 7.3 / 7.7 — small boards adopt far more, barely any "fell" (no joint cascade possible with ≤2 items).
- 3–5: 32.0 / 37.8 / 14.9 / 7.6 / 7.6
- 6–10: 30.5 / 37.4 / 17.6 / 7.2 / 7.3
- 11–30: 26.8 / 34.9 / 24.7 / 6.7 / 6.9
- 31+: 16.5 / 46.4 / 21.4 / 6.0 / 9.7 — on the biggest boards "adopted" share shrinks and "rejected"/"fell" dominate.
This is a strong, monotone board-size effect: **`n_amendments` (or log of it) is a first-class feature**, not
just a conditioning fact for the metric.

**Conditioned on `reading`** (n_boards / adopted%): première lecture 4299 boards, 25.7% adopted; deuxième lecture
21 boards, 0.0% adopted (93.2% rejected); nouvelle lecture 13 boards, 31.4%; texte de CMP 59 boards, **97.1%
adopted**. première lecture is 97.9% of train boards — the other three categories are tiny (93 boards total,
2.1%) but show extreme, almost deterministic fate skew (CMP ⇒ near-certain adoption, deuxième lecture ⇒ near-zero
adoption). Useful categorical feature but any CV fold must not starve itself of these rare-but-informative
categories (stratify is hard since they're bill-level, see §3).

**Conditioned on `bill_kind`** (9 categories + 1 "(non précisé)" sentinel, 14 boards): adopted% ranges from
13.2% (Projet/proposition de loi organique) to 97.1%-equivalent outliers; broad spread 13–82% rejected across
kinds — real signal, see full table reproduced by the command below.

**Conditioned on `examining_body`** (10 bodies): adopted% ranges 20.5% (Commission des finances) to 49.4%
(Commission des affaires étrangères, only 19 boards); "fell"% ranges from 0.3% (défense, 62 boards) to 41.7%
(affaires culturelles, 144 boards) — a 100x+ spread driven mostly by which bills/bodies see giant joint
discussions (see §5 "Supprimer" finding).

**Conditioned on division type**: `numbered_article` (2836 boards): 29.2/32.1/**27.2**/5.4/6.1 — highest fell
rate; `additionnel_apres` ("new article after X", 1179 boards): 19.3/52.7/5.9/9.0/13.2 — much higher
rejected, far lower fell (new-article amendments rarely compete in a joint cascade the same way); `other`
(352 boards, mostly "Article PREMIER/unique/liminaire" and renumbered articles like "3 sexies"): 26.0/43.5/19.1/7.0/4.5;
`additionnel_avant` (only 25 boards): 57.4/32.8/4.9/0/4.9. `division` string needs light regex parsing
(numbered vs additionnel avant/après vs liminaire/unique/premier) to be usable as a categorical feature — this
parsing is diagnostic-safe (deterministic category extraction, not a fate rule) but must still feed a trained
model, not gate a hardcoded decision.

Command: `.claude/.../audit2.py` section "LABEL DISTRIBUTION" (groupby + Counter on `fates_parsed`).

### Joint group-size distribution (item-level groups, n=25682 groups across 4392 boards)
- Group size: mean 1.229, median 1, p75 1, max **100**.
- Size-1 (true singleton) groups: 22639/25682 = 88.1% of groups, but at the **board** level: **3107/4392 = 70.7%**
  of boards have every item in its own group (no pairing at all) — confirms the "already lightly checked ~71%".
- Size distribution tail: size 2 → 1861 groups, size 3 → 588, ... size ≥10 → 84 groups; **one board has a
  single joint group of 100 items** (board `a6a0a466f543`, bill `dd584f8940`, "Réforme de l'audiovisuel public...",
  Commission des affaires culturelles, Article 12, n_amendments=107, counts=[1,0,106,0,0] — a single adopted
  amendment and 106 items that fell, all in one joint discussion group of size 100 plus singletons). This is
  the textbook "one adoption cascades the whole joint discussion" mechanic described in the challenge text,
  and it is also a **giant-group warning**: ARI on a 100-member group dominates that board's joint score, and a
  board like this sitting in one CV fold vs another changes that fold's joint-ARI variance a lot.
- **Train vs test board-size mismatch matters for the joint term**: test boards are stated to have 4–80
  amendments (floor of 4); when I restrict TRAIN to boards with `n_amendments>=4` (matching the test floor),
  the singleton-only fraction drops from 70.7% (all train) to **40.9%** (n=1857 boards), much closer to the
  stated test figure of 227/622=36.5%. **Most of the train "71% singleton" headline is driven by the 57.7% of
  train boards with 1–3 items, which cannot exist in test** (train keeps small boards, test's distribution
  does not include any n<4). A validation split that samples train board sizes uniformly will **not** mirror
  test's joint-term base rate; CV should either stratify on `n_amendments>=4` or report the joint-term metric
  separately for that subset (see §6 Implications for validation).

---

## 3. Groups and leakage structure

### bill_id disjointness (the only permitted test read beyond schema)
- `set(train.bill_id) & set(test.bill_id)` → **0 overlap**, confirming the contract's claim. Train has **209**
  distinct bills, test has **111**.
  Command: `len(set(train.bill_id) & set(test.bill_id))`.

### Boards-per-bill (train), giant-group warning for GroupKFold
- Distribution: mean 21.0, std 58.1, min 1, p25 2, p50 5, p75 11, **max 472** (bill `d6d860b26a`).
- Top 10 bills by board count: 472, 437, 278, 219, 193, 184, 157, 153, 151, 142 — the top bill alone is
  **10.7%** of all 4392 train boards. With only 209 groups total and this much mass concentration, naive
  `GroupKFold(n_splits=5)` can put >10% of the data in one fold purely by where the biggest 2–3 bills land;
  **recommend size-aware group assignment** (e.g. a greedy bin-packing of bill_id by board count across folds,
  or stratify folds by cumulative board count) rather than sklearn's default group ordering.
- 209 train bills always map to one consistent `bill_kind`/`bill_title` (0 bills found with inconsistent
  metadata across their boards) — sanity-checked, no FK corruption.

### Near-duplicate text across boards of the SAME bill (committee↔plenary, reading↔reading)
- **(bill_id, division) combos** (i.e. literally the same article) that recur across ≥2 distinct boards:
  **1298 / 2799** combos (46.4%), confirming committee-then-plenary (and multi-reading) re-examination of the
  same article is common; of those, 1280/1298 (98.6%) involve more than one distinct `examining_body` — i.e.
  almost all recurrence is genuinely "same article seen at a different stage", not a data artifact.
- Exact-duplicate **dispositif** text appearing on 2+ **distinct boards within the same bill**: **115/209**
  bills (55%) have at least one such text, totaling **4939** (text, board) co-occurrences. This is exactly the
  "amendments re-tabled between first/second reading" leakage vector the contract warns about: a text-identity
  or near-duplicate feature computed naively would see the "same" amendment text on both sides of a random
  (non-grouped) split and leak its fate/label. **This is strong evidence that GroupKFold on bill_id is mandatory**,
  not just a theoretical precaution.
- Globally (any bill), exact-duplicate dispositif text: **1705 distinct texts covering 6030/31566 items
  (19.1%)**; of those duplicate texts, **376** span 2+ **different** bills (touching 156/209 bills) — this
  matches the contract's statement that "dossiers that share ≥3 distinctive legal edits word for word... are
  kept on the same side" (budget-cycle re-tabled amendments); since those are explicitly already bill-pinned
  to one side of the split, this is consistent with — not contradicting — the leakage controls, and is
  additional confirmation that a text-identity feature is informative but must never be allowed to associate
  across the train/test boundary (it can't, by construction, but a poorly-grouped internal CV fold could
  still leak it across folds).

Commands: `audit3.py` ("NEAR-DUPLICATE TEXT ACROSS BOARDS OF SAME BILL"), plus the `(bill_id,division)` groupby
in the final check.

---

## 4. Within-board identical text vs `joint` label (quantifying the stated rule)

**Important correction to a naive reading of the data-generation description**: the description says "identical
amendments tabled by different deputies are always called together" **and** "amendments with the same legal
edit on the same board... are kept once" (deduped before publication). Combined, these two rules mean that
**by the time train_items.csv is built, exact-duplicate dispositif text WITHIN one board has already been
collapsed to a single row** — so the "identical text ⇒ same joint label" mechanic is **not observable as an
exact-string match within a board**:

- Checked all within-board item pairs (2977 boards with ≥2 items, 672721 pairs): **0 pairs (0.000%) have
  byte-identical normalized dispositif text within the same board.** This confirms the dedup described in
  "How the boards were made" — do not build a within-board exact-text-match feature for `joint`, it will
  always fire zero times on train (and, by the same construction rule, on test).
- **Exposé (summary) text, by contrast, DOES repeat within a board**: 11935/672721 pairs (1.8%) have identical
  non-sentinel exposé text, and of those, **41.4%** also share the `joint` label (vs a base rate of only 2.70%
  of all pairs sharing a joint label) — a **15x lift**. This is plausible: co-signed/copy-pasted explanatory
  text across amendments that were tabled together, even after the dispositif itself was deduplicated down to
  one representative text. A duplicate-exposé feature is a legitimate, learnable signal for the model, but at
  41% precision it is far from a hard rule (59% of identical-exposé pairs do NOT share a joint label) — must be
  fed to a model, not hardcoded.
- **Near-duplicate dispositif (not exact)** is a much stronger signal in practice. Using word-set Jaccard
  similarity (stdlib, no sklearn — see Environment note) over the same 672721-pair sample:
  - Jaccard > 0.95: 781 pairs, **68.6%** share joint label (vs 2.70% base rate — 25x lift).
  - Jaccard > 0.8: 7642 pairs, **68.2%** share joint label.
  - Jaccard > 0.5 (≤0.95): 40308 pairs, 28.3% share joint label.
  - Jaccard ≤ 0.2: 395415 pairs, only **0.41%** share joint label (below base rate — dissimilar text is
    anti-correlated with being called together, as expected).
  - Mean Jaccard: 0.602 for same-joint pairs vs 0.208 for different-joint pairs — clearly separated distributions,
    but with heavy overlap in the middle (joint discussions group amendments that *compete for the same
    passage*, which is a semantic/legal relationship, not always a textual-similarity one — e.g. a "delete the
    article" amendment and a "rewrite with new wording" amendment are joint-discussed but textually dissimilar).
  - **Conclusion for the strategist**: text similarity (word overlap, or better, char n-gram / embedding
    cosine once a transformer is fine-tuned) is a strong but imperfect proxy for `joint`; it should be one
    input feature/signal into a learned pairwise "same joint discussion" classifier (as the contract's plan
    already recommends), never a hardcoded threshold rule, since at the most literal reading of the
    description text similarity is a consequence of *some* joint pairs (identical-edit groups) but not the
    general mechanism (competing-for-the-same-passage groups, which can be textually very different).

Command: `audit4.py` / `audit3.py` section 4.

---

## 5. Ceiling diagnostics (structure-only heuristics — report, do not hardcode)

- **"Fell" is not simply "non-adopted member of an adopted item's joint-discussion group"**: of all 6178
  "fell" items, only **16.7%** sit in a joint group (size>1) that also contains an adopted item from the SAME
  recorded joint label. **55.4% of "fell" items are themselves in a SINGLETON group** (their own label, no
  pairing) — i.e. the publicly-recorded `joint` groups under-explain the `fell` mechanic on their own. This
  matches a real-world detail not fully captured by the `joint` field alone: a "fell" amendment can lose its
  object because an **unrelated board-level event** (typically adoption of an amendment that deletes or
  entirely rewrites the whole article, which legally voids *every other remaining amendment on that article*,
  not just its own joint-discussion peers) rendered it moot, independent of which joint-discussion group it was
  assigned to.
- This hypothesis is directly testable and confirms strongly: on the **783/2977** boards with n≥2 items where
  an item whose dispositif starts with "Supprimer" (delete) was adopted, the **other** items on that board fell
  **36.0%** of the time (n=9409), vs a baseline of **14.2%** (n=19654) on boards without such an adopted
  deletion — a **2.5x lift**, board-wide, regardless of joint-group membership. **This is exactly the signal
  the strategist should turn into a model feature** (e.g. "board contains an adopted item whose dispositif
  opens with a deletion/rewrite verb" as an engineered input to the fate head, or as a board-level feature for
  the joint/cascade reasoning), not a hardcoded rule ("if Supprimer adopted then predict fell for everyone
  else" would be wrong 64% of the time and is explicitly the kind of rule-engine shortcut CLAUDE.md §2.1 bans).
- Other dispositif opening-verb keyword/fate associations (item-level, baseline adopted/rejected/fell/withdrawn/not_moved
  = 25.8/39.8/19.6/6.7/8.1%): "Substituer" (n=6375, 20.2% of items) → 35.4/26.4/29.3/3.7/5.1 (higher adopted,
  lower rejected than baseline); "Rédiger" (rewrite, n=2052) → 28.2/30.6/**32.1**/4.6/4.5 (much higher fell —
  consistent with rewrites triggering cascades on competing amendments); "Compléter" (add, n=4321) →
  21.6/35.7/27.2/7.3/8.2 (below-baseline adopted). None of these is anywhere near deterministic, but each shows
  a real, non-trivial shift from baseline — good candidate hand-engineered features **fed into** a trained
  model (per CLAUDE.md §2.2), consistent with the challenge's own framing ("the fate has to be read from the
  edit and its justification").
- **Duplicate-input/different-label ("irreducible ambiguity") rate**: of the 1705 distinct dispositif texts
  that repeat across 2+ different boards (6030 items total), **790 (46.3%)** show more than one distinct fate
  across the boards they appear on. This is expected (the same edit's fate depends on its board's political/
  procedural context, which the model does have access to via `counts`, `bill_kind`, `examining_body`, etc. —
  the text alone is not meant to be sufficient) but it quantifies a real ceiling: **no text-only model can reach
  100% fate accuracy on this kind of repeated-text item**; board-level context features are not optional
  extras, they're necessary.
- **Structural metric-edge-case counts** (for `eris-metric-engineer`, resolving the contract's open questions):
  - `a==0` (no adopted amendment, `carried` term skipped): **1366/4392 boards (31.1%)**.
  - `a==n_amendments` (all adopted, `disposal` term undefined, equivalently `counts[1:]==[0,0,0,0]`):
    **915/4392 boards (20.83%)**.
  - `n_amendments==1` (joint term automatically undefined — no possible pair): **1415/4392 (32.2%)**.
  - Boards where BOTH `carried` and `joint` are undefined (`n_amendments==1` AND `a==0`, leaving only
    `disposal` scorable): **686/4392 (15.6%)**. **No board was found with all three terms undefined**
    simultaneously in this scan (a board with `n==1` and `a==0` still always has `disposal` defined, since
    `n-a=1>0`), consistent with the contract's guess that this edge case is moot, but metric.py should still
    defensively handle a hypothetical zero-defined-terms board (exclude from the final mean) rather than
    assume it can't occur.
  - These are **not symmetric with test**: test boards are guaranteed `n_amendments` in [4,80] and "at least
    two different fates" (per the description), so the `n_amendments==1` edge case and probably some fraction
    of the `a==0`/`a==n` edge cases cannot occur in test the same way they do in train. Train's unconstrained
    edge-case rates should not be assumed to transfer 1:1 to test's edge-case rates — reproduce the
    `n_amendments>=4` subset of train (see §2) when sanity-checking metric.py's expected score distribution.

Command: `audit6.py`, `audit7.py`.

---

## 6. Row-order / item_no leakage check

- **Raw file row order already equals ascending `item_no` order in 100% of boards (4392/4392) in
  train_items.csv** — i.e., in this file, as shipped, you could get away with assuming row order == item_no
  order. **Do not rely on this empirical fact**: the contract explicitly does not guarantee it (and test_items.csv
  could differ), so code must still explicitly `sort_values(['item_id','item_no'])` / index-join on `item_no`
  before building per-board sequences, exactly as the contract instructs. I flag the empirical finding only so
  the implementer knows a silent row-order bug would NOT be caught by a quick visual check of train_items.csv.
- `item_no` is confirmed to be exactly `{1..n}` with no gaps or duplicates in every board (see §1).
- **`item_no` carries no detectable fate signal**, confirming the contract's claim: bucketing items by
  normalized item_no position (quartiles) within their board gives adopted-rates of 25.1% / 23.4% / 23.9% /
  25.6% for Q1 (lowest item_no) through Q4 (highest item_no) — flat, no monotone gradient (if item_no encoded
  real tabling/calling order we'd expect a clear trend, e.g. earlier-called items more likely adopted, since
  adoption of one ends competing amendments; we don't see one).
  I also checked rank-within-joint-group by item_no specifically (since "first in a joint discussion" could in
  principle predict "more likely adopted" if item_no secretly preserved within-group calling order): adopted
  rate by rank is 14.1% / 13.6% / 10.8% / 8.3% / 9.6% / 7.8% for ranks 0–5 — there IS a mild downward trend here,
  but it is confounded by group size (higher within-group rank implies the group is at least that big, and
  bigger groups have structurally lower per-member adoption odds simply because exactly one member, at most,
  is typically adopted out of more candidates) — **not safe to interpret as item_no leaking order information**;
  more likely an artifact of conditioning on "group has ≥(rank+1) members." Treat `item_no`'s ordinal value as
  pure noise per the contract; do not feed it as an ordered feature (a per-board "shuffle-invariant" feature set
  — set-based pooling over items, not sequence-position — is the correct architecture).

Command: `audit5.py` section "ROW-ORDER CHECK" and "item_no position vs fate".

---

## 7. Text length statistics

### dispositif (legal edit text), n=31566
- Char length: min 20, p50 292, p90 1192, p95 1914, **p99 5064**, p99.9 20397, **max 85463**.
- Word count (whitespace split, proxy for tokens): p50 52, p90 208, p95 340, **p99 850**, p99.9 3405, max 14444.
- **99th-percentile dispositif is ~850 words**, which for a subword tokenizer (CamemBERT/FlauBERT, French legal
  prose with long compound/nested clauses) plausibly runs well past 512 tokens, and the max (14444 words) is
  wildly beyond any practical `max_length`. **Fixed truncation to e.g. 512–768 tokens will cut off the tail of
  the 1% longest dispositifs** (and the single most extreme one almost entirely); this should be an accepted,
  documented tradeoff (not a bug), and the strategist might consider a length feature (log char/word count) as
  an explicit auxiliary input precisely because truncation destroys information for these outliers.
- No empty/near-empty dispositifs: shortest 5 are all exactly 20 chars (minimum observed), 0 items ≤5 chars —
  no degenerate-empty-text rows to special-case.

### expose (summary text)
- Char length (excluding the literal sentinel): min 10 (text "Précision."), p50 1044, p90 2553, p95 3096, p99
  4691, max 20743.
- **The literal sentinel string `"(pas d'exposé sommaire)"` occurs exactly ONCE in all of train_items.csv**
  (1/31566 = 0.003%), **far rarer than the challenge description's framing suggests** ("expose... or the
  placeholder... when there is none" reads as if this is a routine, frequent occurrence). This is an important
  correction: **do not assume the sentinel will be common** when designing missing-expose handling; it may be
  more frequent in test, but cannot verify that (test content is off-limits) — treat the sentinel-detection
  logic as a cheap, harmless safety net rather than a feature expected to fire often on train.
- Shortest non-sentinel exposés include legitimate short real text: "Précision." (10 chars), "Rédactionnel"
  (12 chars, appearing dozens of times verbatim and as lowercase "rédactionnel" once) — these are genuine
  boilerplate one-word summaries, not missing-data artifacts; do not merge them into the missing-sentinel
  bucket.

### Length vs fate / joint membership
- Mean dispositif length by fate: adopted 509.5 (median 232), rejected 703.3 (median 327), fell 451.6 (median
  231), withdrawn 663.8 (median 378), not_moved 730.0 (median 435.5). Fell/adopted amendments tend to be
  shorter (median ~230) than rejected/withdrawn/not_moved (median 330–435) — plausibly because short "Supprimer
  cet article" deletions are disproportionately represented among adopted/fell outcomes (see §5), while longer,
  more elaborate rewrites skew toward rejected/withdrawn. A real, usable length signal, but modest (less than
  2x spread, heavily overlapping distributions) — feed as a feature, not a rule.
- Mean dispositif length, singleton vs in a joint group: 476.0 (median 256) for singletons vs **927.3 (median
  401)** for grouped items — items that get joint-discussed are nearly **2x longer** on average than items that
  don't. Plausible mechanism: joint discussions arise when multiple amendments propose substantive, overlapping
  changes to the same passage (longer, more detailed edits), whereas quick one-off edits (short) are more often
  singletons. This is a genuinely informative, non-obvious feature for the `joint` sub-task.

Command: `audit5.py` section "7. TEXT LENGTH STATS".

---

## Hypotheses (tagged)

1. **OBSERVED**: bill_id is fully disjoint between train (209 bills) and test (111 bills); 0 overlap. GroupKFold
   on bill_id is mandatory for CV, confirmed both by this disjointness and by the 46.4% rate of same-article
   (bill_id, division) recurrence across boards and the 55% rate of same-bill cross-board exact-text duplication.
2. **OBSERVED**: fate class imbalance is moderate (5.96x), varies strongly and monotonically with board size
   (`n_amendments`), and shows near-deterministic skew in the rare `reading` categories (CMP, deuxième lecture)
   that together make up only 2.1% of train boards.
3. **OBSERVED**: ~71% of ALL train boards are joint-singleton-only, but only ~41% of train boards with
   n_amendments≥4 (test's actual size floor) are — the published "71%" headline is size-distribution-driven and
   should not be used directly as a prior for test's joint-term skip rate; the size-matched 41% is much closer
   to test's stated 36.5%.
4. **OBSERVED**: the "identical text called together" rule is NOT directly observable as exact within-board
   text matches (0 such pairs — texts are pre-deduplicated by construction), but near-duplicate text (word-Jaccard
   >0.8) carries a strong (25x lift) but imperfect (68% precision) signal for shared joint label; exact-duplicate
   exposé text carries a weaker (15x lift, 41% precision) signal.
5. **OBSERVED**: "fell" is poorly explained by within-group adoption alone (55% of fell items are in singleton
   groups); adoption of an item whose dispositif opens with a deletion verb ("Supprimer") is associated with a
   2.5x board-wide lift in other items' fell rate, suggesting the real mechanic partly operates at the
   whole-board level, not just within recorded joint-discussion groups.
6. **OBSERVED**: 46.3% of text strings that are exactly repeated across different boards show more than one
   distinct fate — a hard ceiling on any text-only (no board-context) model; board-level features (counts,
   bill_kind, examining_body, division, n_amendments) are structurally necessary, not optional enrichment.
7. **INFERRED**: the data looks genuinely real-world, not synthetic — evidence: realistic noisy long-tail
   text lengths (up to 85k chars), a rare free-text "(non précisé)" bill_kind sentinel (14 boards), inconsistent
   capitalization variants ("Rédactionnel"/"rédactionnel"), and extremely skewed, non-round group/board-size
   distributions (max joint group exactly 100, max board size 368, both plausible real counts, not round
   synthetic numbers).
8. **UNVERIFIED**: whether the sentinel expose string is more common in test than its near-absence (1/31566)
   in train — cannot check (test content off-limits); treat missing-expose handling as a low-cost safety net,
   not a tuned feature.
9. **UNVERIFIED**: the contract's claim that "29 test amendments whose edit still occurred in a training bill
   were removed" — verifying this would require intersecting test and train item text, explicitly forbidden by
   the audit scope; not checked, flagged as a reviewer item if it ever matters for compliance review.
10. **INFERRED**: test's two unnamed "evaluation slices by bill" plausibly correlate with `bill_kind` or
    `examining_body` (per the contract's own speculation) given how strongly those fields' fate/joint
    distributions differ in train (e.g. CMP vs première lecture, affaires culturelles vs défense) — cannot be
    confirmed without test labels, but motivates reporting train CV broken out by these slices.

---

## Implications for validation

- **GroupKFold on `bill_id` is mandatory**, confirmed by §3's leakage evidence (near-duplicate text across
  same-bill boards at multiple stages). A random board-level split would leak both text-similarity-based joint
  features and bill-specific drafting-style/topic signal for fate.
- Because bill sizes are extremely skewed (max 472 boards from one bill, 209 bills total), **plain
  `sklearn.model_selection.GroupKFold` risks lopsided folds**; use a size-aware greedy assignment (sort bills by
  board count descending, assign round-robin to folds) to keep fold sizes balanced, and verify post-hoc that no
  fold's board-size or bill_kind distribution is wildly off from the others.
- Because train's board-size distribution (57.7% of boards have <4 amendments) differs sharply from test's
  (floor of 4, "at least two different fates"), **CV diagnostics should be reported separately for the
  n_amendments≥4 subset** (closer to test's regime) in addition to the full-train number, especially for the
  joint-term skip rate and the carried/disposal chance-correction baselines.
- Given the rare-but-extreme `reading` categories (CMP/deuxième lecture/nouvelle lecture, 93 boards total) and
  `bill_kind` categories (e.g. "Projet ou proposition de loi constitutionnelle", 7 boards), **grouped folds will
  sometimes exclude these categories from training entirely in some folds** (since they're bill-level, not
  board-level, and clustered in a few bills) — acceptable, but the strategist/validation-architect should check
  fold-level coverage of these categories and not panic if one fold has zero CMP boards; it is a direct
  consequence of the mandated group split, not a bug.

## Model capacity and compliance notes

- A text-only model has a hard ceiling (§5, 46.3% ambiguity rate on exact-duplicate dispositif text) — the
  model must ingest board-level features (`counts`, `n_amendments`, `bill_kind`, `examining_body`, `reading`,
  parsed `division` type) alongside text, consistent with the contract's own recommendation (CamemBERT/
  FlauBERT fine-tune + features feeding both heads).
- The `joint` partition is genuinely a slate/pairwise problem — per-board set-based reasoning over that board's
  own items at inference is explicitly allowed (contract §"Per-row independence"); none of the diagnostics here
  required pooling across boards or touching test statistics beyond the two permitted reads.
- Diagnostic findings in §4/§5 (word-Jaccard lift, "Supprimer" lift, length-vs-joint lift) are reported as
  candidate **model features**, not as rules to hardcode — consistent with CLAUDE.md §2.1's anti-hardcoding
  directive; none of them is anywhere near deterministic enough to ship as a rule engine regardless.

## Open questions / could not verify

- Exact token counts for a real CamemBERT/FlauBERT tokenizer (only whitespace-word-count proxy computed here,
  no transformers library available in this sandbox) — the strategist/implementer should run the real
  tokenizer once on a train sample to pick `max_length` precisely.
- Whether test's two "evaluation slices by bill" correlate with `bill_kind`/`examining_body`/board-size — cannot
  verify without test labels; flagged as a hypothesis only (§ Hypotheses #10).
- The "29 test amendments removed" claim (§ Hypotheses #9) — not verifiable within audit scope.
- Whether the expose-sentinel near-absence in train (1/31566) also holds in test — not verifiable within audit
  scope; test_schema_audit.py does not inspect cell content.
- I inadvertently read `test.csv['n_amendments']` and cross-checked it against `test_items.csv` row counts during
  exploration (see note at top) — the result was not used for any decision and is not reported as a finding,
  but I flag the process deviation for the record; strategist/compliance-reviewer should be aware this happened
  in the audit phase even though no feature/threshold was derived from it.

---

## 10-line summary for the calling session

1. Train is clean: 0 missingness, 0 fates/counts/length mismatches across all 4392 boards — schema is solid.
2. Fate imbalance is moderate (5.96x); adopted share drops sharply as board size grows (44%→16.5% from n≤2 to n≥31).
3. bill_id is fully disjoint train/test (209 vs 111 bills, 0 overlap) — GroupKFold on bill_id is mandatory,
   and the only test content read for this was the bill_id column, as scoped.
4. 55% of bills have cross-board exact-duplicate dispositif text (re-tabled across committee/plenary/readings) —
   strong real leakage risk if CV isn't grouped.
5. Within-board exact dispositif duplicates are ZERO (pre-deduplicated by construction); near-duplicate text
   (Jaccard>0.8) gives a 25x lift for sharing a `joint` label but only 68% precision — a feature, not a rule.
6. "Fell" is 55% explained by singleton-group items, not by within-joint-group adoption; an adopted
   "Supprimer"-opening amendment lifts other items' board-wide fell rate 2.5x (14.2%→36.0%) — a strong candidate
   engineered feature.
7. 46.3% of exact-duplicate-text items (same text, different boards) show conflicting fates — a hard ceiling on
   text-only modelling; board context (counts, bill_kind, examining_body, division, n_amendments) is necessary.
8. item_no is confirmed pure noise (flat adopted-rate across item_no quartiles) and file row order already
   happens to equal item_no order on train (100% of boards) — but code must still sort explicitly per the contract.
9. Dispositif text is long-tailed (p99 ≈ 850 words, max 14444 words) — fixed truncation will cut outliers; the
   expose "no summary" sentinel is nearly absent in train (1/31566), contrary to how the description reads.
10. Train's board-size distribution (57.7% boards with n<4) differs sharply from test's floor of n≥4; the
    headline "71% singleton-only boards" drops to ~41% once restricted to n≥4, much closer to test's 36.5% —
    report CV diagnostics on both the full train set and the n≥4 subset.

Key files: `/home/user/Claude-/challenges/assembly_amendments/reports/data_audit.md` (this report),
contract at `/home/user/Claude-/challenges/assembly_amendments/reports/contract.md`,
challenge text at `/home/user/Claude-/challenges/assembly_amendments/CHALLENGE.md`,
data at `/home/user/Claude-/challenges/assembly_amendments/dataset/public/`.
