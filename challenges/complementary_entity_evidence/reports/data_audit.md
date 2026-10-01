# Data audit: complementary_entity_evidence (train only; test: schema/row count/id pattern only)

Commands: ad-hoc pandas diagnostics on `dataset/public/train.csv` + `train_targets.csv`; `test_schema_audit.py` for test. All numbers below are OBSERVED unless marked.

## Facts
| Item | Value |
|---|---|
| Train rows / slates | 5,696 / 772 (matches description) |
| Slate sizes (train) | 4: 48, 5: 44, 6: 66, 7: 24, **8: 590** (76% of slates are size 8) |
| Test rows / slates (schema only) | 1,197 / 164; sizes 4: 15, 5: 9, 6: 8, 7: 12, 8: 120 |
| Role vocabulary | 73 roles in train (57 appear in known_roles, 69 in new roles); 11 roles have ≤ 3 positives; 1 role in test known_roles is outside the train vocabulary |
| New roles per candidate | 0: 2,508 (44%), 1: 2,692 (47%), 2: 461, 3: 33, 4: 2 |
| Known roles per seed | 0: 1,674, 1: 3,449, 2: 510, 3: 63 (row counts) |
| `oracle_rank` | values {0,1,2}; **larger = selected first**: sorting by it descending gives slate score 1.000 (ascending 0.336) |
| Hard mask | new ∩ known = ∅ in all 5,696 rows (structural, guaranteed by definition) |
| Distinct candidate sentences | 2,659; **1,785 of them appear in more than one slate** (different seeds → different known masks) |
| Label contradictions across appearances | **0** of 3,133 comparable pairs (a role observed present in one appearance is never observed absent in another) |
| Seeds | 772 distinct seed sentences, none equals any candidate sentence |
| Cohorts (anchor lower-cased + type) | 256; slates per cohort: min 1, median 2, max 27 |
| Anchor types | Location 3,208 rows, Person 1,246, Event 706, Organization 536 |
| Sentence length | mean 11.5 words, max 23 (cheap to train; MAX_LEN 64 would suffice) |
| One anchor per slate | yes (0 slates with more than one anchor) |
| Top new roles | in:Location:located_in:Location 571, in:Organization:located_in:Location 464, in:Person:visits:Location 280, in:Event:Occurs_at:Location 211, out:Person:holds_title:Title 199 |
| Roles vs anchor type | 7 roles occur with more than one anchor type, so type is informative but not a hard rule |

## Baselines on train under the exact metric (no model)
random pair (expected) **0.513**; id order 0.506; role prior by anchor type, pointwise top-2 0.506; type prior with expected-union decode 0.528.
→ the anchor type and role frequency carry almost no signal; the text of the candidate carries it. Everything above ~0.55 must come from understanding which relation the sentence asserts about the anchor.

## Implications
- **Decision unit** = slate; **valid output** = distinct finite scores per candidate; the pair is picked by the top-2 scores.
- Label factorisation holds (D4 verified): merge per-sentence observed labels across appearances; known roles are unobserved for that row (masked BCE).
- 44% of candidates add nothing (all-zero new roles): the metric is dominated by finding the 1-2 informative candidates and avoiding redundancy; precision on rare roles matters less.
- Role skew (location roles dominate): redundancy-aware decode matters when several candidates assert the same location role.
- Validation: group by cohort (256), ~154 slates per fold with 5 folds (test has 164 slates).
- Capacity: only 3.4k distinct short sentences incl. seeds, 256 independent cohorts: strong regularisation, 3-4 epochs, avoid memorising anchor names (anchor-type markers, optional anchor masking ablation).
- Compute: sentences are short; a large encoder at MAX_LEN 64 should train in minutes per fold on an A10G. The dev sandbox has 4 CPU cores and no GPU: only small encoders can be cross-validated here.

## Leakage and compliance notes
No ordering/id signal used (slate key only for grouping). Test seeds' known_roles are used only as the per-row mask at inference. Test-only vocabulary is not used (role vocabulary from train only).
