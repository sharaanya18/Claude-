Complementary Entity Evidence Selection
Overview
An entity dossier already contains one evidence sentence and an inventory of the relation roles observed in that sentence. A curator can add only two more sentences. Rank a slate of real candidate sentences to maximize the new supported relation roles, without wasting the two slots on redundant evidence.

All candidates mention the same lexical anchor as the existing evidence. Lexical similarity alone therefore does not establish value: a fluent and closely related sentence may repeat known roles, while a less similar sentence may add a missing role. Candidates can overlap with each other as well as with the seed sentence. The target measures complementary evidence selection, not single-sentence graph extraction.

The selected runtime is one NVIDIA A10G with a 90-minute training-and-inference budget. Train a model using the supplied examples. The input sentences and underlying annotations are real; preparation derives marginal-role sets without generating sentences, labels, or negative assertions.

Dataset
File descriptions
dataset/public/train.csv — Candidate rows organized into training slates.
dataset/public/train_targets.csv — Exactly one labeled target per training row, with columns id,target. The target JSON has query_id, roles, slate_size, and oracle_rank. Roles are the candidate's incident directed relation-role inventory minus the roles already known from the seed. The slate size is the expected number of candidates and makes partial-slate evaluation invalid. The rank records one deterministic optimal two-candidate choice; other choices can be equally optimal. It is supervision, not a requirement to imitate a particular tie choice.
dataset/public/test.csv — Held-out candidate slates with identical feature columns.
dataset/public/sample_submission.csv — Seeded random finite ranking scores for every test candidate.
Column descriptions
id — Opaque candidate key of the form ev_<slate-key>_<candidate-key>. Rows sharing the middle key are one slate. Use these keys only for grouping and output alignment.
anchor — Mention text shared by the seed and candidates.
anchor_type — Supplied mention category.
query_text — The real seed evidence sentence, unchanged.
known_roles — JSON list of roles already supported by the seed. Each role combines an incoming/outgoing direction and a relation category. These are supplied observations, not predictions.
candidate_text — A real candidate sentence, unchanged.
anchor_start and anchor_end — Zero-based Python character offsets delimiting the anchor mention in the candidate; the end is exclusive.
target — Training-only JSON supervision described above.
prediction — Finite numeric ranking score. Larger means select earlier within the slate. Scores need not be probabilities.
Preparation pools the supplied records, deduplicates exact sentences, and assigns each retained sentence to one lexical-anchor cohort. Cohorts need at least five sentences and three distinct incident-role inventories. A deterministic text-only rule assigns the most frequent eligible anchor in a sentence. Seed rows use up to eight other real sentences from their cohort, selected independently of the role values. Only slates with nonzero marginal utility and differing two-candidate utilities are retained. All anchor cohorts are assigned wholly to training or testing; no sentence crosses that boundary. Public/private scoring also keeps entire cohorts and slates together. Approximately one third of held-out cohorts supply public feedback, with the rest reserved for private ranking. The larger public allocation reduces feedback variance from the limited number of independent cohorts.

The holdout measures transfer to unseen lexical anchors, not a temporal split or verified identity resolution. A shared surface form can be ambiguous; the task uses supplied lexical cohorts rather than claiming global entity identity. Annotation omissions, rare roles, read-speech sampling, and the exclusion of small or uniform cohorts limit coverage. Repeated candidate appearances are not independent new sentences. Training has 5,696 candidate rows in 772 slates representing 3,431 distinct sentences and 256 normalized anchor/type cohorts; testing has 1,197 candidate rows in 164 slates representing 743 distinct sentences and 59 normalized anchor/type cohorts. These are derived decision counts, not 6,893 independent records. Seeds are selected by a deterministic input-only hash before reading utilities; seeds and candidate sentences are disjoint, so public seed roles cannot reveal a candidate's annotations by lookup.

Evaluation
Higher is better; the score range is [0,1]. For each slate, select the two highest scoring candidates, breaking score ties by ascending candidate ID. Let each candidate's hidden new-role set be its incident role inventory minus the seed's known inventory. The slate score is the size of the selected pair's union divided by the largest union attainable by any pair in that same slate. Average slate scores equally.

This is Marginal Role Coverage@2. It directly measures how efficiently a two-item evidence budget expands the recorded inventory. Repeated roles count once, unannotated facts receive no credit, and every retained slate has a positive denominator. A perfect complementary pair scores 1 even if several equally good pairs exist. An empty or incomplete submission is rejected. Public and private scores use the same formula over complete slates; partial-slate scoring is unsupported. Incorrect but finite rankings receive their actual utility, not a parsing penalty.

Complete evaluation code:

import json
from itertools import combinations
import numpy as np
import pandas as pd

def _answers(answers):
    if list(answers.columns) not in (["id", "prediction"], ["id", "prediction", "visibility"]):
        raise Exception("Answers must contain id,prediction and optional visibility.")
    if answers.empty or answers.isna().any().any() or answers.id.astype(str).duplicated().any():
        raise Exception("Answers must be non-empty with unique non-missing IDs.")
    result = {}
    for row in answers.itertuples(index=False):
        item = json.loads(row.prediction)
        if set(item) != {"query_id", "roles", "slate_size", "oracle_rank"} or not isinstance(item["query_id"], str):
            raise Exception("Invalid answer payload.")
        roles = item["roles"]
        if not isinstance(roles, list) or any(not isinstance(r, str) or not r for r in roles) or len(roles) != len(set(roles)):
            raise Exception("Invalid answer role set.")
        result[str(row.id)] = item
    return result

def decode_answers(answers):
    parsed = _answers(answers)
    result = answers.copy()
    result.prediction = result.id.astype(str).map(lambda rid: parsed[rid]["oracle_rank"])
    return result

def _numeric(value):
    # Oracle payloads may be copied by calibration; membership remains gold-side.
    if isinstance(value, str) and value.lstrip().startswith("{"):
        value = json.loads(value)["oracle_rank"]
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise Exception("Predictions must be finite real ranking scores.") from exc
    if not np.isfinite(value):
        raise Exception("Predictions must be finite real ranking scores.")
    return value

def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if list(submission.columns) != ["id", "prediction"]:
        raise Exception("Submission columns must be exactly id,prediction.")
    if submission.empty or submission.isna().any().any():
        raise Exception("Submission must be non-empty without missing values.")
    ids = submission.id.astype(str)
    if ids.duplicated().any():
        raise Exception("Duplicate submission IDs.")
    gold = _answers(answers)
    if len(submission) != len(gold) or set(ids) != set(gold):
        raise Exception("Submission IDs must match scored answer IDs exactly.")
    predictions = {str(r.id): _numeric(r.prediction) for r in submission.itertuples(index=False)}
    queries = {}
    for rid, item in gold.items():
        queries.setdefault(item["query_id"], []).append(rid)
    scores = []
    for members in queries.values():
        expected = {gold[rid]["slate_size"] for rid in members}
        if len(expected) != 1 or any(type(n) is not int or n < 2 or n > 8 for n in expected) or len(members) != next(iter(expected)):
            raise Exception("A scoring partition must retain complete candidate slates.")
        role_sets = {rid: set(gold[rid]["roles"]) for rid in members}
        optimum = max(len(role_sets[a] | role_sets[b]) for a, b in combinations(members, 2))
        if optimum == 0:
            raise Exception("Scored slate has no new supported roles.")
        selected = sorted(members, key=lambda rid: (-predictions[rid], rid))[:2]
        scores.append(len(role_sets[selected[0]] | role_sets[selected[1]]) / optimum)
    score = float(np.mean(scores))
    if not 0 <= score <= 1:
        raise Exception("Score outside [0,1].")
    return score

Submission
Write a UTF-8 CSV at ./working/submission.csv with exactly:

id — Every test candidate ID exactly once.
prediction — A finite real number determining its within-slate ordering.
Requirements
Example using actual prepared IDs (illustrative scores, not answers):

id,prediction
ev_54b40ac3c866a7d7e8a6_b3691eba4e85c0c0,0.37
ev_54b40ac3c866a7d7e8a6_3a79f5d5d87b17f3,0.62

Include all 1,197 candidate rows, with no missing, duplicate, or unknown IDs.
Use the exact column order id,prediction; do not include slate IDs as extra columns.
Global score scale does not matter; only within-slate ordering determines selection.
Train using the public training data. Use test candidates only for inference.
Pretrained general-purpose language representations and learned ranking/role models are allowed. The complete pipeline must perform meaningful training and fit the declared budget.
Do not hardcode the example scores or training oracle tie choices as hidden predictions.
What Not To Use
Do not search the web, retrieve annotation mirrors, or match sentence text to external labels. This is a closed-data selection task.
Do not reverse-map opaque identifiers or treat the deterministic candidate order as target evidence.
Do not infer hidden roles from repeated leaderboard probes or share hidden rankings.
Do not add generated sentences, synthetic labels, pseudo-labels, or external training examples.
