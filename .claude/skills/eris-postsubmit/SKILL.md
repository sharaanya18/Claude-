---
name: eris-postsubmit
description: After a submission returns a public score for an Eris challenge: compute the CV-vs-public gap, judge whether the CV harness is trustworthy, check against the AI baseline, log it, and pick the single highest-leverage next experiment. Use right after each submission result.
---

# /eris-postsubmit

Inputs: submission number, public LB, the CV score it was based on. Gap = CV − public.
1. **Is CV trustworthy?** A gap above ~0.02 (or an order disagreement across submissions) means look for split mismatch, leakage or selection double-dipping (`/eris-validation`, `eris-red-team`). The public slice is small and
   noisy: a gap alone is not a reason to retune on it.
2. **Above the AI baseline?** State the exact minimum needed from the contract/leaderboard.
3. **One next experiment** with the highest expected private gain, not a menu.
Append a row to `CHALLENGE_NOTES.md § Submission history` (sub, based on exp, public LB, credits left, gap, verdict). Never freeze a constant *because* it scored best across submissions without re-deriving it from train-only validation.
