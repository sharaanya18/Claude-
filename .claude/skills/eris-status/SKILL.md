---
name: eris-status
description: Dashboard across all challenge workspaces: domain, best CV, best public LB, credits, phase, and the next best use of time. Use at the start of a session or when deciding what to work on.
---

# /eris-status

For each `challenges/<slug>/`: read `CHALLENGE_NOTES.md` (domain, best CV, best LB, status checkboxes, submission history) and `working/experiment_log.csv` (latest CV). Print a table
`Challenge | Domain | Best CV | Best public LB | Credits | Phase`. Then recommend the next best use of time: (1) challenges with a good CV never submitted (the free CSV check costs nothing), (2) pending
experiments with a measurable CV gain, (3) challenges near closing, (4) new challenges needing a quick credible baseline. Credits reset 24 h after each use; per challenge 6 + 1 per 4 h; global ≈ 15–25 per day.
