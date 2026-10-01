---
name: eris-close
description: When an Eris challenge is finished or its result is known, write the specific lessons that would have changed a decision, update CHALLENGE_NOTES.md, and record transferable lessons for future challenges. Use before cleanup.
---

# /eris-close

Fill `CHALLENGE_NOTES.md § Key insights` with specifics: what was unique about this dataset; the single change with the biggest gain; the experiment that failed most unexpectedly and why; what to do differently.
Add to the playbook (`/eris-pattern`) only lessons specific enough to have changed a past decision ("max_length=256 truncates entity spans; use stride 64" yes; "use better features" no). Record final CV vs public vs private gap and
its cause. Update status to closed. Do not delete data until the analysis is done.
