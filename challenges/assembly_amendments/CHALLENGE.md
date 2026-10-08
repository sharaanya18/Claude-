## Overview
In the French National Assembly every bill is amended article by article, first in the committee that examines it and then in the plenary sitting. An amendment proposes an exact change to the text, the legal edit, and comes with a short explanatory summary written by its authors. When the sitting reaches an article it calls the amendments tabled on it in an order fixed beforehand, and each one ends in one of five ways. It is adopted. It is rejected on a vote. It falls, meaning it is never put to the vote because an earlier decision, typically the adoption of a competing amendment or of a rewrite or deletion of the article, left nothing for it to change. It is withdrawn by its author, often at the request of the rapporteur or the government. Or it is not moved, because none of its authors was there to defend it.

Before the sitting, the Assembly's services put the amendments that compete for the same passage into one joint discussion, and identical amendments tabled by different deputies are always called together. This grouping is what makes the amendments of an article interact: once one member of a joint discussion is adopted, the members called after it fall.

This challenge gives you, for each article, the amendments that were discussed on it, with their authors removed, and asks you to reconstruct how the article went: which amendments carried, how each of the others ended, and which amendments were called together.

## Data
train.csv and test.csv: one row per board. A board is one article of one text in one examining body, or the set of amendments proposing new articles after (or before) that article. Columns: item_id (the board id); bill_id (an opaque id shared by all boards of the same bill); bill_title; bill_kind (the procedure: member's bill, government bill, finance bill, organic or constitutional bill, resolution); reading (first reading, second reading, new reading); text_examined (the text as tabled or transmitted, or the committee's text when the plenary examines it); examining_body (the committee, or "Séance publique" for the plenary sitting); division (the article, for example "Article 3", or "article additionnel après l'article 3" for amendments that propose new articles after it); n_amendments; and counts, a JSON list [adopted, rejected, fell, withdrawn, not moved] giving how many of the board's amendments ended in each way.

train_items.csv and test_items.csv: one row per amendment: item_id (the board it belongs to), item_no (1 to n_amendments, in a random order), dispositif (the legal edit, in French) and expose (the authors' summary, in French, or "(pas d'exposé sommaire)" when there is none).

train_targets.csv: the training answers in exactly the submission layout.

sample_submission.csv: a structurally valid placeholder that claims nothing.

There are 4,392 training boards with 31,566 amendments and 622 test boards with 8,851 amendments. Bills, not boards, are split: the committee stage, the plenary stage and every reading of a bill sit on the same side. Test boards hold 4 to 80 amendments with at least two different fates; training boards are kept whatever their size or mix of fates. The test set is divided into two evaluation slices by bill.

## What you predict
fates: a JSON list with one code per amendment, in item_no order: 0 adopted, 1 rejected, 2 fell, 3 withdrawn, 4 not moved. Each code must be used exactly as many times as the board's counts say.

joint: a JSON list with one integer group label per amendment, in item_no order. Amendments that were called together share a label and every other amendment has a label of its own. Only which amendments share a label matters, not the label values.

## How the boards were made
The source records every amendment's processing state and, for discussed amendments, its fate. Only amendments that were discussed on an article of a bill are used. Amendments ruled inadmissible are left out, because their text is not published; so are sub-amendments, amendments that only move budget credits between programmes (a table of figures, not a legal edit), and amendments withdrawn before the sitting.

Amendments with the same legal edit on the same board, usually identical amendments tabled by several deputies or groups, are kept once. The copy kept is the earliest tabled, and its fate is the fate most of the copies had, with ties going to the earliest copy. The joint label comes from the services' records of joint discussions and of groups of identical amendments: two amendments share a label when they were called in the same joint discussion or in the same group of identical amendments, directly or through another amendment of the board.

## Leakage controls
The author, the author type (deputy, rapporteur or government), the political group, the co-signatories, the number of identical copies and the tabling order are not published, and the names and abbreviations of the political groups, and phrases such as "les députés socialistes" in which authors name their party, are masked as "[groupe]" in the texts, so the fate has to be read from the edit and its justification rather than from who tabled it. Amendment numbers, dates, the calling order and every outcome field are withheld, and the amendments of a board are shuffled and renumbered. Bills are split by legislative dossier, and dossiers that share three or more distinctive legal edits word for word (for example amendments re-tabled from one budget to the next) are kept on the same side. The 29 test amendments whose edit still occurred in a training bill were removed. Board and bill ids are opaque digests.

## Submission format
Submit a CSV with exactly the columns item_id, fates and joint, one row per test board.

item_id,fates,joint

a0b1c2d3e4f5,"[1,2,0,1,4]","[0,1,1,2,3]"

## Evaluation
Each test board receives up to three terms, each corrected so that an uninformed answer scores zero.

carried, weight 0.35: adopted or not. If the fates list has the wrong length, an unreadable value, or uses any code a different number of times than the counts say, this term and the next are -1. Otherwise the share of amendments put on the right side of adopted / not adopted is chance-corrected against a random settlement that respects the counts: (earned - expected) / (1 - expected). A board with no adopted amendment is not scored on this term.

disposal, weight 0.35: for the amendments that were not adopted, how they ended (rejected, fell, withdrawn, not moved). The share placed on the right fate, where placing one among the adopted counts as wrong, is chance-corrected the same way.

joint, weight 0.30: the adjusted Rand index between your grouping and the recorded one, so leaving every amendment alone scores 0, and so does putting them all in one group unless that is the recorded grouping. A list of the wrong length or one that cannot be read scores 0. Boards where no two amendments were called together (227 of the 622) are not scored on this term.

A board's score is the weighted mean of the terms defined on it, each floored at -1. The final score is 100 times the mean board score, floored at 0.

## Why these terms and weights
The fates carry 0.70 of the weight because they are the question the task is about, and they are split in two because the outcome is hierarchical: whether an amendment carried is the decision that changes the law, and how an amendment that did not carry was disposed of (voted down, overtaken by another decision, withdrawn or abandoned) is a second, different reading of the same sitting. Each half is scored inside its article, against the article's own counts, because which amendments carry only means something relative to the others competing for the same text; giving the counts removes how productive a particular sitting was from the question.

The joint grouping carries the rest because it is what ties the fates together: an amendment falls because something in its joint discussion was adopted first, so a model that cannot tell which edits compete for the same passage cannot explain the falls. The adjusted Rand index is used because it is chance-corrected by construction and gives nothing for the two trivial answers.

## What not to use
Do not use any external copy of these amendments, of their outcomes, of the records or reports of the sittings, or any model trained on them, and do not look up the withheld authors or outcomes. The task is to learn the fates and the grouping from the training boards provided.
