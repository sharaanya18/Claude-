# Data audit (train only)
- 3675 train charts (binary 0/255 PNG, height 300, width in {680,760,840,920}); test 1367, same renderer. Plot area: cols 38..W-39, rows 18..281.
- Marks: 7468; 30.7% of charts empty; mean 2.0 marks/chart (max 25). Mark x in [40, 880]; median gap between marks 84 px; only 0.16% of gaps < 5 px (ok for greedy matching).
- Bracket mix of marks: weak 27%, split 21%, firm 52% (micro: weak 39%/firm 35%; antibiotics/lims firm ~60%; numeric & duplicates fieldtypes weaker).
- Folds = whole (dataset_name, fieldtype) cells: fold0 = pas|categorical (1814 charts, 49% of train); other folds are unions of small cells.
- Test cells: antibiotics|categorical 144, lims|uniqueidentifier 188, micro|categorical 706, pas|datetime 329.
- n_points: day ~6300 (up to 12k), week ~890, month ~210; width tied to span, not to n_points.
- Information ceiling: label is human consensus; same-chart disagreement is irreducible. Reference CNN 0.529 on test.
