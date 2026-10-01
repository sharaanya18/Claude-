# U4 — Surface scale and sun geometry from rover views (user-supplied, verbatim; Domain: Fine-Tuning, Hard, A10G, tags image/large-scale)

## Overview
This is a fine-tuning task: adapt a pretrained image backbone, or train one from scratch, to read the acquisition state of a small colour view of the Martian surface.

The images are real. They come from a mast-mounted colour camera on a robotic rover crossing a crater floor, a river delta and the terrain around them on Mars. The camera sits at a fixed height, points down at a wide range of angles, and can zoom, so two frames of the same ground can differ by more than an order of magnitude in how much terrain they hold. Nothing is computationally generated. Each released view was cut from one frame at an undisclosed position and magnification, turned through an undisclosed in-plane angle, resampled, re-encoded, and stripped of all metadata.

For each view, produce four numbers describing where the camera stood relative to the ground it looked at, and where the Sun was. Nothing about the camera, day, location or pointing is supplied, and the terrain carries none of the objects a person normally uses to judge size.

## Prediction targets
standoff_m - metres from the camera to the ground at the centre of the view, on a level surface passing under the rover. About 2.25 to 20.4.
footprint_m - metres of ground the view spans across the line of sight, at that same point. About 0.06 to 3.4.
sun_elev_deg - elevation of the Sun above the local horizontal when the frame was taken. About 5.5 to 88.
sun_bearing_deg - angle between the camera's horizontal facing and the Sun's. 0 means facing the Sun, 180 the Sun behind. Full range 0 to 180.
All four are unchanged by the in-plane rotation and mirroring applied to the views, so any orientation is equally valid evidence.

## Data
train/images/<row_id>.jpg, test/images/<row_id>.jpg - one 192 x 192 RGB JPEG per row. 15,114 train, 6,468 test, no image in both.
train.csv - 15,114 rows: row_id, capture_group, then the four targets above.
test.csv - 6,468 rows: row_id, capture_group.
sample_submission.csv - 6,468 rows, the five submission columns.
row_id is mzv- plus 8 lowercase alphanumerics, and is the file name of that row's image.
capture_group is mzv-g plus 6 lowercase alphanumerics, shared by every row from the same rover working location, so rows sharing a code are not independent. Every code in test.csv names a location absent from train.csv - hold out whole groups when validating. It marks a location, not a target: answering each row with its group's mean true value scores the floor.
Every value in sample_submission.csv is an independent random placeholder inside its column's band - not labels.

## The answer key
Nobody estimated these values: each was computed exactly from a construction record withheld in full, so the key has no annotation noise and every disagreement is information loss. Producing them from anything other than the released pixels and the training labels is prohibited.

## Submission
One CSV, header, exactly: row_id, standoff_m, footprint_m, sun_elev_deg, sun_bearing_deg
Exactly one row per row_id in test.csv - 6,468 - no duplicate row_id, no missing, extra or renamed column, every value finite. Values outside the accepted bands are clipped, not rejected: standoff_m to [2.20, 30.0], footprint_m to [0.03, 10.0], sun_elev_deg to [3, 89], sun_bearing_deg to [0, 180].

## Scoring
View Reconstruction Skill. Maximize, range [0.05, 1.0]; perfect = 1.0.
1 - Clip predictions and truths into the bands above.
2 - Reconstruction. With r = standoff_m, w = footprint_m and fixed H = 1.995 m:
d = sqrt(r^2 - H^2); a = (d/r, 0, -H/r) optical axis; v = (H/r, 0, d/r) frame up; f = 96 / tan(arctan((w/2)/r)) focal length in pixels. Camera at C=(0,0,H), frame centre meeting the ground at G=(d,0,0). A world point P lands at den = Px*a_x + (Pz-H)*a_z; x = f*Py/den; y = f*(Px*v_x + (Pz-H)*v_z)/den.
3 - The three marks. h0 = 0.15*w_true, delta = 0.50*w_true, s = 0.50*w_true, b = sun_bearing_deg: M1=(d,0,h0) top of a vertical mark at G; M2=(d+s*cos(b), s*sin(b), 0) a ground mark toward the Sun; M3=(d+delta,0,0). Each projected twice, with predicted and true (r,w,e,b): disp = min((|M1p-M1t|+|M2p-M2t|+|M3p-M3t|)/(3*192), 3).
4 - Five terms: S_view = exp(-disp/0.25529558); S_foot = exp(-|ln(w_pred)-ln(w_true)|/0.88505864); S_stand = exp(-|ln(r_pred)-ln(r_true)|/0.63674155); S_elev = exp(-|e_pred-e_true|/0.25593667); S_bear = exp(-|b_pred-b_true|/1.00008771). (angles in radians in the two solar terms). Each constant makes the best constant submission score exactly 0.50 on that term over training.
5 - Blend: S_axes = 0.25*(S_foot+S_stand+S_elev+S_bear); raw = exp(0.25*ln(S_view) + 0.75*ln(S_axes)).
6 - Chance correction: score = clip((mean(raw) - REF)/(1 - REF), 0.05, 1.0), REF = 0.4934220017 (best constant on training).
Value of each target (replacing one target at a time from the best constant): standoff 0.1054, footprint 0.1548, sun_elev 0.1706, sun_bearing 0.1035; both solar 0.3727; both length 0.4051; all four 1.0.

## Forbidden approaches
A solution must train a model on the provided training data and use it to produce the test values; non-ML solutions are rejected. Public pretrained weights are allowed and encouraged - fine-tuning a public backbone from a standard model hub is explicitly permitted and worth roughly a factor of four here. They are not external data.
Forbidden: (a) Test-time augmentation of any kind. (b) Pseudo-labelling, self-training, or any transductive use of the test set. (c) Lookup tables or values fitted to test labels. (d) Anything that fits, calibrates or normalises on test statistics, or touches the test set beyond the final prediction pass. (e) Inference-only solutions that ship weights already trained for this task. (f) External data of any kind. (g) Basing any development decision on the test set. (h) Reading any text, marker or identifier visible in an image, including OCR. (i) Matching a released view against anything outside this challenge (hash, feature matching, embedding search, reverse image search) to identify where it came from or recover any property of how it was taken. (j) Hand-coded heuristics as the solution - fixed thresholds, hand-written edge/texture/frequency rules, formulas with hand-chosen constants, template matching, or nearest-neighbour lookup on raw pixels; classical descriptors may be features inside a learned model. (k) Exploiting incidental file artefacts - file size, JPEG quantisation tables, encoder fingerprints, file or row_id ordering.
Allowed: any learned model fitted on provided training data (CNNs, ViTs, hybrids, from scratch or fine-tuned) and ensembles of models you trained. Training-time augmentation is allowed and expected to help, provided every augmentation is one the targets are genuinely invariant to.
Compute: A10G. (Runtime limit not stated in the pasted text; assume the platform default.)
