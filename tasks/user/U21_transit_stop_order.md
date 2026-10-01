Transit Stop Order Reconstruction
Editor View
This challenge is closed

No further submissions, edits, or withdrawals are possible. 
How payouts are processed

Transit Stop Order Reconstruction
Creator

duongnguyen
Approver

aikyatan
Flag Problem
Domain
Other
Difficulty
Easy
Scoring
↑ Higher is better
Compute
A10G
Status
Accepted
Dataset source is visible after the challenge closes.
Description
Leaderboard
(0)
Your Submissions
Overview:
A bus takes a fixed list of stops in a fixed order. You are given the list, scrambled, and asked for
the order.

Each case is one service pattern from one transport network: between 10 and 40 stops, presented in a
randomised order, plus the identifier of the stop the vehicle departs from. You submit the stops
arranged in the order the vehicle visits them. There are 14,424 training cases with their true orders
and 4,822 evaluation cases without.

Two things are available to work from, and neither is sufficient alone.

The first is position. Every stop has an approximate location on a plane. If routes were efficient
tours this would settle the question, and it very nearly does: a nearest-neighbour tour from the given
first stop, improved by 2-opt, scores 0.317. It stops there because real routes are not shortest
tours. They follow roads and one-way systems, they double back to serve a school or a hospital, they
run out and return along the same corridor, and they are laid out to be legible to passengers rather
than short. Every point of the remaining 0.718 is a place where operational reality departs from
geometry.

The second is the network. Each case belongs to a network whose other patterns you are given in full,
in their true order, through the training file. Stops are shared between routes, so a stop you have
to place may already appear in fifteen other patterns whose order you know. That is direct evidence
about which way traffic runs along a corridor and which stops sit between which. It is also
incomplete: the evaluation patterns come from routes held out entirely, and averaged over these
networks 57.8 percent of stops are served by only one route, so about half the stops on a typical
route have no other pattern to learn from.

Task definition:
A case is a set S of n stop identifiers with 10 <= n <= 40, together with a designated element f of S.
Exactly one permutation of S is correct: the order in which the vehicle visits those stops, which
begins at f. You submit a permutation of S and it is compared against that one. Cases are independent
of each other and are scored independently; there are 14,424 labelled cases and 4,822 unlabelled ones.

Every case belongs to exactly one of 58 networks and is only interpretable inside it. Stop identifiers
are scoped to a network and never recur across networks, so no stop, no coordinate and no ordering
evidence is shared between two networks. A solution is therefore 58 independent problems that happen
to share a functional form, not one problem with 58 subsets.

Coordinate regime:
Every stop has a planar position. Those positions were transformed once, per network independently,
before release, by exactly the following steps:

Positions were projected from the sphere onto a local plane in kilometres.
The network's centroid was translated to the origin.
The network was rotated about the origin by an angle drawn independently per network and not
disclosed.
The network was rescaled so that the mean distance from the origin to one of its stops is 1.
Every stop was displaced by independent Gaussian noise on both axes with standard deviation 0.16,
drawn once per stop and fixed from then on.
What follows from that, and what you may rely on. Within a network, relative distances and angles are
faithful up to the one unknown rotation and the one common scale, so the shape of a route is
recoverable but its absolute orientation is not. Across networks nothing is comparable. The
displacement in step 5 is a property of the stop rather than of an observation, so a stop presents the
same displaced position in every case it appears in, and averaging its coordinate over those
appearances returns that same point and removes nothing. The displacement is not small: the mean
distance between consecutive stops on a route is 0.092 in these units, so a standard deviation of
0.16 is about 1.7 times one step along the route, which is why the nearest unvisited stop is
frequently not the next one.

What is not provided, in any form: a street or road network, link geometry, a routing graph, travel
times, headways, timestamps, traces or sequences of vehicle positions, stop names, operator
identities, and any distance matrix other than the one you compute yourself from the supplied
coordinates. The order has to come from the point set, the designated start, and the other routes.

Cross-route transfer:
The split is by route, inside each network. A route contributes several patterns, since it is
operated in two directions and often has short workings and variants, and all patterns of a route fall
on the same side of the split. For an evaluation case, no pattern of its own route appears anywhere in
the training data.

That makes every transferable signal indirect, and the quantities that govern how much is available
are these. A stop appearing in an evaluation case may also be served by other routes of the same
network, whose complete ordered patterns are supplied in the training files; 51.5 per cent of the
stops in a typical evaluation case appear somewhere in the training patterns, and the remaining
48.5 per cent appear nowhere and carry no ordering evidence at all. Averaged over the 58 networks,
57.8 per cent of stops are served by exactly one route. Of the consecutive stop pairs that make up the
evaluation answers, 37.2 per cent occur as a consecutive pair somewhere in the training patterns,
in one direction or the other.

So the ordered evidence is partial by construction and unevenly distributed: dense urban networks
supply a great deal of it and sparse interurban ones almost none, and no single case can be solved by
transfer alone or by geometry alone.

Nearest published work, and the axis on which each differs:

No published work combines these four: a per-case permutation target over opaque items, a geometry
that is deliberately rotated, rescaled and per-item displaced so absolute position is unrecoverable,
supervision drawn only from other routes of the same network, and an evaluation whose split unit is
the route rather than the case.

Evaluation:
The metric is the mean pairwise ordering agreement, written mean_ordering_agreement.

For one case, your submitted order is compared with the true order of the same stops. Take every pair
of stops in the case. A pair counts as agreed when you placed both of them and placed them in the
true relative order. A pair counts as wrong in every other situation, which includes placing them the
wrong way round AND leaving either stop out of your answer entirely. The case score is
(agreed - wrong) / (total pairs), and any case scoring below zero is recorded as zero. The challenge
score is the mean over every evaluation case, so it runs from 0 to 1 and higher is better.

Omitted stops are charged. A stop you leave out is counted as wrongly ordered against every other
stop in the case, so a partial answer cannot score better than a complete one. Submitting only the
stops you are confident about is strictly worse than committing to a full order: listing half of a
20-stop case in perfect order scores 0.000, and dropping a single stop from an otherwise perfect
answer scores 0.760 rather than 1.000. Rank all the stops.

Why this metric: the deliverable is a full ordering and every pairwise relation in it matters, which
is exactly what a pair-agreement score measures and what a position-by-position accuracy would not.
Getting one early stop wrong shifts every later position and would be punished as a total failure by
an exact-position score, while this charges it only for the pairs it actually inverts. It is directly
interpretable as the fraction of stop pairs you ordered correctly, rescaled so that ordering exactly
half of the pairs correctly scores zero. It needs no weights, no threshold and no cutoff rank, so
there is nothing to tune in the evaluation.

The denominator is every pair in the case, never only the pairs you chose to answer. That is what
makes the score unexploitable by partial submission: a scheme that reports only easy pairs cannot
shrink what it is measured against.

Scores are floored at zero per case rather than allowed to run negative. An ordering and its exact
reverse are equally informative about the route, so a submission that recovers the shape of the route
but runs it backwards would otherwise score close to -1 and be ranked below a random guess, which
measures direction rather than ordering skill. The first stop of every case is given to you as an
input, so getting the direction right requires no inference; flooring at zero keeps a direction
mistake from dominating the score.

No second component is combined with it. A top-k or first-stop-accuracy term would be redundant,
since the first stop is given to you as an input. A per-network macro average would also be
redundant: the evaluation networks are not held out, only routes within them are, and every case
contributes exactly one score regardless of which network it came from.

Floor and ceiling, all measured on the evaluation cases. A perfect submission scores 1.000. The
supplied sample_submission, which lists the stops in the order they are given, scores 0.071, and a
random shuffle scores 0.070, because the given order carries no information; that is the guessing
floor, and it sits above zero because a case that is ordered worse than chance is recorded as zero
rather than as a negative number. Sorting the stop identifiers as strings scores 0.070. Placing the
given first stop at the front and leaving the rest as supplied scores 0.142. Ranking by how often a
stop precedes the others in the training patterns scores 0.173, ordering by a stop's average relative
position across the training patterns scores 0.216, and an adjacency win-count over the training
patterns scores 0.085. Ordering the stops along the dominant spatial axis of the case scores 0.236. A
nearest-neighbour tour built from an arbitrary end scores 0.165. The strongest route that involves no
learning is the nearest-neighbour tour from the given first stop improved by 2-opt, at 0.317. The
distance from 0.317 to 1.000 is what a model has to find in the network structure and in the ways
real routes deviate from efficient tours.

Dataset:
The following files are provided in public/.

train.csv - one row per training case, 14,424 rows.
case_id - text - identifier for this case.
network - text - an opaque code for the transport network this case belongs to. Stop identifiers
are only meaningful within one network.
n_stops - integer - how many stops are in this case, between 10 and 40, and therefore how long your
answer must be.
first_stop_id - text - the stop the vehicle departs from. It is always one of the stops in
stop_ids, and it always belongs at position 1 of the true order.
stop_ids - text - the stops of this case, space-separated, in a randomised order that carries no
information. No stop appears twice.

train_labels.csv - the training labels, one row per training case, 14,424 rows. Join to train.csv on
case_id.
case_id - text - identifier matching train.csv.
ordered_stop_ids - text - the same stops as that case's stop_ids, space-separated, in the true
order the vehicle visits them.

test.csv - one row per evaluation case, 4,822 rows, with exactly the same columns as train.csv in the
same order. Its cases come from routes that contribute no training case, and no labels are provided.

stop_coordinates.csv - one row per stop, 96,027 rows, covering every stop referenced by any case.
network - text - the network code, joining to train.csv and test.csv.
stop_id - text - the stop identifier, unique across the whole file.
x - number - planar coordinate in arbitrary units. Approximate, see Overview.
y - number - planar coordinate in arbitrary units. Approximate, see Overview.

sample_submission.csv - a correctly formatted submission listing each case's stops in the order they
are given, scoring 0.071.

Submission:
Submit a CSV with exactly these two columns, named exactly as shown. A submission whose columns are
renamed is rejected rather than scored.

case_id - text - one row for each case_id in test.csv, 4,822 rows plus a header row.
ordered_stop_ids - text - that case's stops, space-separated, in your predicted visiting order.
List all of them; a stop you leave out is treated as unplaced and costs you every pair it appears
in.

Example, with fabricated identifiers for illustration:

case_id,ordered_stop_ids
case_0001example,s_00000000aa s_00000000bb s_00000000cc
case_0002example,s_00000000cc s_00000000aa s_00000000bb

Requirements: a header row, one row per test case, in any order. Write the file without a pandas index
column. An empty ordered_stop_ids cell scores zero for that case rather than being rejected, and a
duplicated case_id is rejected.

Rules:
The only valid input signal is the stop sets, the supplied coordinates, the given first stop, and the
network structure you can learn from the training cases and their labels.

The following approaches are not allowed:

Hardcoding an order for specific test cases.
Using the case_id, network or stop_id strings as a prediction signal. They are opaque hashes
provided so you can join rows, not so you can mine them.
Using the order in which stops appear inside stop_ids, or the row order of any supplied file, as a
prediction signal. Presentation order is randomised per case and carries none.
Identifying the real transport networks these cases come from and consulting any external source
about them: a published timetable, a transport data portal, a routing or mapping service, a
geocoder, or any dataset of stops or routes. Recovering the real stops behind the anonymised
identifiers, or the real order behind a case, is retrieval of the answer key rather than modelling,
and it is the one exploit that would defeat the challenge outright. The coordinates are rotated,
rescaled and displaced per network specifically so that they cannot be matched against a real map.
Using private, role-gated, or API-key-based models, or calling any external inference API at
inference time.
Using non-reproducible external weights or artifacts not publicly available.
The intended task is to learn how a real service route is laid out: that it tends to advance steadily
rather than jump between distant points, that it doubles back for specific kinds of destination, that
corridors are traversed in a consistent direction, and that a stop's neighbours in one route predict
its neighbours in another. The exploits above bypass exactly that, and the pool-order and
coordinate-only attacks in particular were measured during construction and sit at the random floor.

Pretrained model policy:
There is no pretrained model for this task, and nothing here can be solved by recalling text. Any
approach must be fitted to the supplied training cases.

Fine-tuning a publicly available pretrained model is permitted, provided the fitting genuinely uses the
supplied training data. Using private, role-gated or API-key-based models, and calling any external
inference API at prediction time, are prohibited above.
