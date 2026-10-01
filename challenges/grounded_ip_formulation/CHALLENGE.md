Weakly Supervised Grounded Integer Program Formulation
Domain: Sequence To Sequence. Difficulty: Easy (label). Scoring: higher is better. Compute: A10G. Tags: text.

Overview:
Each example is an operational planning problem stated in words: a plant choosing the daily mix of six products
under material, machine and labour limits, a carrier deciding how many trucks to send along each of a dozen
routes, a blender meeting nutrient minimums at least cost, an investor splitting a budget under exposure limits.
Half of the problems give their data in a table inside the text. Each is a mixed-integer linear program in words.

Write the program that a solver can run. Declare the objective and every constraint, say which variables are
integer or binary, and give each coefficient and limit as a reference to the number in the text that supplies
it. The grader solves your program.

Four things define the task together, and change what has to be learned:

The target is a grounded, executable model. What is scored is not an answer and not free text: it is a program
whose every coefficient and limit points at a number of the problem, run by the grader's solver.

The supervision is the outcome, not the model. Every training case comes with the optimal value of its reference
formulation only; the formulation itself is given for one training case in ten. Many different programs share an
optimum, so learning from optimal values means inferring, from a single number per problem, which quantities are
decisions, which sentences and table columns are constraints, which limits apply to one variable, to a group or to
the total, and which variables must be whole numbers.

The evaluation re-runs the model on data it has not seen. Each evaluation program is also solved on hidden
counterfactual versions of its problem in which every number has been changed. A program that reaches the right
optimum on the real numbers with a slack wrong constraint or a coincidental coefficient reaches a different
optimum once the numbers move, while the right formulation tracks the reference exactly; and a structure score
compares the rows under one renaming of the variables for the whole model.

The evaluation problems are bigger and of a different mix. Training problems have 4 or more decision variables
and are mostly production-mix problems; evaluation problems all have at least 6 (7 at the median, up to 18),
and about half of them are transport, routing or distribution problems, against a fifth of the training
problems. A model has to carry what it learned to larger models of partly different kinds.

Task definition:
The prediction unit is one word problem. For each case you return one formulation as a text string.

The target is the reference formulation of the problem: a model over nonnegative variables, continuous unless
declared integer or binary, with every coefficient and limit bound to a number of the text. Every target has a
finite optimum.

There are 347 evaluation cases and 2,479 training cases. Training problems have 4 or more decision variables.
Evaluation problems are those whose reference formulation has at least 6 decision variables; they are larger
than a typical training problem, and a model has to carry what it learned from smaller problems to larger ones.

What the inputs are:
For every case: problem, the word problem, with line breaks written as a pilcrow (U+00B6) so that each problem
is one line of text and Markdown tables keep their rows; and numbers_json, a JSON list of [reference, value] for
every distinct number in the problem. References are x<k> with k a per-problem random numbering, so which
reference is which number has to be read from the values. A number is a run of digits with an optional decimal
point and thousands commas; a percentage is listed as a fraction (25% has value 0.25). Numbers that occur more
than once share one reference.

How the data was built:
The problems come from a published optimization-modelling collection in which every problem was written together
with a runnable solver program that builds its model. No reference formulation here was transcribed or typed by
hand. Each program was executed against a symbolic stand-in for the solver interface that recorded the objective,
the constraints, the variable types and the bounds exactly as the program built them, and every numeric constant
of the program was traced back to the number printed in the problem that equals it, which becomes its reference.
A program was discarded if it used a number not printed in the problem (other than 0, 1, -1 and 100), a nonlinear
term or a variable without a lower bound of zero, or failed to run. Every reference optimum was then computed with
the solver the grader uses, cases whose reference has no finite or a zero optimum were dropped, and every
evaluation reference was checked to keep a finite optimum on at least three counterfactual versions of its
numbers. The references are therefore machine-extracted, solver-verified and grounded number by number; their
fidelity to the wording is that of the programs written for the problems, and a small share of them model the
text imperfectly (an assumed or a missing constraint), in training and evaluation alike.

Supervision and split:
train_labels.csv gives the optimal value of every training case. train_seed_formulations.csv gives the
formulation of 248 training cases, a seeded random tenth. No other formulation is supplied.

The split holds out SCENARIOS. The collection states many scenarios several times, in prose and as a table, or
with other numbers; problems that open with the same sentence or use the same set of numbers are one scenario
and are kept on one side, so no evaluation scenario appears in training in any wording. Problems whose reference
formulation has no finite optimum, has an optimum of zero, or has no finite optimum on counterfactual versions
of its numbers are excluded.

Evaluation:
The metric is the mean grounded-formulation score over the 347 evaluation cases. For one case it is the average
of three terms:

Value term: 1 if the optimum of your program, solved with the real numbers, equals the reference optimum within
a relative tolerance of 0.0001, otherwise 0.

Counterfactual term: the share of 3 hidden variants of the problem on which your program and the reference
formulation reach the same optimum (relative tolerance 1e-6). In each variant every number of the problem is
multiplied by its own hidden factor between 0.55 and 1.45 (never between 0.95 and 1.05); numbers with the same
value share a factor, and year-like whole numbers from 1990 to 2030 are left unchanged. Variants on which the
reference formulation has no finite optimum are skipped and replaced by the next one.

Structure term: the F1 overlap between the rows of your program and the rows of the reference after your variables
are renamed to the reference's by ONE one-to-one renaming for the whole model. A row is compared as its sense
(after rewriting it as less-than-or-equal, or as an equality with its first coefficient positive), its set of
(variable, coefficient) pairs and its right-hand side, all computed with the real numbers; the objective is compared
as its direction and its (variable, coefficient) pairs; one more item records which variables are integer and which
binary. Because the renaming is the same in every row, a coefficient attached to the wrong variable makes its row
differ, while choosing other variable names or another row order changes nothing. The renaming is the one that
makes the most rows agree: variables are paired by how they appear across the model (the rows they occur in, their
coefficients, and, refined over three rounds, the variables they occur with), and the pairing is then improved by
pairwise exchanges. The reference scores 1 against any consistent renaming of itself, and variables of yours that
the reference does not have stay unpaired.

Case score = (value term + counterfactual term + structure term) / 3. Higher is better, 1 is perfect. A formulation
that does not parse, is not linear, references a number that does not exist, or has no finite optimum scores 0 on
the terms that need its optimum.

Why this metric: each term catches a failure the other two miss. The value term is the answer, and it is the only
term a program earns that merely reproduces the optimum. The counterfactual term is what makes the formulation
grounded: a constraint that is slack at the real numbers, a coefficient that coincides with the right one, or
numbers typed in instead of referenced leave the real optimum unchanged but give a different optimum once the
numbers move, and a formulation fitted to optimal values alone produces exactly such programs. The structure term
gives partial credit for a program that gets some rows right, which neither optimum-based term can see. It is
invariant to a consistent renaming of variables and to row order, so it rewards the model and not its spelling,
but it keeps variable identity across the whole model: two coefficients exchanged between variables inside a row
leave the row's sorted coefficient values unchanged, and a per-row comparison that ignored variables would still
give that row full credit, whereas under one model-wide renaming the row no longer matches. The three are weighted
equally because each is a distinct property of a correct model: the right
decision value, the right dependence on the data, the right constraints and variable types.

Floor and ceiling, measured on the evaluation cases. A perfect submission scores 1.0000. The supplied
sample_submission.csv (maximise v subject to v at most x0) scores 0.0088. Giving every case the most common seed
formulation scores 0.0059, and giving each case another case's formulation scores 0.0086. Typing in the reference
optimum as a one-line program (maximise v subject to v at most the optimum, or minimise it subject to at least
the optimum) scores 0.3368: the value term only. Writing the reference formulation with the numbers typed in
instead of referenced scores 0.6667: value and structure, but no counterfactual term.

Dataset:
The following files are provided in public/.

train.csv - one row per training case, 2,479 rows.
case_id - text - opaque identifier.
problem - text - the word problem, line breaks written as a pilcrow.
numbers_json - text - JSON list of [reference, value].

test.csv - one row per evaluation case, 347 rows, with exactly the same columns as train.csv in the same order.

train_labels.csv - the optimal value of every training case, 2,479 rows. Join on case_id.
case_id - text - identifier matching train.csv.
optimal_value - number - the optimum of the reference formulation.

train_seed_formulations.csv - the reference formulation of 248 training cases. Join on case_id.
case_id - text - identifier matching train.csv.
formulation - text - the reference formulation.

sample_submission.csv - a correctly formatted submission, scoring 0.0088.
case_id - text - identifier matching test.csv.
formulation - text - the predicted formulation.

Submission:
Submit a CSV with these two columns.

case_id - text - one row for each case_id in test.csv, 347 rows plus a header row.
formulation - text - one program.

Formulation syntax: statements separated by semicolons. The first is the objective, max or min (maximize,
minimize and a trailing colon are also accepted) followed by a linear expression. Each further statement is a
constraint (a linear expression, one of <= >= =, and a linear expression) or a declaration: int followed by
variable names makes them integer, bin makes them binary (0 or 1). Expressions use references (x<k>), plain
numbers (optional thousands commas, an optional trailing % meaning divide by 100), variable names (letters,
digits and underscores, starting with a letter or underscore, not of the form x<k> or q<k>), + - * / and
parentheses; a product or quotient may involve at most one variable side. All variables are nonnegative. At
most 60 variables and 150 constraints.

Example, with fabricated identifiers and references for illustration:

case_id,formulation
p01a2b3c,max x4cake_a + x1cake_b + x9cake_c; x2cake_a + x7cake_b + x2cake_c <= x5; cake_a + cake_b + cake_c >= x0; cake_a <= x3; int cake_a, cake_b, cake_c
p0f9e8d7,min x6route_1 + x2route_2 + x8open_depot; route_1 + route_2 >= x1; route_1 <= x4open_depot; bin open_depot; int route_1, route_2

Requirements: a header row and one row per evaluation case, in any order. A duplicated case_id, a missing case_id
or formulation column, or a file that shares no case_id with the evaluation set is rejected.

Rules:
The only valid input signal is each case's problem text and number list, together with what is learned from the
training cases, their optimal values and the seed formulations.

The following approaches are not allowed:

Hardcoding formulations for specific evaluation cases. Every submitted formulation must be the output of the
trained model's inference over that case. Writing, editing or patching the formulation of any evaluation case
by hand, or keeping any lookup of formulations keyed by case_id, is hand-labelling. Choosing among several
outputs of the trained model for the same case is allowed.
Using a pretrained language model without training it on the supplied data (prompting, in-context examples, or
training cases retrieved into the prompt) as the model that produces the formulations. Its weights must be
fitted on the supplied training cases.
Producing or repairing formulations with hand-written parsing rules or templates (keyword-to-constraint tables,
regular-expression fixes of the model's output, positional defaults). Such rules may build model inputs or
features only; a solution that keeps most of its score with the trained model removed is rule-based and not
allowed.
Searching evaluation cases for formulations. Searching training cases for the formulations that reproduce their
given optimal values is the intended use of the weak supervision; nothing comparable is given, or may be done,
for evaluation cases.
Using case_id strings or the row order of any file as a prediction signal. They are random codes issued so that
files can be joined.
Looking up these problems, their solution programs or their formulations in any outside copy of this material
or of any published collection of optimization word problems, or using any model trained on such formulations.
The only data is the files in public/; downloading or searching the web for any copy of them is not allowed. A
copy of the formulations is the answer key.
Fitting on evaluation cases in any form: no training on their text, no pseudo-labelling and no test-time
adaptation.
Using private, role-gated or API-key-based models, or calling any external inference API at any point.
Using non-reproducible external weights or artifacts that are not publicly available.
The intended task is to learn, mostly from optimal values, how a decision problem with many products, resources
and requirements, stated in prose or in a table, becomes a mixed-integer program, and to bind every coefficient
and limit to the number that supplies it. The lookups, hand rules and untrained prompting above reach a
formulation without learning any of that.

Pretrained model policy:
Fine-tuning a publicly available pretrained language model (for example T5, BART, or an open-weight decoder
model) is permitted, provided its weights are fitted on the supplied training cases, optimal values and seed
formulations. General knowledge of language and of optimization modelling is legitimate; a model or artifact
trained on formulations of these problems is the lookup prohibited above.
