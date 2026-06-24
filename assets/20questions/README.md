twentyquestions
===============
The twentyquestios dataset provides
[20 Questions](https://en.wikipedia.org/wiki/Twenty_Questions) style questions
and answers collected from real games of twentyquestions between people on
Mechanical Turk.

In this directory, you'll find the following files:

  - twentyquestions-train.jsonl
  - twentyquestions-dev.jsonl
  - twentyquestions-test.jsonl
  - twentyquestions-all.jsonl

The files represent the train, dev, and test splits as well as all the data
together before being split. Train, dev, and test were split by dividing up the
questions and subjects into train, dev, and test buckets. Thus, dev has
questions and subjects never seen in train, and test has questions and subjects
seen in neither train nor dev.

Each file is in [JSON Lines](http://jsonlines.org/) format with the following
keys:

  - **subject**: the subject from the game of 20 Questions
  - **question**: the question that the player asked
  - **answer**: the answer that the other player gave during the game, with
    null for questions that were crowdsourced outside of a game of 20
    Questions.
  - **quality_labels**: a list of labels for the quality of the question.
    "good" means the question is high quality, otherwise specific issues are
    called out such as "guess" if the question was a guess or "not-yes-no" if
    the question is not a yes or no question.
  - **score**: the number of crowd workers that assessed the question as high
    quality (out of 3)
  - **high_quality**: the majority vote for the quality of the question
  - **labels**: the truth-values assigned by 3 different crowd workers (each
    value is one of "always", "usually", "sometimes", "rarely", "never" or
    "bad")
  - **is_bad**: whether or not any crowd workers labeled the assertion as bad
  - **true_votes**: the number of votes out of 3 for the question being true
    about the subject
  - **majority**: the majority label for true or false
  - **subject_split_index**: the smallest index (0 train, 1 dev, 2 test) in
    which the subject appears
  - **question_split_index**: the smallest index (0 train, 1 dev, 2 test) in
    which the question appears

Note that some of the rows were obtained after the games of 20 Questions by
taking existing rows and changing the subject to flip the answer from true to
false or vice versa. This step was done to reduce subject-only biases in the
data.
