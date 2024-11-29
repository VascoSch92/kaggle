
# Kaggle

---

This repository contains my work for (some) Kaggle competitions.

### Specifically:
- A framework for different competitions involving **tabular data** (see the module `tools`).
- Code for generating submissions for various competitions.

## How to Reproduce an Experiment/Submission

Experiments and submissions are managed via the command line using the following formula:

```bash
python . COMPETITION_NAME --FLAGS
```
Where:
- `COMPETITION_NAME`: The name of the competition.
- `--FLAGS`: The steps in the pipeline.

This command executes the respective pipeline and produces a submission file 
in the folder `kaggle/PROJECT_NAME/data`. The submission file is named 
`submission_UID.*`, where `UID` is a unique identifier for the submission.

Retrieving Code for a Specific Submission
To retrieve the code associated with a specific submission:

Search the commit history for the commit containing the `UID` of the submission.
Check out the commit corresponding to the desired `UID`.
Run the command included in the commit message to regenerate the submission.
Commit messages for submissions follow this format:

```text
UID(COMMANDS)
```
Where:
- `UID`: The unique identifier for the submission.
- `COMMANDS`: The exact command used to generate the submission.

## Competitions

### Exploring Mental Health Data

Playground Series - Season 4, Episode 11

Page of the competition [here](https://www.kaggle.com/competitions/playground-series-s4e11)

| Commit Message                         | Public Score | Private Score |
|----------------------------------------|--------------|---------------|
| `b9de241d(mental-health --light-lgbm)` | 0.94131      | TBC           |
| `45dcb497(mental-health --light-lgbm)` | 0.94131      | TBC           |


### Spaceship Titanic

Predict which passengers are transported to an alternate dimension

Page of the competition [here](https://www.kaggle.com/competitions/spaceship-titanic)

| Commit Message                             | Score   | Ranking |
|--------------------------------------------|---------|---------|
| `130f8c4b(spaceship-titanic --light-lgbm)` | 0.80219 | top 15% |
