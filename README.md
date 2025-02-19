
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

- [Spaceship Titanic](#spaceship-titanic)
- [Exploring Mental Health Data](#exploring-mental-health-data)
- [Regression With An Insurance Dataset](#regression-with-an-insurance-dataset)

### Spaceship Titanic

Predict which passengers are transported to an alternate dimension

**Page of the competition:** [link](https://www.kaggle.com/competitions/spaceship-titanic)

| Commit Message                                                     | Score    | Ranking |
|--------------------------------------------------------------------|----------|---------|
| `8355f6cb(spaceship-titanic --booster-ensemble --stratification)`  | 0.80523  | top 11% |
| `8dae923c(spaceship-titanic --catboost)`                           | 0.80476  | top 12% |


### Exploring Mental Health Data

Playground Series - Season 4, Episode 11

**Page of the competition:** [link](https://www.kaggle.com/competitions/playground-series-s4e11)

**Ranked:** 438/2685 (top 17%)

| Commit Message                                                | Public Score | Private Score |
|---------------------------------------------------------------|--------------|---------------|
| `1e64c2da(mental-health --booster-ensemble --stratification)` | 0.94168      | 0.94071       |
| `b9de241d(mental-health --light-lgbm)`                        | 0.94131      | 0.94011       |


### Regression with an Insurance Dataset

Playground Series - Season 4, Episode 12

**Page of the competition:** [link](https://www.kaggle.com/competitions/playground-series-s4e12/overview)

**Ranked:** 449/2390 (top 19%)

| Commit Message                                              | Public Score | Private Score |
|-------------------------------------------------------------|--------------|---------------|
| `32e0a842(insurance --stacking-regressor --stratification)` | 1.04487      | 1.04665       |
| `07bf2a1d(insurance --light-lgbm --stratification)`         | 1.04487      | 1.04675       |


### Exploring Predictive Health Factors

**Page of the competition:** [link](https://www.kaggle.com/competitions/exploring-predictive-health-factors)

**Ranked:** 23/138 (top 17%)

| Commit Message                           | Public Score | Private Score |
|------------------------------------------|--------------|---------------|
| `74b12e68(health-factors --light-lgbm)`  | 0.916        | 0.796         |
| `c412e12e(health-factors --light-lgbm)`  | 0.913        | 0.798         |
