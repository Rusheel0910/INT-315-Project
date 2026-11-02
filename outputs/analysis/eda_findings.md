# Exploratory Data Analysis Findings

The cleaned dataset contains 4,424 records and 37 columns.

## Target and Scholarship Label

The available `Target` column records student outcomes, not scholarship eligibility.
`Scholarship holder` is a recorded holder-status field and is not treated as an eligibility decision or used to create a target.

Observed `Target` distribution: Graduate: 2,209 (49.9%); Dropout: 1,421 (32.1%); Enrolled: 794 (17.9%).
Largest-to-smallest outcome class count ratio: 2.78.

## Academic Performance

Spark-computed mean and median grade summaries by student outcome are in `academic_performance_by_outcome.csv`; the box plot uses the second-semester grade where available.
These are descriptive associations and do not establish causation.

## Economic Features

The dataset contains these economic indicators: Unemployment rate, Inflation rate, GDP. They are contextual indicators, not student family-income measurements.
Group averages by student outcome are in `economic_indicators_by_outcome.csv`.

No family-income column is present, so family-income analysis and its requested plot are not applicable.

## Data Quality and Figures

Detailed missing-value and duplicate counts are recorded in `cleaning_summary.json`. Exact duplicates were retained because the file contains no student identifier for judging whether repeated rows represent the same student.
Generated figures: target_distribution.png, numerical_histograms.png, numerical_correlation.png, academic_performance_by_outcome.png, scholarship_holder_by_outcome.png.

A genuine scholarship-eligibility model requires an approved eligibility label. No derived label was created in this analysis.
