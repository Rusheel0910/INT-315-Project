# Power BI Dashboard Guide

## Status and Scope

Power BI Desktop is installed, but this VS Code session cannot operate its native interface. The PBIX has not been created or tested. Follow the steps below in Power BI Desktop and save the report as `powerbi/Scholarship_Dashboard.pbix`.

The model predicts `Target` student outcomes (`Dropout`, `Graduate`, `Enrolled`), not scholarship eligibility. Use the title **Student Outcome Classification Evaluation** and include a visible note: **Predicted label is student outcome, not scholarship eligibility.**

## Import the CSVs

From the project root, choose **Home > Get data > Text/CSV** and import each file. In the preview, confirm comma delimiter and **UTF-8** file origin, then select **Load**. Rename each query/table exactly as shown:

| File | Table name | Rows | Fields |
|---|---|---:|---|
| `outputs/predictions/test_predictions.csv` | `TestPredictions` | 829 | 41: 36 source features plus class indices, labels, and confidence |
| `outputs/metrics/evaluation_metrics.csv` | `EvaluationMetrics` | 5 | `metric`, `value` |
| `outputs/metrics/per_class_metrics.csv` | `PerClassMetrics` | 3 | `outcome_class`, `precision`, `recall`, `f1`, `support` |
| `outputs/metrics/confusion_matrix.csv` | `ConfusionMatrix` | 9 | `actual_label`, `predicted_label`, `records` |
| `outputs/metrics/actual_vs_predicted.csv` | `OutcomeComparison` | 3 | `outcome`, `actual_records`, `predicted_records` |
| `outputs/metrics/outcome_distribution.csv` | `OutcomeDistribution` | 6 | `label_type`, `outcome`, `records` |

In Power Query, set `value`, class metrics, `prediction_confidence`, academic grades and `records` to Decimal Number or Whole Number as appropriate; labels and categories should be Text. Select **Close & Apply**. No relationships are needed because the measures and chart data are table-local. The `TestPredictions` table is the held-out test set; do not describe it as the full-dataset distribution.

## DAX Measures

Create these measures in `EvaluationMetrics` using **Modeling > New measure**:

```DAX
Total Evaluated Students =
CALCULATE(
    MAX(EvaluationMetrics[value]),
    EvaluationMetrics[metric] = "evaluated_records"
)

Model Accuracy =
CALCULATE(
    MAX(EvaluationMetrics[value]),
    EvaluationMetrics[metric] = "accuracy"
)

Weighted Precision =
CALCULATE(
    MAX(EvaluationMetrics[value]),
    EvaluationMetrics[metric] = "weighted_precision"
)

Weighted Recall =
CALCULATE(
    MAX(EvaluationMetrics[value]),
    EvaluationMetrics[metric] = "weighted_recall"
)

Weighted F1 =
CALCULATE(
    MAX(EvaluationMetrics[value]),
    EvaluationMetrics[metric] = "weighted_f1"
)
```

Format `Model Accuracy`, `Weighted Precision`, `Weighted Recall`, and `Weighted F1` as percentages with one decimal place. These measures report the full held-out test evaluation and should not be presented as eligibility metrics.

## Single-Page Layout

Set the page to 16:9. Use a white background, charcoal headings, teal for actual counts, muted amber for predicted counts, and a restrained red/orange conditional format for errors. Use one font consistently (Aptos is suitable), titles around 16–20 pt, labels around 10–12 pt. Arrange a short title and scope note at top, five KPI cards in one row, two main analytical visuals in the middle, and the remaining charts/slicers in a compact bottom band. Avoid decorative shapes that compete with data.

### KPI Cards

Add five Card visuals, each using one measure as the card value:

1. `Total Evaluated Students`
2. `Model Accuracy`
3. `Weighted Precision`
4. `Weighted Recall`
5. `Weighted F1`

Use those exact labels. Do not use cards such as “eligible students” or “predicted scholarship awards.” The exported evaluated test count is 829.

### Visuals and Field Wells

1. **Actual vs Predicted Student Outcomes**, Clustered column chart. X-axis: `OutcomeComparison[outcome]`. Y-axis / Values: `OutcomeComparison[actual_records]` and `OutcomeComparison[predicted_records]`. Legend: the two value series. Use teal for actual, amber for predicted.
2. **Confusion Matrix**, Matrix visual. Rows: `ConfusionMatrix[actual_label]`. Columns: `ConfusionMatrix[predicted_label]`. Values: Sum of `ConfusionMatrix[records]`. Apply background conditional formatting to Values to make off-diagonal errors easy to notice. The nine exported cells include zero-count combinations and total 829.
3. **Student Outcome Distribution (Test Set)**, Donut chart. Legend: `OutcomeDistribution[outcome]`. Values: Sum of `OutcomeDistribution[records]`. Visual-level filter: `OutcomeDistribution[label_type]` is `Actual`. Optionally add a second small donut for `Predicted`; keep the chart title explicit.
4. **Precision and Recall by Outcome**, Clustered bar chart. Y-axis: `PerClassMetrics[outcome_class]`. X-axis / Values: `PerClassMetrics[precision]` and `PerClassMetrics[recall]`. Format the axis as percentage; keep the two series in distinct colors.
5. **Academic Performance by Actual Outcome**, Clustered bar chart. Y-axis: `TestPredictions[actual_label]`. X-axis / Values: Average of `TestPredictions[Curricular units 2nd sem (grade)]`. This compares a real academic feature on the held-out test records. It is descriptive, not causal.

### Slicers

Add compact dropdown slicers using `TestPredictions[Gender]`, `TestPredictions[Course]`, and `TestPredictions[Scholarship holder]`. Optionally add `TestPredictions[Age at enrollment]` as a numeric range slicer. Label the last categorical slicer **Recorded scholarship-holder status**, not eligibility. These slicers filter row-based test feature charts; the precomputed overall KPI metrics remain full-test-set metrics and are not recalculated per slicer selection.

## Save and Validate

Save the report as `powerbi/Scholarship_Dashboard.pbix`. Confirm that `TestPredictions` has 829 rows, `ConfusionMatrix` has nine rows totaling 829, and the five cards show 829, 74.8%, 76.0%, 74.8%, and 75.0% respectively. Save genuine screenshots of the completed Power BI page in `screenshots/`. The existing PNG files in `diagrams/` are EDA figures, not Power BI screenshots.
