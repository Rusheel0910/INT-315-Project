from pathlib import Path

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, functions as F

from data_cleaning import TARGET_COLUMN
from export_utils import write_spark_csv


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = ROOT / "outputs" / "predictions" / "test_predictions.csv"
METRICS_PATH = ROOT / "outputs" / "metrics"


def main() -> None:
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError("Run src/train_model.py before model evaluation.")

    spark = (
        SparkSession.builder.appName("INT315 Student Outcome Evaluation")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        predictions = (
            spark.read.option("header", "true")
            .option("sep", ",")
            .option("encoding", "UTF-8")
            .option("inferSchema", "true")
            .csv(str(PREDICTIONS_PATH))
        )
        required = {
            "actual_label",
            "predicted_label",
            "actual_class_index",
            "predicted_class_index",
        }
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"Prediction export missing required columns: {sorted(missing)}")

        evaluator = MulticlassClassificationEvaluator(
            labelCol="actual_class_index",
            predictionCol="predicted_class_index",
        )
        metric_values = {
            "accuracy": evaluator.setMetricName("accuracy").evaluate(predictions),
            "weighted_precision": evaluator.setMetricName("weightedPrecision").evaluate(predictions),
            "weighted_recall": evaluator.setMetricName("weightedRecall").evaluate(predictions),
            "weighted_f1": evaluator.setMetricName("f1").evaluate(predictions),
        }
        matrix = (
            predictions.groupBy("actual_label", "predicted_label")
            .count()
            .withColumnRenamed("count", "records")
        )
        classes = sorted(
            row[0]
            for row in predictions.select("actual_label")
            .union(predictions.select("predicted_label"))
            .distinct()
            .collect()
        )
        actual_total = predictions.count()
        support_by_class = {
            row["actual_label"]: row["records"]
            for row in predictions.groupBy("actual_label")
            .count()
            .withColumnRenamed("count", "records")
            .collect()
        }
        predicted_by_class = {
            row["predicted_label"]: row["records"]
            for row in predictions.groupBy("predicted_label")
            .count()
            .withColumnRenamed("count", "records")
            .collect()
        }
        confusion_counts = {
            (row["actual_label"], row["predicted_label"]): int(row["records"])
            for row in matrix.collect()
        }
        true_positive = {
            row["actual_label"]: row["records"]
            for row in matrix.where(F.col("actual_label") == F.col("predicted_label"))
            .select("actual_label", "records")
            .collect()
        }

        class_rows = []
        for label in classes:
            tp = int(true_positive.get(label, 0))
            support = int(support_by_class.get(label, 0))
            predicted = int(predicted_by_class.get(label, 0))
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            class_rows.append((label, precision, recall, f1, support))

        metric_rows = [
            ("evaluated_records", float(actual_total)),
            ("accuracy", metric_values["accuracy"]),
            ("weighted_precision", metric_values["weighted_precision"]),
            ("weighted_recall", metric_values["weighted_recall"]),
            ("weighted_f1", metric_values["weighted_f1"]),
        ]
        METRICS_PATH.mkdir(parents=True, exist_ok=True)
        metrics_df = spark.createDataFrame(metric_rows, ["metric", "value"])
        classes_df = spark.createDataFrame(
            class_rows,
            ["outcome_class", "precision", "recall", "f1", "support"],
        )
        write_spark_csv(metrics_df, METRICS_PATH / "evaluation_metrics.csv")
        write_spark_csv(classes_df, METRICS_PATH / "per_class_metrics.csv")
        full_matrix_df = spark.createDataFrame(
            [
                (actual, predicted, confusion_counts.get((actual, predicted), 0))
                for actual in classes
                for predicted in classes
            ],
            ["actual_label", "predicted_label", "records"],
        )
        write_spark_csv(full_matrix_df, METRICS_PATH / "confusion_matrix.csv")
        comparison_rows = [
            (label, int(support_by_class.get(label, 0)), int(predicted_by_class.get(label, 0)))
            for label in classes
        ]
        write_spark_csv(
            spark.createDataFrame(
                comparison_rows,
                ["outcome", "actual_records", "predicted_records"],
            ),
            METRICS_PATH / "actual_vs_predicted.csv",
        )
        distribution_rows = [
            (kind, label, count)
            for kind, counts in (
                ("Actual", support_by_class),
                ("Predicted", predicted_by_class),
            )
            for label, count in counts.items()
        ]
        write_spark_csv(
            spark.createDataFrame(
                distribution_rows,
                ["label_type", "outcome", "records"],
            ),
            METRICS_PATH / "outcome_distribution.csv",
        )

        print(f"Evaluation target: {TARGET_COLUMN} (student outcome; not scholarship eligibility)")
        print(f"Evaluated test records: {actual_total}")
        for name, value in metric_values.items():
            print(f"{name}: {value:.4f}")
        print("Per-class metrics (class, precision, recall, F1, support):")
        for row in class_rows:
            print(f"{row[0]}, {row[1]:.4f}, {row[2]:.4f}, {row[3]:.4f}, {row[4]}")
        print(f"Confusion matrix CSV: {METRICS_PATH / 'confusion_matrix.csv'}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
