from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import IndexToString, StringIndexerModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F

from data_cleaning import TARGET_COLUMN
from export_utils import write_spark_csv
from preprocessing import build_preprocessing_pipeline, prepare_splits


ROOT = Path(__file__).resolve().parents[1]
CLEANED_PATH = ROOT / "outputs" / "cleaned_data" / "students.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "predictions" / "test_predictions.csv"


def main() -> None:
    if not CLEANED_PATH.is_file():
        raise FileNotFoundError("Run src/data_cleaning.py before model training.")

    spark = (
        SparkSession.builder.appName("INT315 Student Outcome Decision Tree")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        dataframe = (
            spark.read.option("header", "true")
            .option("sep", ",")
            .option("encoding", "UTF-8")
            .option("inferSchema", "true")
            .csv(str(CLEANED_PATH))
        )
        training, testing, continuous, categorical, features = prepare_splits(dataframe)
        prep_stages, feature_col = build_preprocessing_pipeline(continuous, categorical)
        classifier = DecisionTreeClassifier(
            labelCol="label",
            featuresCol=feature_col,
            predictionCol="prediction",
            probabilityCol="probability",
            maxDepth=5,
            seed=315,
        )
        pipeline = Pipeline(stages=prep_stages + [classifier])
        fitted = pipeline.fit(training)
        outcome_indexer = next(
            stage
            for stage in fitted.stages
            if isinstance(stage, StringIndexerModel) and stage.getOutputCol() == "label"
        )
        predictions = fitted.transform(testing)
        predictions = IndexToString(
            inputCol="prediction",
            outputCol="predicted_label",
            labels=outcome_indexer.labels,
        ).transform(predictions)
        predictions = predictions.withColumnRenamed(TARGET_COLUMN, "actual_label")
        probabilities = vector_to_array("probability")
        predictions = predictions.withColumn(
            "prediction_confidence",
            F.array_max(probabilities),
        ).select(
            *features,
            F.col("label").alias("actual_class_index"),
            F.col("prediction").alias("predicted_class_index"),
            "actual_label",
            "predicted_label",
            "prediction_confidence",
        )

        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_spark_csv(predictions, PREDICTIONS_PATH)
        print("Classification target: Target (student outcome; not scholarship eligibility)")
        print(f"Training records: {training.count()}")
        print(f"Test records: {testing.count()}")
        print(f"Features: {len(features)} ({len(continuous)} continuous, {len(categorical)} categorical)")
        print(f"Outcome classes: {', '.join(outcome_indexer.labels)}")
        print(f"Decision tree maxDepth: {classifier.getMaxDepth()}")
        print(f"Prediction export: {PREDICTIONS_PATH}")
        print(f"Exported prediction rows: {predictions.count()}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
