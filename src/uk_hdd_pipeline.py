from pyspark import pipelines as dp
from pyspark.sql import functions as F


# 1. BRONZE LAYER: Streaming ingestion from Unity Catalog Volume via Auto Loader
@dp.table(
    name="bronze_ecmwf_raw",
    comment="Raw 00z and 12z ECMWF 24-hour temperature drops for London, Birmingham, and Glasgow",
)
def bronze_ecmwf_raw():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .load("/Volumes/workspace/uk_hdd/ecmwf_drops/")
        .withColumn("ingested_at_utc", F.current_timestamp())
        .withColumn("source_file", F.col("_metadata.file_path"))
    )


# 2. SILVER LAYER: Physics-validated Delta table with Liquid Clustering & Iceberg UniForm
@dp.table(
    name="silver_uk_temp",
    comment="Validated UK city temperatures with hourly Heating Degree Days (HDD) and Iceberg UniForm",
    cluster_by=["city", "forecast_cycle"],
    table_properties={
        "delta.columnMapping.mode": "name",
        "delta.enableIcebergCompatV2": "true",
        "delta.universalFormat.enabledFormats": "iceberg",
    },
)
@dp.expect_or_drop("valid_uk_temperature", "temp_2m_c BETWEEN -40.0 AND 45.0")
@dp.expect_or_drop("valid_forecast_hour", "valid_hour BETWEEN 0 AND 23")
@dp.expect_or_drop("valid_city", "city IN ('London', 'Birmingham', 'Glasgow')")
def silver_uk_temp():
    return (
        spark.read.table("bronze_ecmwf_raw")
        .dropDuplicates(["forecast_cycle", "station_id", "valid_hour"])
        .withColumn(
            "hdd_hourly",
            F.round(F.greatest(F.lit(0.0), F.lit(15.5) - F.col("temp_2m_c")) / F.lit(24.0), 4),
        )
    )


# 3. GOLD LAYER: Aggregated 24h HDD & 00z vs. 12z Revision Shifts per UK City
@dp.table(
    name="gold_uk_hdd_shifts",
    comment="24-hour total HDD per UK city comparing 00z baseline against 12z forecast revision",
)
def gold_uk_hdd_shifts():
    silver = spark.read.table("silver_uk_temp")

    agg = silver.groupBy("station_id", "city", "forecast_cycle").agg(
        F.round(F.min("temp_2m_c"), 2).alias("min_temp_c"),
        F.round(F.avg("temp_2m_c"), 2).alias("avg_temp_c"),
        F.round(F.sum("hdd_hourly"), 2).alias("total_24h_hdd"),
    )

    run_00z = agg.filter(F.col("forecast_cycle") == "00z").select(
        "station_id",
        "city",
        F.col("avg_temp_c").alias("avg_temp_00z"),
        F.col("total_24h_hdd").alias("hdd_00z"),
    )

    run_12z = agg.filter(F.col("forecast_cycle") == "12z").select(
        "station_id",
        F.col("avg_temp_c").alias("avg_temp_12z"),
        F.col("total_24h_hdd").alias("hdd_12z"),
    )

    return (
        run_00z.join(run_12z, on="station_id", how="inner")
        .withColumn("hdd_shift_delta", F.round(F.col("hdd_12z") - F.col("hdd_00z"), 2))
        .withColumn(
            "trading_signal",
            F.when(F.col("hdd_shift_delta") >= 2.0, "HIGH_GAS_DEMAND_SURGE")
            .when(F.col("hdd_shift_delta") > 0.0, "MODERATE_HEATING_INCREASE")
            .otherwise("NORMAL_DEMAND"),
        )
    )