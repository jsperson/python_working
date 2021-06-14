# Read a csv file and set the headers
df = (spark.read
      .options(header=True)
      .csv("/home/repl/workspace/mnt/data_lake/landing/ratings.csv"))

df.show()

# Define the schema
schema = StructType([
  StructField("brand", StringType(), nullable=False),
  StructField("model", StringType(), nullable=False),
  StructField("absorption_rate", ByteType(), nullable=True),
  StructField("comfort", ByteType(), nullable=True)
])

better_df = (spark
             .read
             .options(header="true")
             # Pass the predefined schema to the Reader
             .schema(schema)
             .csv("/home/repl/workspace/mnt/data_lake/landing/ratings.csv"))
pprint(better_df.dtypes)

# Specify the option to drop invalid rows
ratings = (spark
           .read
           .options(header=True, mode="DROPMALFORMED")
           .csv("/home/repl/workspace/mnt/data_lake/landing/ratings_with_invalid_rows.csv"))
ratings.show()


print("BEFORE")
ratings.show()

print("AFTER")
# Replace nulls with arbitrary value on column subset
ratings = ratings.fillna(4, subset=["comfort"])
ratings.show()

from pyspark.sql.functions import col, when

# Add/relabel the column
categorized_ratings = ratings.withColumn(
    "comfort",
    # Express the condition in terms of column operations
    when(col("comfort") > 3, "sufficient").otherwise("insufficient"))

categorized_ratings.show()