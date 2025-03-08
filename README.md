# Predicting House Prices with PySpark Regression Model

## Overview
This project utilizes Apache Spark ML to build a regression model in PySpark for predicting house prices based on the California Housing Prices dataset. The implementation includes data preprocessing, feature selection, model training, and performance evaluation.

## Dataset
- **Source:** [California Housing Prices Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Target Variable:** `median_house_value`
- **Features:**
  - `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`

## Requirements
The following Python libraries are required:
```python
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
```

## Implementation Steps

### 1. Initialize Spark Session
```python
spark = SparkSession.builder.appName('df-project').getOrCreate()
```

### 2. Load Data
```python
csv = spark.read.csv("/content/housing.csv", inferSchema=True, header=True)
csv.show()
```

### 3. Preprocessing
- Rename target variable to `label`
```python
csv = csv.withColumnRenamed("median_house_value", "label")
```
- Handle missing values
```python
imputer = Imputer(inputCols=['total_bedrooms'], outputCols=['total_bedrooms']).setStrategy("mean")
```
- Encode categorical features
```python
indexer = StringIndexer(inputCol="ocean_proximity", outputCol="ocean_proximity_index")
encoder = OneHotEncoder(inputCol="ocean_proximity_index", outputCol="ocean_proximity_encoded")
```

### 4. Feature Engineering
```python
feature_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                   'total_bedrooms', 'population', 'households', 'median_income',
                   'ocean_proximity_encoded']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
```

### 5. Model Training
Using Gradient-Boosted Tree Regression:
```python
gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=50)
```
Create pipeline:
```python
pipeline = Pipeline(stages=[imputer, indexer, encoder, assembler, gbt])
```

### 6. Split Data and Train Model
```python
train, test = csv.randomSplit([0.7, 0.3])
test = test.withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)
```

Train the model:
```python
pipeline_model = pipeline.fit(train)
```

### 7. Make Predictions
```python
prediction = pipeline_model.transform(test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show()
```

### 8. Evaluate Model Performance
Calculate RMSE (Root Mean Square Error):
```python
evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print("Root Mean Square Error (RMSE):", rmse)
```

### 9. Visualization
```python
df = predicted.toPandas()
df.plot.scatter(x='trueLabel', y='prediction', alpha=0.5)
```

## Conclusion
This project demonstrates how to implement a regression model using Apache Spark ML for predicting house prices. The pipeline efficiently handles missing values, encodes categorical data, and uses a gradient-boosted tree model for prediction. Future improvements could include hyperparameter tuning and trying alternative regression models.

## Author
**Ramazan Denli**

