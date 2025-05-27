import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Spark session
spark = SparkSession.builder.appName("NLP-NB-Streamlit").getOrCreate()

# Load dataset
df = spark.read.csv("SMSSpamCollection", sep="\t", inferSchema=True)
df = df.withColumnRenamed("_c0", "class").withColumnRenamed("_c1", "text")
df = df.withColumn("length", length(df["text"]))

# Define pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stop_word_remover = StopWordsRemover(inputCol="token_text", outputCol="stop_tokens")
count_vec = CountVectorizer(inputCol="stop_tokens", outputCol="c_vec")
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer(inputCol="class", outputCol="label")
cleaned = VectorAssembler(inputCols=["tf_idf", "length"], outputCol="features")

data_prep_pipe = Pipeline(stages=[ham_spam_to_num, tokenizer, stop_word_remover, count_vec, idf, cleaned])
fitted_pipeline = data_prep_pipe.fit(df)
final_data = fitted_pipeline.transform(df).select("label", "features")

# Train/test split and model training
train, test = final_data.randomSplit([0.7, 0.3], seed=42)
nb = NaiveBayes()
spam_detector = nb.fit(train)
results = spam_detector.transform(test)

# Evaluation
evaluator = MulticlassClassificationEvaluator()
accuracy = evaluator.evaluate(results, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(results, {evaluator.metricName: "f1"})
precision = evaluator.evaluate(results, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(results, {evaluator.metricName: "weightedRecall"})

# Confusion Matrix
y_true = results.select("label").rdd.flatMap(lambda x: x).collect()
y_pred = results.select("prediction").rdd.flatMap(lambda x: x).collect()
cm = pd.crosstab(pd.Series(y_true, name='Actual'), pd.Series(y_pred, name='Predicted'))

# Streamlit UI
st.title("📩 SMS Spam Detection")
st.write("Enter an SMS message below to predict whether it's Spam or Not Spam.")

# Show Metrics
st.markdown("### 📊 Model Evaluation Metrics")
st.markdown(f"- **Accuracy**: `{accuracy * 100:.2f}%`")
st.markdown(f"- **Precision**: `{precision * 100:.2f}%`")
st.markdown(f"- **Recall**: `{recall * 100:.2f}%`")
st.markdown(f"- **F1-Score**: `{f1 * 100:.2f}%`")

# Confusion Matrix with expander (tooltip-like behavior)
with st.expander("🧩 View Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# User Input for Prediction
user_input = st.text_area("✍️ Enter your message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        user_df = spark.createDataFrame([(user_input,)], ["text"])
        user_df = user_df.withColumn("length", length(user_df["text"]))
        user_cleaned = fitted_pipeline.transform(user_df).select("features")
        user_prediction = spam_detector.transform(user_cleaned)
        prediction_result = user_prediction.select("prediction").collect()[0][0]

        if prediction_result == 0.0:
            st.success("✅ Prediction: HAM (Not Spam)")
        else:
            st.error("🚫 Prediction: SPAM")
