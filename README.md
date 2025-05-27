SMS Spam Detection using Pyspark

In the era of digital communication, SMS continues to be a commonly used medium for communication, which unfortunately makes it a target for spam. This project addresses the need to classify SMS messages as spam or ham (not spam) using machine learning techniques. This project utilizes Apache Spark's MLlib with PySpark for scalable processing and Naive Bayes for classification, all integrated within a Streamlit application for real-time user interaction. The model is trained on the 'SMSSpamCollection' dataset, using techniques like tokenization, stop word removal, TF-IDF vectorization, and length feature engineering. The project not only demonstrates high accuracy but also presents an intuitive interface for end-users to interact with the model. The implementation emphasizes big data principles and real-time analytics using modern tools.

Dataset Collection:
The data used in this project is the publicly available SMS SpamCollection dataset from the UCI Machine Learning Repository. It contains 5,574 SMS messages, each labeled as either 'ham' (not spam) or 'spam'

Data Cleaning and Preprocessing 
Text data must be cleaned and normalized before being used for machine learning. The following preprocessing steps were employed: 

• Lowercasing: Converts all characters to lowercase to reduce redundancy. 

• Tokenization: Breaks SMS messages into individual words. 

• Stop Word Removal: Removes common English words that do not contribute meaningfully to classification.

• Message Length: Calculates the character length of each message as a numeric feature. 

• String Indexing: Converts 'spam' and 'ham' to numerical values: 'spam' → 1, 'ham' → 0. 

The goal of preprocessing is to transform unstructured text into structured numerical features suitable for machine learning.

Feature Engineering:
 
Feature engineering is a critical step in text classification. For this project, the following features were derived: 

• Token Vectors: Created using PySpark's CountVectorizer, converting text tokens into a vector of term frequencies. 

• TF-IDF Scores: Applied using IDF, assigning weights to tokens based on their relative importance.  
 
• Message Length: Added as an auxiliary feature to help differentiate short informal messages from longer structured spam. 

• Feature Vector Assembly: Combined the TF-IDF vector and the message length into a single features vector using VectorAssembler.

This multi-dimensional feature space enables the Naive Bayes classifier to learn both textual patterns and structural properties of the messages. 

Model Building and Training:
 
The model chosen for this project is Naive Bayes, implemented using PySpark MLlib. It is well-suited for high-dimensional text data due to its simplicity and strong performance in bag-of-words models. 

Dataset Link: https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip >> smsspamcollection.zip


