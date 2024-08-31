# pip install nltk spacy transformers pandas scikit-learn

# Loading the dataset
import pandas as pd

data = pd.read_csv('IMDB Dataset.csv')

# Tokenization
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

data['tokens'] = data['review'].apply(word_tokenize)

# Stop Words Removal
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
data['tokens'] = data['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

# Reducing words to their base or root form
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
data['lemmas'] = data['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Converting all words to lowercase to ensure uniformit
data['processed_text'] = data['lemmas'].apply(lambda x: ' '.join([word.lower() for word in x]))

# Spliting the dataset into training (80%) and testing (20%) sets using Scikit-Learn
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['processed_text'], data['sentiment'], test_size=0.2, random_state=42)

# Feature Extraction: Converting the processed text into numerical features using techniques like TF-IDF.
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training a sentiment analysis model using a simple classifier like Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predicting the test set results and evaluate the model using accuracy, precision, recall, and F1-score.
from sklearn.metrics import classification_report

y_pred = model.predict(X_test_tfidf)
report = classification_report(y_test, y_pred)
print(report)

# Saving the Preprocessed Dataset
data.to_csv('preprocessed_dataset.csv', index=False)


