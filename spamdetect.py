import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from wordcloud import WordCloud

# Read data
df = pd.read_csv("spam.csv", encoding='latin-1')

# Drop unnamed columns
df = df.drop('Unnamed: 2', axis=1)
df = df.drop('Unnamed: 3', axis=1)
df = df.drop('Unnamed: 4', axis=1)

# First rows of data
print("First few rows of the data set:")
print(df.head())

# Wordcloud
df['cleaned_text'] = df['text'].str.replace('[^a-zA-Z\s]', '').str.lower()
text_messages = ' '.join(df['cleaned_text'].dropna())
wordcloud = WordCloud(width=800, height=400).generate(text_messages)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud')
plt.show()

# Histogram
df['message_length'] = df['text'].apply(len)
plt.hist(df[df['label'] == 'spam']['message_length'], bins=30, alpha=0.5, label='Spam')
plt.hist(df[df['label'] == 'ham']['message_length'], bins=30, alpha=0.5, label='Not spam')
plt.xlabel('Message length')
plt.ylabel('Count')
plt.legend()
plt.title('Histogram of message length')
plt.show()

# TF-IDF Visualization
tfidf_vectorizer = TfidfVectorizer()
tfidf_features = tfidf_vectorizer.fit_transform(df['text'])
pca = PCA(n_components=2)
tfidf_pca_result = pca.fit_transform(tfidf_features.toarray())
plt.scatter(tfidf_pca_result[:, 0], tfidf_pca_result[:, 1], c=df['label'].apply(lambda x: 1 if x == 'spam' else 0))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('TF-IDF Visualization')
plt.show()

# Train model
df = df[['label', 'text']]
df = df.rename(columns={'label': 'label', 'text': 'message'})

df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

y_pred = svm_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification report:\n", classification_report(y_test, y_pred))

# Input
def predict_spam(message):
    message_tfidf = tfidf_vectorizer.transform([message])
    prediction = svm_classifier.predict(message_tfidf)
    return "spam" if prediction[0] == 1 else "not spam"


user_message = input("Enter an SMS message: ")
prediction = predict_spam(user_message)
print ("Prediction: this message is", prediction)