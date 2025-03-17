import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("IMDB Dataset.csv")


nltk.download('stopwords')

stop_words = set(stopwords.words("english"))


def clean(text):
    text = text.lower()  
    text = re.sub(r"<.*?>", "", text)  
    text = re.sub(r"[^a-zA-Z]", " ", text) 
    text = " ".join([word for word in text.split() if word not in stop_words])  
    return text

df["cleaned_review"] = df["review"].apply(clean)


vectorizer = TfidfVectorizer(max_features=5000) 
X = vectorizer.fit_transform(df["cleaned_review"])  
y = df["sentiment"].map({"positive": 1, "negative": 0}) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train) 


y_pred = model.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred) 

print(f"Model accuracy: {accuracy * 100:.2f}%")


def guess_sentiment(new_comment):
    new_comment = clean(new_comment)
    new_comment_vectorized = vectorizer.transform([new_comment]) 
    guess = model.predict(new_comment_vectorized)[0] 
    return "Positive" if guess == 1 else "Negative"

while True:
    new_comment = input("Write your comment (press 'q' to quit): ")
    if new_comment.lower() == 'q':
        break
    print("Prediction:", guess_sentiment(new_comment))
