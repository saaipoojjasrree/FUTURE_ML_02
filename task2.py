import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


data = {
    "text": [
        "My internet is not working",
        "Payment failed but money deducted",
        "App is crashing frequently",
        "Need help with account login",
        "Feature request for dark mode",
        "Unable to reset password",
        "Website is very slow",
        "Billing issue not resolved"
    ],
    "category": [
        "Technical",
        "Billing",
        "Technical",
        "Account",
        "Feature",
        "Account",
        "Technical",
        "Billing"
    ]
}

df = pd.DataFrame(data)


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["category"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = MultinomialNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

sample = ["I cannot login to my account"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

print("Predicted category:", prediction[0])