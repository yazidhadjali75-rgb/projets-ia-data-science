
from sklearn.feature_extraction.text import CountVectorizer
import exercice1 as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

print(df.head())
print(df["label"].value_counts())
vectorizer= CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]
