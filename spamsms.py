import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/content/drive/MyDrive/spam.csv", encoding='latin-1')
drop_col_list = ["Unnamed: 2","Unnamed: 3","Unnamed: 4"]
df = df.drop(df[drop_col_list], axis=1)
df.columns = ['v1', 'v2']
df.info()
df['v2'] = df['v2'].apply(lambda x: x.lower())
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['v2'])
X_train, X_test, y_train, y_test = train_test_split(X, df['v1'], test_size=0.2, random_state=42)
clf_nb = MultinomialNB().fit(X_train, y_train)
clf_lr = LogisticRegression().fit(X_train, y_train)
clf_svm = SVC().fit(X_train, y_train)
y_pred_nb = clf_nb.predict(X_test)
y_pred_lr = clf_lr.predict(X_test)
y_pred_svm = clf_svm.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
plt.figure(figsize=(12,8))
fig = sns.countplot(x= df["v1"], palette=['teal','coral'])
fig.set_title("Number of Spam and Ham")
fig.set_xlabel("Classes")
fig.set_ylabel("Number of Data points")
plt.show(fig)
