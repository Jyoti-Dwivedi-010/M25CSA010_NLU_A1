
# run this in terminal: pip install scikit-learn pandas requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#data loading
# i used expanded categories in both sports and politics data 
categories = [
    'rec.sport.baseball', 'rec.sport.hockey',
    'talk.politics.mideast', 'talk.politics.misc',
    'talk.politics.guns', 'talk.religion.misc'
]

print("Fetching 20 Newsgroups with Expanded Categories")
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# Merging specific newsgroups into broader "Sport" and "Social_Politics" classes
y_mapped = np.array(['Sport' if 'sport' in data.target_names[t] else 'Social_Politics' for t in data.target])

# Feature engineering using TF-IDF with Bigrams
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(data.data)
X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.2, random_state=42)

# visualization code
def save_performance_plot(results):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
    plt.ylim(0.80, 1.0)
    plt.title('Quantitative Model Comparison (Accuracy)', fontsize=14)
    plt.ylabel('Accuracy')
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')
    plt.savefig('performance_comparison.png')
    print("Saved: performance_comparison.png")

def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Politics', 'Sport'], yticklabels=['Politics', 'Sport'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'cm_{model_name.replace(" ", "_")}.png')
    print(f"Saved: cm_{model_name.replace(' ', '_')}.png")

def save_top_keywords(vectorizer, clf, n=10):
    feature_names = vectorizer.get_feature_names_out()
    # Log-probability difference: Class 1 (Sport) minus Class 0 (Social_Politics)
    coefs = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]

    # Generating SPORT Keywords i.e. those with the Highest positive values
    top_indices_sport = np.argsort(coefs)[-n:]
    words_sport = [feature_names[i] for i in top_indices_sport]
    weights_sport = coefs[top_indices_sport]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=weights_sport, y=words_sport, palette='magma')
    plt.title(f'Top {n} Most Predictive Keywords for SPORT', fontsize=14)
    plt.xlabel('Log-Probability Weight')
    plt.tight_layout()
    plt.savefig('top_keywords_sport.png')
    plt.close()
    print("Saved: top_keywords_sport.png")

    # Generating POLITICS Keywords i.e. those with the lowest negative values
    top_indices_pol = np.argsort(coefs)[:n]
    words_pol = [feature_names[i] for i in top_indices_pol]
    weights_pol = np.abs(coefs[top_indices_pol]) 

    plt.figure(figsize=(10, 8))
    sns.barplot(x=weights_pol, y=words_pol, palette='viridis')
    plt.title(f'Top {n} Most Predictive Keywords for POLITICS', fontsize=14)
    plt.xlabel('Log-Probability Weight (Importance)')
    plt.tight_layout()
    plt.savefig('top_keywords_politics.png')
    plt.close()
    print("Saved: top_keywords_politics.png")

# model training on 3 different ML approaches i.e. SVM, Naive Bayes and Random Forest
models = {
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results_acc = {}
print("\n Quantitative Comparison")
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results_acc[name] = acc

    print(f"\nTechnique: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    save_confusion_matrix(y_test, y_pred, name)
    if name == "Naive Bayes":
        save_top_keywords(vectorizer, clf)

save_performance_plot(results_acc)

# understanding unigrams vs bigrams
print("\n Impact of N-grams")
vec_uni = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 1))
X_uni = vec_uni.fit_transform(data.data)
X_tr_u, X_ts_u, y_tr_u, y_ts_u = train_test_split(X_uni, y_mapped, test_size=0.2, random_state=42)
nb_uni = MultinomialNB().fit(X_tr_u, y_tr_u)
print(f"Accuracy with Unigrams only: {accuracy_score(y_ts_u, nb_uni.predict(X_ts_u)):.4f}")

# Testing with ambiguous test cases
print("\n Qualitative Test on Ambiguous Cases")
tricky_samples = [
    "The government is debating a new law regarding firearm safety in schools.",
    "The baseball team visited a historical cathedral during their away game.",
    "A peaceful protest was organized to support the rights of minor league players.",
    "The court ruled that the religious symbols must be removed from the stadium."
]
sample_vec = vectorizer.transform(tricky_samples)
sample_preds = models["Linear SVM"].predict(sample_vec)

for text, pred in zip(tricky_samples, sample_preds):
    print(f"Text: {text}\nPredicted: {pred}\n")