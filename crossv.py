import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv('ulasan_digitalent_labeled.csv')
print(df.columns)
print(df.head())

# Pra-pemrosesan teks (optional)
df['cleaned_ulasan'] = df['cleaned_ulasan'].astype(str).str.lower()

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_ulasan'])
y = df['label']

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
accuracies = []

for train_idx, test_idx in skf.split(X, y):
    print(f"\n====== Fold {fold} ======")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Model: Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print("Akurasi:", acc)
    print(classification_report(y_test, y_pred))

    # Simpan model
    model_filename = f'model_fold_{fold}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model Fold {fold} disimpan sebagai: {model_filename}")

    fold += 1

# Ringkasan hasil CV
average_accuracy = sum(accuracies) / len(accuracies)
print("\nRata-rata Akurasi:", average_accuracy)

# Visualisasi akurasi tiap fold
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), accuracies, marker='o', linestyle='-', color='b')
plt.title('Akurasi Tiap Fold')
plt.xlabel('Fold Ke-')
plt.ylabel('Akurasi')
plt.ylim(0, 1)
plt.grid(True)
plt.show()
