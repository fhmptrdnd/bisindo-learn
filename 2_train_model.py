import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Membaca dataset...")
df = pd.read_csv('dataset_isyarat.csv')

# label (y); fitur/koordinat (X)
X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FNN / MLP
print("Melatih Jaringan Syaraf Tiruan...")
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64), # 2 hidden layer: 128 dan 64
    activation='relu',            # fungsi aktivasi
    solver='adam',                # optimizer
    max_iter=1000,                # epoch
    random_state=42
)

clf.fit(X_train, y_train)

# evaluasi
y_pred = clf.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
print(f"Training Selesai! Akurasi Model: {akurasi * 100:.2f}%")

# simpan
joblib.dump(clf, 'model_isyarat.pkl')
print("Model berhasil disimpan sebagai 'model_isyarat.pkl'")