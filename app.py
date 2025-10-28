from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = Flask(__name__)

# === Load Model dan Data ===
model = joblib.load('decision_tree_zoo.pkl')
zoo = pd.read_csv('zoo.csv')
classes = pd.read_csv('class.csv')

# Gabung dataset
df = zoo.merge(classes, left_on='class_type', right_on='Class_Number', how='left')
df.drop(['Class_Number', 'Number_Of_Animal_Species_In_Class', 'Animal_Names'], axis=1, inplace=True)
df.rename(columns={'Class_Type': 'animal_class'}, inplace=True)

X = df.drop(columns=['animal_name', 'animal_class', 'class_type'])
y = df['animal_class']

# Split untuk evaluasi ROC dan Confusion Matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === CONFUSION MATRIX ===
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title('Confusion Matrix - Decision Tree Zoo')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
plt.close()

# === ROC Curve Multi-Class ===
y_bin = label_binarize(y, classes=sorted(y.unique()))
n_classes = y_bin.shape[1]

clf = OneVsRestClassifier(DecisionTreeClassifier(max_depth=4, random_state=42))
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.3, random_state=42)
clf.fit(X_train_bin, y_train_bin)
y_score = clf.predict_proba(X_test_bin)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
roc_auc["macro"] = auc(all_fpr, mean_tpr)

plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
plt.plot(all_fpr, mean_tpr, color='navy',
         label='ROC macro-average (AUC = {:.2f})'.format(roc_auc["macro"]),
         linewidth=3)
colors = sns.color_palette("husl", n_classes)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.8, alpha=0.8,
             label='{} (AUC = {:.2f})'.format(sorted(y.unique())[i], roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Kurva ROC Multi-class - Decision Tree Zoo')
plt.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('static/roc_curve.png')
plt.close()

# === GRAFIK LAIN (distribusi, feature importance, tree) ===
plt.figure(figsize=(8, 5))
sns.countplot(x='animal_class', data=df, palette='viridis')
plt.title('Distribusi Jenis Hewan')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('static/distribusi.png')
plt.close()

feature_importance = pd.DataFrame({
    'Fitur': X.columns,
    'Pentingnya': model.feature_importances_
}).sort_values(by='Pentingnya', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Pentingnya', y='Fitur', data=feature_importance, palette='mako')
plt.title('Peringkat Pentingnya Fitur (Feature Importance)')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()

plt.figure(figsize=(25, 12))
plot_tree(model,
          feature_names=X.columns,
          class_names=sorted(df['animal_class'].unique()),
          filled=True, rounded=True)
plt.title("Visualisasi Pohon Keputusan (Overfit)")
plt.tight_layout()
plt.savefig('static/tree_plot.png')
plt.close()

# === ROUTES ===
@app.route('/')
def home():
    return render_template('index.html', features=X.columns)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [int(request.form.get(feat, 0)) for feat in X.columns]
    df_input = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(df_input)[0]
    return render_template('index.html', features=X.columns, result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
