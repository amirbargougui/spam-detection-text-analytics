import os # gérer les fichiers et dossiers
import pandas as pd  # gérer les données
import numpy as np # faire des calculs
import matplotlib.pyplot as plt # afficher des graphiques
import seaborn as sns    

# outils
from sklearn.model_selection import train_test_split # séparer les données
from sklearn.feature_extraction.text import TfidfVectorizer # transformer texte → nombres
from sklearn.linear_model import LogisticRegression # modèle de classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc # évaluation

#output images
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Style des graphiques
plt.style.use('ggplot')



print("\n📦 Chargement des données...")

file_path = os.path.join(os.path.dirname(__file__), "spam.csv")

try:
    df = pd.read_csv(file_path, encoding='utf-8')
except:
    df = pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip')

if len(df.columns) == 1:
    df = df[df.columns[0]].str.split(',', n=1, expand=True)

df.columns = ['label', 'message']
df.dropna(inplace=True)

print("✅ Dataset prêt")



# Ajout une nouvelle colonne : longueur du message

df['length'] = df['message'].apply(len)

#  ANALYSE DES DONNÉES (EDA)
# 1. Distribution des classes (spam vs ham)
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=df)
plt.title("Distribution des messages : HAM vs SPAM")
plt.xlabel("Type de message")
plt.ylabel("Nombre de messages")
plt.savefig(os.path.join(output_dir, "fig1_distribution.png"), dpi=300)
plt.close()

# 2.                 la longueur des messages

sns.histplot(data=df, x='length', hue='label', bins=30)
plt.title("Distribution de la longueur des messages")
plt.xlabel("Nombre de caractères")
plt.ylabel("Fréquence")
plt.savefig(os.path.join(output_dir, "fig2_length.png"), dpi=300)
plt.close()

# 3. Comparaison avec boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x='label', y='length', data=df)
plt.title("Comparaison de la longueur : HAM vs SPAM")
plt.xlabel("Type de message")
plt.ylabel("Longueur")
plt.savefig(os.path.join(output_dir, "fig3_boxplot.png"), dpi=300)
plt.close()


#.MODÈLE 

X = df["message"]
y = (df["label"] == "spam").astype(int)

# Division données en train/test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Transformer le texte en vecteurs numériques (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)

# Apprentissage sur train
X_train_vec = vectorizer.fit_transform(X_train)

# Transformation du test
X_test_vec = vectorizer.transform(X_test)

# Création du modèle
model = LogisticRegression(max_iter=1000)

# Entraînement du modèle
model.fit(X_train_vec, y_train)


# . ÉVALUATION

# Prédiction des classes
# Prédiction des probabilités
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)[:,1]

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# . VISUALISATION DES RÉSULTATS

# . Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['HAM','SPAM'],
            yticklabels=['HAM','SPAM'])
plt.title("Matrice de confusion du modèle")
plt.xlabel("Prédiction")
plt.ylabel("Valeur réelle")
plt.savefig(os.path.join(output_dir, "fig4_confusion.png"), dpi=300)
plt.close()

# . courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.title("Courbe ROC du modèle")
plt.xlabel("Faux positifs (FPR)")
plt.ylabel("Vrais positifs (TPR)")
plt.legend()
plt.savefig(os.path.join(output_dir, "fig5_roc.png"), dpi=300)
plt.close()

# . Mots importants
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]

top_spam = np.argsort(coefs)[-10:]
top_ham = np.argsort(coefs)[:10]

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.barh(feature_names[top_spam], coefs[top_spam])
plt.title("Mots associés au SPAM")

plt.subplot(1,2,2)
plt.barh(feature_names[top_ham], coefs[top_ham])
plt.title("Mots associés au HAM")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig6_words.png"), dpi=300)
plt.close()

# . TEST


print("\n🎯 Test du modèle :")

samples = [
    "Win a free iPhone now",
    "Hey bro are you coming tonight",
    "URGENT you won 1000 dollars",
    "Can you send me the file please"
]

# Tester chaque message
for msg in samples:
    vec = vectorizer.transform([msg])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    label = "SPAM 🔴" if pred == 1 else "HAM 🟢"
    print(f"{msg} --> {label} ({prob:.2f})")

print("\n✅ Figures générées dans /outputs/")