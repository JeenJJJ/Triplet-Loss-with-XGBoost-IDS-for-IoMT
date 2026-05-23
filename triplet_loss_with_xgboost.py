import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_curve, auc, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from tensorflow.keras import layers, Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
DATA_PATH = 'wustl-ehms-2020_with_attacks_categories.csv'
# -----------------------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def create_triplets(X, y, num_triplets=5000):
    triplets = []
    classes = np.unique(y)
    for _ in range(num_triplets):
        anchor_class   = np.random.choice(classes)
        negative_class = np.random.choice(classes[classes != anchor_class])
        anchor_idx   = np.random.choice(np.where(y == anchor_class)[0])
        positive_idx = np.random.choice(np.where(y == anchor_class)[0])
        negative_idx = np.random.choice(np.where(y == negative_class)[0])
        triplets.append((X[anchor_idx], X[positive_idx], X[negative_idx]))
    A, P, N = zip(*triplets)
    return np.array(A), np.array(P), np.array(N)

def embedding_model(input_dim, embedding_dim=64):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(embedding_dim)(x)
    return Model(inputs, x, name="EmbeddingModel")

def triplet_loss(y_true, y_pred, margin=1.0):
    anchor, positive, negative = y_pred[:,0,:], y_pred[:,1,:], y_pred[:,2,:]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))

# -----------------------------------------------------------------------
# FIGURE 1 — Histogramme trafic normal
# -----------------------------------------------------------------------
def plot_hist_normal(df):
    col = 'SrcLoad' if 'SrcLoad' in df.columns else df.columns[5]
    data = df[df['Label'] == 0][col].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=50, color='steelblue', edgecolor='white')
    ax.set_title('Source Load Distribution — Normal Traffic (Class 0)')
    ax.set_xlabel('Source Load')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('histo normal.png', dpi=150)
    plt.close()
    print("Saved: histo normal.png")

# -----------------------------------------------------------------------
# FIGURE 2 — Histogramme trafic attaque
# -----------------------------------------------------------------------
def plot_hist_attack(df):
    col = 'SrcLoad' if 'SrcLoad' in df.columns else df.columns[5]
    data = df[df['Label'] == 1][col].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=50, color='tomato', edgecolor='white')
    ax.set_title('Source Load Distribution — Attack Traffic (Class 1)')
    ax.set_xlabel('Source Load')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig('histo anor.png', dpi=150)
    plt.close()
    print("Saved: histo anor.png")

# -----------------------------------------------------------------------
# FIGURE 3 & 4 — Distribution avant/après SMOTE
# -----------------------------------------------------------------------
def plot_smote(y_before, y_after):
    fig, ax = plt.subplots(figsize=(5, 4))
    _, counts = np.unique(y_before, return_counts=True)
    ax.bar(['Normal (0)', 'Attack (1)'], counts, color=['steelblue', 'tomato'])
    ax.set_title('Class Distribution Before SMOTE')
    ax.set_ylabel('Number of samples')
    for i, c in enumerate(counts):
        ax.text(i, c + 50, str(c), ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('smote_before.png', dpi=150)
    plt.close()
    print("Saved: smote_before.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    _, counts = np.unique(y_after, return_counts=True)
    ax.bar(['Normal (0)', 'Attack (1)'], counts, color=['steelblue', 'tomato'])
    ax.set_title('Class Distribution After SMOTE')
    ax.set_ylabel('Number of samples')
    for i, c in enumerate(counts):
        ax.text(i, c + 50, str(c), ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('smote_after.png', dpi=150)
    plt.close()
    print("Saved: smote_after.png")

# -----------------------------------------------------------------------
# FIGURE 5 — t-SNE (axes fixes -100 à 100 comme l'original)
# -----------------------------------------------------------------------
def plot_tsne(embeddings, labels):
    print("Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1],
                         c=labels, cmap='tab10', s=5, alpha=0.7)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_title('t-SNE Visualization of Triplet Loss Embeddings')
    plt.tight_layout()
    plt.savefig('tsne.png', dpi=150)
    plt.close()
    print("Saved: tsne.png")

# -----------------------------------------------------------------------
# FIGURE 6 — Matrice de confusion
# -----------------------------------------------------------------------
def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix — Triplet Loss + XGBoost')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

# -----------------------------------------------------------------------
# FIGURE 7 — ROC curve notre modèle seul
# -----------------------------------------------------------------------
def plot_roc_ours(y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC Curve — Triplet Loss + XGBoost')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_ours.png', dpi=150)
    plt.close()
    print("Saved: roc_ours.png")

# -----------------------------------------------------------------------
# FIGURE 8 — Comparaison ROC : SVM, Isolation Forest, LSTM, notre modèle
# -----------------------------------------------------------------------
def plot_roc_comparison(X_train_raw, X_test_raw, y_train, y_test, y_pred_prob_ours):
    print("Training baseline models for ROC comparison...")
    fig, ax = plt.subplots(figsize=(8, 6))

    # SVM
    print("  -> SVM...")
    svm = SVC(probability=True, kernel='rbf', C=1, random_state=42)
    svm.fit(X_train_raw, y_train)
    fpr, tpr, _ = roc_curve(y_test, svm.predict_proba(X_test_raw)[:, 1])
    ax.plot(fpr, tpr, label=f'SVM (AUC = {auc(fpr,tpr):.2f})', lw=1.5)

    # Isolation Forest
    print("  -> Isolation Forest...")
    iso = IsolationForest(n_estimators=100, contamination=0.125, random_state=42)
    iso.fit(X_train_raw)
    scores_iso = -iso.decision_function(X_test_raw)
    fpr, tpr, _ = roc_curve(y_test, scores_iso)
    ax.plot(fpr, tpr, label=f'Isolation Forest (AUC = {auc(fpr,tpr):.2f})', lw=1.5)

    # LSTM
    print("  -> LSTM (3 epochs)...")
    X_train_lstm = X_train_raw.reshape(X_train_raw.shape[0], 1, X_train_raw.shape[1])
    X_test_lstm  = X_test_raw.reshape(X_test_raw.shape[0],  1, X_test_raw.shape[1])
    lstm_model = Sequential([
        KerasLSTM(64, input_shape=(1, X_train_raw.shape[1])),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
    lstm_model.fit(X_train_lstm, y_train, epochs=3, batch_size=64, verbose=0)
    y_prob_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_prob_lstm)
    ax.plot(fpr, tpr, label=f'LSTM (AUC = {auc(fpr,tpr):.2f})', lw=1.5)

    # Notre modèle
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_ours)
    ax.plot(fpr, tpr, color='red', lw=2,
            label=f'Triplet Loss + XGBoost (AUC = {auc(fpr,tpr):.2f})')

    ax.plot([0,1],[0,1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig('compar.png', dpi=150)
    plt.close()
    print("Saved: compar.png")

# -----------------------------------------------------------------------
# FIGURE 9 — Architecture overview
# -----------------------------------------------------------------------
def plot_architecture():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    boxes = [
        'Raw IoMT\nFeatures\n(44 dimensions)',
        'Preprocessing\n+ SMOTE',
        'Embedding Net\n(Triplet Loss)\n→ 64 dimensions',
        'XGBoost\nClassifier',
        'Normal /\nAnomaly'
    ]
    colors = ['#AED6F1', '#A9DFBF', '#F9E79F', '#F1948A', '#D7BDE2']
    for i, (box, col) in enumerate(zip(boxes, colors)):
        ax.add_patch(plt.Rectangle((i*0.2, 0.2), 0.18, 0.6,
                     color=col, ec='gray', lw=1.2))
        ax.text(i*0.2 + 0.09, 0.5, box, ha='center', va='center', fontsize=9)
        if i < len(boxes) - 1:
            ax.annotate('', xy=((i+1)*0.2, 0.5), xytext=(i*0.2+0.18, 0.5),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Proposed Framework Architecture', fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig('image.png', dpi=150)
    plt.close()
    print("Saved: image.png")

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    labels = df['Label'].values
    X = df.drop(columns=['Label', 'Attack Category']).select_dtypes(include=['number']).values

    plot_hist_normal(df)
    plot_hist_attack(df)
    plot_architecture()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, labels)
    plot_smote(labels, y_res)

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42)

    print("Training Triplet Loss embedding model...")
    A, P, N = create_triplets(X_res, y_res, num_triplets=5000)
    embedding_net = embedding_model(input_dim=X_res.shape[1])

    inp_a = layers.Input(shape=(X_res.shape[1],))
    inp_p = layers.Input(shape=(X_res.shape[1],))
    inp_n = layers.Input(shape=(X_res.shape[1],))
    emb_a = embedding_net(inp_a)
    emb_p = embedding_net(inp_p)
    emb_n = embedding_net(inp_n)
    merged = layers.concatenate([emb_a[:,tf.newaxis,:],
                                  emb_p[:,tf.newaxis,:],
                                  emb_n[:,tf.newaxis,:]], axis=1)
    triplet_model = Model(inputs=[inp_a, inp_p, inp_n], outputs=merged)
    triplet_model.compile(optimizer='adam', loss=triplet_loss)
    triplet_model.fit([A, P, N], np.zeros(len(A)), epochs=5, verbose=1)

    print("Generating embeddings...")
    embeddings = embedding_net.predict(X_res)

    idx = np.random.choice(len(embeddings), min(3000, len(embeddings)), replace=False)
    plot_tsne(embeddings[idx], y_res[idx])

    X_train_emb, X_test_emb, y_train, y_test = train_test_split(
        embeddings, y_res, test_size=0.2, random_state=42)

    print("Training XGBoost on embeddings...")
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
        eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_emb, y_train)
    y_pred      = xgb_model.predict(X_test_emb)
    y_pred_prob = xgb_model.predict_proba(X_test_emb)[:, 1]

    print(f"\nAccuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC       : {auc(*roc_curve(y_test, y_pred_prob)[:2]):.4f}")

    plot_confusion(y_test, y_pred)
    plot_roc_ours(y_test, y_pred_prob)
    plot_roc_comparison(X_train_raw, X_test_raw, y_train_raw, y_test_raw, y_pred_prob)

    print("\nFiles generated:")
    print("  image.png, histo normal.png, histo anor.png")
    print("  smote_before.png, smote_after.png")
    print("  tsne.png, confusion_matrix.png, roc_ours.png, compar.png")

if __name__ == "__main__":
    main()
