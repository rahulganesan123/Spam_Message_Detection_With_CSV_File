
# ============================================================
# NLP + MACHINE LEARNING
# PROJECT: SPAM MESSAGE DETECTION
# ============================================================
#
# NLP TECHNIQUE:
#   Sentence Transformers - all-MiniLM-L6-v2
#
# MACHINE LEARNING MODELS:
#   1. Logistic Regression
#   2. Decision Tree
#   3. Random Forest
#   4. Linear SVM
#   5. K-Nearest Neighbors (KNN)
#
# WORKFLOW:
#   1. Load CSV Dataset
#   2. Explore and Clean Data
#   3. Prepare Messages and Labels
#   4. Generate Sentence Embeddings
#   5. Split Training and Testing Data
#   6. Train Five ML Models
#   7. Compare Model Accuracy
#   8. Predict New Messages
#
# LABELS:
#   0 = NORMAL
#   1 = SPAM
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

from pathlib import Path

import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

DATASET_PATH = Path(__file__).resolve().parent / "spam_dataset_1200.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TEST_SIZE = 0.20
RANDOM_STATE = 42

LABEL_NAMES = {
    0: "NORMAL",
    1: "SPAM"
}


# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place spam_dataset_1200.csv beside this Python file."
    )

df = pd.read_csv(DATASET_PATH)

# Check required columns
required_columns = ["message", "label"]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing required CSV column: '{column}'"
        )

print("\nFirst 5 Records:")
print(df.head())

print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nOriginal Record Count:")
print(len(df))


# ------------------------------------------------------------
# 4. EXPLORE AND CLEAN DATA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: DATA EXPLORATION AND CLEANING")
print("=" * 60)

# Display category counts if the column exists
if "category" in df.columns:

    print("\nCategory Counts Before Cleaning:")
    print(df["category"].value_counts(dropna=False))

# Remove rows with missing messages or labels
df = df.dropna(subset=["message", "label"]).copy()

# Remove leading and trailing spaces from messages
df["message"] = df["message"].astype(str).str.strip()

# Remove empty messages
df = df[df["message"] != ""].copy()

# Convert labels to numeric values
df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

# Remove rows with invalid labels
df = df.dropna(subset=["label"]).copy()

# Ensure labels contain only 0 or 1
df = df[df["label"].isin([0, 1])].copy()

# Convert labels to integers
df["label"] = df["label"].astype(int)

# Remove duplicate messages
df = df.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)

if df.empty:
    raise ValueError(
        "No valid messages remain after data cleaning."
    )

if df["label"].nunique() != 2:
    raise ValueError(
        "The cleaned dataset must contain both labels: "
        "0 (NORMAL) and 1 (SPAM)."
    )

print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nRecords After Cleaning:")
print(len(df))

print("\nLabel Distribution:")
print(df["label"].value_counts().sort_index())

print("\nClass Names:")
print(LABEL_NAMES)


# ------------------------------------------------------------
# 5. PREPARE INPUT MESSAGES AND LABELS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: PREPARING INPUTS AND LABELS")
print("=" * 60)

# Input features: text messages
messages = df["message"].tolist()

# Target labels: 0 = NORMAL, 1 = SPAM
labels = df["label"].tolist()

print("\nTotal Input Messages:", len(messages))
print("Total Labels:", len(labels))

print("\nSample Message:")
print(messages[0])

print("\nCorresponding Label:")
print(labels[0])


# ------------------------------------------------------------
# 6. LOAD SENTENCE TRANSFORMER
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: LOADING SENTENCE TRANSFORMER")
print("=" * 60)

print("\nLoading Model:", EMBEDDING_MODEL_NAME)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Sentence Transformer Loaded Successfully.")


# ------------------------------------------------------------
# 7. CONVERT MESSAGES INTO EMBEDDINGS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5: GENERATING SENTENCE EMBEDDINGS")
print("=" * 60)

X = embedding_model.encode(
    messages,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("\nEmbedding Type:")
print(type(X))

print("\nEmbedding Shape:")
print(X.shape)

print("\nTotal Messages:")
print(X.shape[0])

print("\nEmbedding Dimensions:")
print(X.shape[1])


# ------------------------------------------------------------
# 8. SPLIT TRAINING AND TESTING DATA
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6: TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# ------------------------------------------------------------
# 9. CREATE MACHINE LEARNING MODELS
# ------------------------------------------------------------

classifiers = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Linear SVM": LinearSVC(
        random_state=RANDOM_STATE
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}


# ------------------------------------------------------------
# 10. TRAIN AND EVALUATE ALL MODELS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 7: TRAINING AND EVALUATION")
print("=" * 60)

results = {}
trained_models = {}

for name, model in classifiers.items():

    print(f"\nTraining Model: {name}")

    # Train the model
    model.fit(
        X_train,
        y_train
    )

    # Predict labels for the test data
    predictions = model.predict(
        X_test
    )

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Store the accuracy
    results[name] = accuracy

    # Store the trained model
    trained_models[name] = model

    print(f"Accuracy: {accuracy:.2%}")


# ------------------------------------------------------------
# 11. COMPARE MODEL ACCURACY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 8: MODEL ACCURACY COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Accuracy"]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

results_df["Accuracy (%)"] = (
    results_df["Accuracy"] * 100
).round(2)

print(
    results_df[["Model", "Accuracy (%)"]].to_string(
        index=False
    )
)

best_model_name = results_df.loc[0, "Model"]

print("\nBest Model by Test Accuracy:")
print(best_model_name)

print(
    f"Test Accuracy: {results[best_model_name]:.2%}"
)


# ------------------------------------------------------------
# 12. PREPARE NEW MESSAGES
# ------------------------------------------------------------

new_messages = [

    "Congratulations you won 50000 cash",

    "Please attend the project meeting tomorrow",

    "Claim your free lottery reward immediately",

    "Please send me the Python notes",

    "Click the project link and review the document",

    "Click here immediately to receive your cash prize",

    "You won the college coding competition",

    "Your account has won a special cash reward",

    "Please claim your travel reimbursement",

    "Can we discuss the React project tomorrow",

    "Free training session starts tomorrow",

    "You have been selected for a free gift"
]


# ------------------------------------------------------------
# 13. GENERATE EMBEDDINGS FOR NEW MESSAGES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 9: GENERATING NEW MESSAGE EMBEDDINGS")
print("=" * 60)

new_X = embedding_model.encode(
    new_messages,
    convert_to_numpy=True,
    show_progress_bar=False
)

print("\nNew Messages:", len(new_messages))
print("New Embedding Shape:", new_X.shape)


# ------------------------------------------------------------
# 14. PREDICT USING ALL FIVE MODELS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 10: NEW MESSAGE PREDICTIONS")
print("=" * 60)

for index, message in enumerate(new_messages):

    print("\n" + "-" * 60)
    print("Message:")
    print(message)

    # Select the embedding for the current message
    current_embedding = new_X[index:index + 1]

    # Predict using every trained model
    for name, model in trained_models.items():

        prediction = model.predict(
            current_embedding
        )[0]

        result = LABEL_NAMES[int(prediction)]

        print(f"{name:22}: {result}")


# ------------------------------------------------------------
# 15. PROJECT COMPLETED
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SPAM DETECTION PROJECT COMPLETED")
print("=" * 60)
