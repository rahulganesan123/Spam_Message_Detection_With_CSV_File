# Spam Message Detection Using Sentence Transformers and Machine Learning

## 1. Project Overview

This project implements a Spam Message Detection System using Natural Language Processing (NLP), Sentence Transformers, and traditional Machine Learning algorithms.

The system reads labeled text messages from a CSV dataset, converts each message into a numerical sentence embedding, trains five machine-learning classifiers, compares their accuracy, and predicts whether new messages are spam or normal.

Unlike a traditional TF-IDF approach, this project uses a pretrained Sentence Transformer to generate dense numerical representations of messages. These embeddings can capture semantic relationships between words and sentences.

### Classification Labels

| Label | Category | Meaning |
|---|---|---|
| 0 | NORMAL | A legitimate message |
| 1 | SPAM | An unwanted or potentially fraudulent or promotional message |

The actual meaning of each label must match the labeling convention used in the dataset.

## 2. Project Objectives

- Load and inspect a labeled CSV dataset using Pandas.
- Handle missing values, empty messages, invalid labels, and duplicate messages.
- Convert text messages into sentence embeddings using Sentence Transformers.
- Split the dataset into training and testing sets.
- Train five machine-learning classification algorithms.
- Compare model performance using accuracy.
- Identify the highest-accuracy model among the tested candidates.
- Apply all trained models to new messages and compare their predictions.

## 3. Technologies and Libraries

| Library or Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Loading, cleaning, and analyzing the dataset |
| Sentence Transformers | Generating semantic sentence embeddings |
| all-MiniLM-L6-v2 | Pretrained text embedding model |
| Scikit-learn | Dataset splitting, model training, and evaluation |
| Logistic Regression | Linear classification model |
| Decision Tree | Rule-based tree classifier |
| Random Forest | Ensemble of decision trees |
| Linear SVM | Linear maximum-margin classifier |
| KNN | Classification based on nearby training examples |

## 4. Dataset Structure

The program expects a file named `spam_dataset_1200.csv` in the same directory as the Python script.

The CSV must contain at least these two columns:

| Column | Description |
|---|---|
| `message` | Text content of the message |
| `label` | Numerical target: 0 for NORMAL or 1 for SPAM |

The code also displays category counts if an optional `category` column exists.

Example:

```csv
message,label,category
"Please attend the meeting tomorrow",0,normal
"Congratulations! Claim your free prize",1,spam
"Please send the project report",0,normal
"Win cash rewards immediately",1,spam
```

These are illustrative examples. The actual dataset should contain correctly labeled messages.

The filename suggests that the original dataset may contain 1,200 records. However, the actual record count depends on the CSV contents and the records removed during cleaning.

## 5. Complete Project Workflow

```text
CSV Dataset
    |
    v
Pandas Data Loading
    |
    v
Data Exploration
    |
    v
Data Cleaning
    |
    +-- Remove Missing Values
    +-- Remove Empty Messages
    +-- Validate Labels
    +-- Remove Duplicate Messages
    |
    v
Prepare Messages and Labels
    |
    v
Sentence Transformer
all-MiniLM-L6-v2
    |
    v
Sentence Embeddings
    |
    v
Train-Test Split
    |
    +---------------------------+
    |                           |
    v                           v
Training Data               Testing Data
    |                           |
    v                           |
Train Five ML Models            |
    |                           |
    +---------------------------+
                |
                v
         Predict Test Labels
                |
                v
          Accuracy Score
                |
                v
       Compare Five Models
                |
                v
          New Messages
                |
                v
       Generate Embeddings
                |
                v
       Predict with All Models
                |
                v
          SPAM / NORMAL
```

## 6. Detailed Code Explanation

### Step 1: Import Libraries

The script imports the tools required to load data, create embeddings, train models, and evaluate predictions.

```python
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
```

**Key functions and classes:**

- `Path`: Constructs a reliable path to the CSV file.
- `pd.read_csv()`: Loads the dataset into a DataFrame.
- `SentenceTransformer`: Loads the pretrained embedding model.
- `train_test_split()`: Divides the data into training and testing sets.
- The five classifiers: Learn to distinguish spam from normal messages.
- `accuracy_score()`: Measures how many test predictions are correct.

The cleaned version does not require NLTK or a separate TF-IDF vectorizer because it uses Sentence Transformers for text representation.

### Step 2: Configure the Project

The configuration section defines the dataset location, embedding model, test-set proportion, random seed, and label names.

```python
DATASET_PATH = Path(__file__).resolve().parent / "spam_dataset_1200.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TEST_SIZE = 0.20
RANDOM_STATE = 42

LABEL_NAMES = {
    0: "NORMAL",
    1: "SPAM"
}
```

`Path(__file__).resolve().parent` identifies the folder containing the Python script. This avoids relying on the terminal's current working directory when locating the CSV.

`TEST_SIZE = 0.20` reserves 20% of the valid data for testing.

`RANDOM_STATE = 42` makes the random split reproducible, provided the dataset and environment remain unchanged.

`LABEL_NAMES` converts numerical predictions into readable categories.

### Step 3: Load the CSV Dataset

```python
df = pd.read_csv(DATASET_PATH)
```

Pandas reads the CSV file and stores its contents in a DataFrame.

A DataFrame is a two-dimensional data structure with rows and columns, similar to a spreadsheet.

The code checks that the required `message` and `label` columns exist before continuing.

It then displays:

- The first five records.
- The original dataset shape.
- The original record count.

The dataset shape is represented as:

```text
(number of rows, number of columns)
```

For example, a dataset containing 1,200 records and three columns would have a shape of `(1200, 3)` before cleaning.

### Step 4: Explore and Clean the Data

Real-world datasets may contain missing values, empty strings, invalid labels, and repeated messages. These can cause errors or affect model training.

The code performs several cleaning operations.

**4.1 Remove missing messages and labels**

```python
df = df.dropna(subset=["message", "label"]).copy()
```

Rows without a message or target label are removed because they cannot provide a complete supervised-learning example.

**4.2 Remove empty messages**

```python
df["message"] = df["message"].astype(str).str.strip()

df = df[df["message"] != ""].copy()
```

Leading and trailing spaces are removed. Messages that become empty are discarded.

**4.3 Validate labels**

```python
df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(subset=["label"]).copy()

df = df[df["label"].isin([0, 1])].copy()

df["label"] = df["label"].astype(int)
```

The label column is converted to numeric values. Invalid values are discarded, and only the supported labels `0` and `1` are retained.

**4.4 Remove duplicate messages**

```python
df = df.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)
```

Repeated message text is removed so that each exact message appears only once.

This implementation retains the first occurrence of a duplicated message. If identical messages have conflicting labels, those conflicts should be investigated separately rather than automatically trusting the first label.

**Why is data cleaning important?**

The purpose is to ensure that the model receives valid input messages and meaningful target labels. Cleaning also makes the number of records used for training explicit.

### Step 5: Separate Input Features and Labels

```python
messages = df["message"].tolist()

labels = df["label"].tolist()
```

The input and target are separated into two Python lists.

- `messages` contains the raw text.
- `labels` contains the corresponding numerical classifications.

For supervised machine learning, each message must remain aligned with its correct label.

For example:

```text
Message: "Please send the project report"
Label:   0

Message: "Claim your free prize now"
Label:   1
```

The model learns from these examples to predict the labels of messages it has not seen during training.

### Step 6: Load the Sentence Transformer

```python
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
```

Sentence Transformers provides pretrained models that transform text into numerical vectors called **sentence embeddings**.

The model used here is `all-MiniLM-L6-v2`.

It produces a dense embedding with 384 dimensions for each input message.

For example, a message is converted from readable text into a vector of numerical values:

```text
"Claim your free cash reward"
                |
                v
    Sentence Transformer
                |
                v
[0.12, -0.08, 0.31, ..., -0.04]
                |
                v
        384 numerical values
```

The values above are illustrative, not actual model output.

These embeddings represent information learned by the pretrained model about the text. Semantically related messages can have similar vector representations, although similarity is not guaranteed to correspond perfectly to spam categories.

**Important:** The first run may require an internet connection to download the pretrained model. Subsequent runs can generally reuse its local cached files.

### Step 7: Generate Sentence Embeddings

```python
X = embedding_model.encode(
    messages,
    convert_to_numpy=True,
    show_progress_bar=True
)
```

The `encode()` method converts all messages into numerical vectors.

- `messages`: The list of input sentences.
- `convert_to_numpy=True`: Returns the embeddings as a NumPy array.
- `show_progress_bar=True`: Displays progress during encoding.

If there are 1,200 valid messages and every message produces a 384-dimensional embedding, the resulting matrix has the shape:

```text
(1200, 384)
```

Each row represents one message, and each column represents one dimension of its embedding.

The number of rows may be lower if records were removed during cleaning.

Unlike TF-IDF, which represents text using vocabulary-based weights, sentence embeddings are dense, fixed-dimensional representations generated by a pretrained neural network.

### Step 8: Split Training and Testing Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)
```

The dataset is divided into two subsets.

**Training set:** Used to teach the five classifiers.

**Testing set:** Used to measure how well the trained models predict labels for held-out examples.

With 1,200 valid messages, an 80/20 split would typically result in:

| Subset | Approximate records |
|---|---:|
| Training | 960 |
| Testing | 240 |

The actual counts depend on the number of records remaining after cleaning.

`stratify=labels` attempts to preserve the proportion of spam and normal examples in both subsets.

The sentence embeddings are generated before the split because the embedding model is pretrained and is not fitted to this dataset's labels. The five classification models are trained only on the training portion.

### Step 9: Create Five Machine-Learning Models

The project compares five classifiers.

#### 9.1 Logistic Regression

```python
LogisticRegression(max_iter=1000, random_state=42)
```

Logistic Regression learns a decision boundary from the numerical features and estimates class probabilities internally.

It is a useful baseline for binary classification and can work effectively with dense sentence embeddings.

`max_iter=1000` allows additional optimization iterations if the model needs them to converge.

#### 9.2 Decision Tree

```python
DecisionTreeClassifier(random_state=42)
```

A Decision Tree learns a series of feature-based splits that divide the training examples into groups.

It can model nonlinear decision boundaries, but a single tree can overfit its training data.

#### 9.3 Random Forest

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Random Forest combines predictions from multiple decision trees.

- `n_estimators=100`: Creates 100 trees.
- `random_state=42`: Makes its random processes reproducible.
- `n_jobs=-1`: Uses the available CPU cores for supported parallel operations.

Combining trees can improve generalization compared with an individual tree, although the actual performance depends on the dataset and parameters.

#### 9.4 Linear Support Vector Machine

```python
LinearSVC(random_state=42)
```

Linear SVM learns a linear decision boundary that separates the classes with a large margin.

It is commonly used for classification tasks involving high-dimensional numerical representations.

Unlike Logistic Regression, `LinearSVC` does not directly provide calibrated class probabilities through `predict_proba()`.

#### 9.5 K-Nearest Neighbors

```python
KNeighborsClassifier(n_neighbors=5)
```

KNN classifies an example based on the labels of nearby training examples.

With `n_neighbors=5`, the classifier considers five neighboring training vectors and assigns a class based on their votes.

The result can be affected by the distance metric, embedding geometry, and the balance of the classes.

All five models receive the same training embeddings and target labels, which makes their test-set accuracy easier to compare.

### Step 10: Train and Evaluate All Models

The classifiers are stored in a dictionary:

```python
classifiers = {
    "Logistic Regression": LogisticRegression(...),
    "Decision Tree": DecisionTreeClassifier(...),
    "Random Forest": RandomForestClassifier(...),
    "Linear SVM": LinearSVC(...),
    "KNN": KNeighborsClassifier(...)
}
```

The code loops through each model:

```python
for name, model in classifiers.items():
```

For each classifier, it performs three operations.

**10.1 Train the model**

```python
model.fit(X_train, y_train)
```

The classifier learns the relationship between sentence embeddings and the correct labels.

**10.2 Predict the test labels**

```python
predictions = model.predict(X_test)
```

The trained classifier predicts SPAM or NORMAL for each held-out test message.

**10.3 Calculate accuracy**

```python
accuracy = accuracy_score(
    y_test,
    predictions
)
```

Accuracy measures the fraction of test examples classified correctly.

The formula is:

\[
\text{Accuracy} =
\frac{\text{Correct Predictions}}
{\text{Total Test Examples}}
\]

For example, if a model correctly classifies 216 out of 240 test messages, its accuracy is 90%.

This is only an illustrative example, not a measured result from this project.

The code saves the accuracy in `results` and the fitted classifier in `trained_models`. Saving the trained objects in dictionaries allows them to be reused later without retraining.

### Step 11: Compare Model Accuracy

```python
results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Accuracy"]
)
```

Pandas converts the recorded results into a table.

The table is sorted by accuracy in descending order so that the model with the highest test accuracy appears first.

```python
results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)
```

The code then displays each model's accuracy as a percentage.

The highest-scoring model is identified using:

```python
best_model_name = results_df.loc[0, "Model"]
```

This selects the top model based on the current test-set accuracy.

**Evaluation note:** Comparing several models and selecting the winner on the same test set can lead to optimistic estimates of performance. For a more rigorous experiment, use a validation set or cross-validation to select the model, and reserve an untouched test set for the final evaluation.

Accuracy alone can also be misleading when spam and normal classes are imbalanced. Precision, recall, F1-score, and a confusion matrix would provide a more complete evaluation.

### Step 12: Prepare New Messages

The project defines a list of messages that are separate from the labeled training dataset.

Examples include:

```python
new_messages = [
    "Congratulations you won 50000 cash",
    "Please attend the project meeting tomorrow",
    "Claim your free lottery reward immediately"
]
```

These messages are used to demonstrate how the trained classifiers make predictions on new text.

The sample list contains both spam-like and normal-looking examples, but their actual classification is determined by the trained models.

### Step 13: Generate Embeddings for New Messages

```python
new_X = embedding_model.encode(
    new_messages,
    convert_to_numpy=True,
    show_progress_bar=False
)
```

The same pretrained Sentence Transformer converts each new message into a 384-dimensional embedding.

If there are 12 new messages, the embedding matrix has the shape:

```text
(12, 384)
```

Using the same embedding model ensures that the new vectors are produced in the same feature space as the training embeddings.

No new classifier training is required.

### Step 14: Predict Using All Five Models

The code loops through the new messages:

```python
for index, message in enumerate(new_messages):
```

For each message, it selects its corresponding embedding:

```python
current_embedding = new_X[index:index + 1]
```

The slice retains the two-dimensional shape expected by scikit-learn's `predict()` method.

Each trained classifier then predicts a label for that message.

```python
prediction = model.predict(
    current_embedding
)[0]
```

The numerical result is converted to a readable label:

```python
result = LABEL_NAMES[int(prediction)]
```

The output shows the same message alongside predictions from all five models.

For example, the output format might look like this:

```text
Message:
Claim your free lottery reward immediately

Logistic Regression  : SPAM
Decision Tree        : SPAM
Random Forest        : SPAM
Linear SVM           : SPAM
KNN                  : SPAM
```

This is an illustrative output format only. Actual predictions depend on the dataset, trained models, and their learned decision boundaries.

Comparing predictions helps identify whether different classifiers agree or disagree on a particular message.

## 7. How Sentence Embeddings Differ from TF-IDF

An important learning outcome of this project is the difference between traditional text vectorization and pretrained sentence embeddings.

| Feature | TF-IDF | Sentence Transformers |
|---|---|---|
| Representation | Vocabulary-based sparse vectors | Dense numerical vectors |
| Feature meaning | Term frequency and document rarity | Contextual and semantic information learned by a pretrained model |
| Feature dimensions | Depends on the learned vocabulary | Fixed for a selected model |
| Semantic similarity | Limited direct semantic understanding | Can represent semantic similarities |
| Pretrained model | Not required | Required in this project |
| Computational cost | Usually lower | Higher due to neural-network encoding |

For example, the messages "Claim your free reward" and "Get your complimentary prize" use different words but have similar meanings.

TF-IDF primarily represents their word usage, while a sentence embedding model can also capture learned semantic relationships between the messages.

This does not guarantee better spam classification. Performance must be measured on a representative labeled dataset.

## 8. How to Run the Project

### Prerequisites

Install Python and ensure that pip is available.

### Step 1: Arrange the Project Files

```text
spam-detection/
|
|-- spam_detection_embeddings.py
|-- spam_dataset_1200.csv
|-- README.md
|-- requirements.txt
```

### Step 2: Install Dependencies

Create a `requirements.txt` file containing:

```text
pandas
sentence-transformers
scikit-learn
```

Install the libraries:

```bash
python -m pip install -r requirements.txt
```

### Step 3: Run the Python Script

```bash
python spam_detection_embeddings.py
```

The first run may download the `all-MiniLM-L6-v2` model. An internet connection is needed if the model is not already cached locally.

### Step 4: Review the Results

The script displays:

1. Dataset information and cleaning results.
2. The number of valid messages and their labels.
3. Sentence embedding dimensions.
4. Training and testing sample counts.
5. Accuracy for each of the five classifiers.
6. A sorted model comparison table.
7. Predictions from all five classifiers for every new message.

## 9. Project Limitations

- **Dataset quality:** The model can only learn from the quality and diversity of the labeled messages.
- **Accuracy is incomplete:** Accuracy alone does not show how many spam messages were missed or how many legitimate messages were incorrectly flagged.
- **Model selection:** Choosing a winner using test accuracy can bias the final performance estimate.
- **Duplicate handling:** Removing exact duplicate text may discard examples with conflicting labels, which should be investigated separately.
- **No persistent model saving:** The script keeps trained models in memory. They are lost when the Python process ends.
- **No deployment interface:** The project currently runs in the terminal and does not expose a web API or user interface.

## 10. Future Improvements

Potential next steps include:

- Add precision, recall, F1-score, and a confusion matrix.
- Compare models using cross-validation.
- Add a validation set for model selection.
- Tune hyperparameters for each classifier.
- Save the selected model and embedding model configuration for reuse.
- Create a prediction function that accepts one message at a time.
- Develop a Streamlit interface for interactive spam detection.
- Build a FastAPI endpoint for serving predictions.
- Evaluate performance on a larger and more representative dataset.

## 11. Key Learning Outcomes

Through this project, I practiced:

- Loading and cleaning text datasets with Pandas.
- Preparing labeled data for supervised machine learning.
- Generating sentence embeddings using a pretrained transformer model.
- Understanding the difference between TF-IDF and dense embeddings.
- Training five different machine-learning classifiers.
- Evaluating and comparing classification accuracy.
- Reusing trained models to predict unseen messages.
- Building an end-to-end NLP and machine-learning workflow.

## 12. Conclusion

This project demonstrates how pretrained sentence embeddings can be combined with traditional machine-learning algorithms to perform binary text classification.

Sentence Transformers converts raw text into numerical representations, and five classifiers learn to distinguish between spam and normal messages. Their performance is compared using a held-out test set, and each model is applied to new messages.

The project provides practical experience with semantic text representations, supervised classification, model evaluation, and experimentation with different algorithms.

**Technologies:** Python · Pandas · Sentence Transformers · Scikit-learn · Logistic Regression · Decision Tree · Random Forest · Linear SVM · KNN
