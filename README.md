# Spelling-Correction-with-Noisy-Input-Simulation
This project trains a Transformer-based spelling correction model using the **T5 architecture**. The model learns to correct artificially misspelled text, simulating human-like typos, and is evaluated on standard metrics like WER, CER, and ROUGE-L.


---

##  Overview

-  Introduces artificial typos to clean text using a variety of error strategies (insertion, deletion, substitution, transposition, vowel/consonant swap).
-  Trains a **T5-base** model to correct these misspellings.
-  Evaluates performance using **Character Error Rate (CER)**, **Word Error Rate (WER)**, and **ROUGE-L**.
-  Visualizes evaluation results with `seaborn`.

---

##  Tools & Libraries

| Tool/Library   | Purpose                                 |
|----------------|------------------------------------------|
| `transformers` | T5 model, tokenizer, training pipeline   |
| `datasets`     | Efficient dataset handling               |
| `evaluate`     | Built-in metric evaluation               |
| `jiwer`        | Calculates WER                           |
| `nltk`         | BLEU and text processing utilities       |
| `matplotlib`, `seaborn` | Visualization                   |
| `torch`        | Model training backend                   |

---

##  Dataset
https://github.com/google-research-datasets/wiki-split
Three datasets are used (provided in TSV format):

- `tune.tsv`: Training data
- `validation.tsv`: Validation data
- `test.tsv`: Test data

Each dataset contains clean (correct) text samples. Artificial typos are added during preprocessing.

---

##  Project Pipeline

### 1. **Typo Injection**
Random spelling errors are applied to each sentence using a variety of mutation strategies (e.g., insertions, deletions, swaps, vowel/consonant substitutions).

### 2. **Data Preprocessing**
Each sample is converted into a `(misspelled, correct)` pair and tokenized using the `T5Tokenizer`.

### 3. **Model Training**
- Model: `T5-base`
- Trainer: Hugging Face `Trainer`
- Early stopping, learning rate scheduling, and logging are included.

### 4. **Evaluation**
Evaluates the model on:
- **CER**: Character-level accuracy
- **WER**: Word-level accuracy
- **ROUGE-L**: Sequence-level match score

Sample predictions are printed and visualized in a bar chart.



