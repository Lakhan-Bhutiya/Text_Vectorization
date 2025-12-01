#  Text Vectorization with TF, TF-IDF & Normalization 

This notebook explains how raw text is transformed into **numerical features** that machine-learning models can understand.  
It covers **Bag-of-Words, TF, Binary Encoding, TF-IDF, Normalization (L1/L2)** and shows how to build feature matrices using `scikit-learn`.

This repository is the second part of the NLP Learning Series.  
(Notebook 1 covers text preprocessing with NLTK.)

---

##  Topics Covered

### 🔹 1. Term Frequency (TF)
- Counts how many times each word appears.
- Example: “did did it” → did=2, it=1.
- Implemented using **CountVectorizer**.

### 🔹 2. Binary Vectorization
- Converts words into 0/1 values.
- 1 = word appears at least once.
- Useful for simple presence-based models.

### 🔹 3. TF with L1 Normalization
- Converts counts into proportions.
- Each row (sentence) sums to 1.
- Removes length bias between sentences.

### 🔹 4. TF-IDF (without normalization)
- Computes importance of words based on rarity.
- Rare words get higher weight.
- Common words get lower weight.
- Formula: **TF × IDF**

### 🔹 5. TF-IDF with L2 Normalization (Standard Method)
- Most widely used text vectorization method.
- Each vector is scaled so length = 1.
- Best for cosine similarity & ML models.
- Default in `TfidfVectorizer`.

### 🔹 6. Vocabulary Building
- Shows how words are mapped to vector positions.
- Example: `{'day': 1, 'good': 3, ...}`

### 🔹 7. DataFrame Output for Easy Analysis
- Converts TF/TF-IDF matrices into tables.
- Helps visualize how words are represented numerically.

### 🔹 8. Full Comparison of All Methods
- TF  
- Binary  
- TF + L1  
- TF-IDF (raw)  
- TF-IDF + L2  
- Explained with tables and code.

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `Text_Vectorization.ipynb` | Main notebook covering TF, TF-IDF, L1/L2 normalization, vocabulary, and vector comparison |

---

## Why This Notebook Matters

After cleaning text (Book 1), the next step is **turning text into numbers** so machine-learning models can use it.

This notebook teaches:

- How feature vectors work  
- Why normalization is important  
- The difference between TF and TF-IDF  
- How vocabulary defines vector structure  
- How to prepare text for ML algorithms  

This is an essential step before building:

- Sentiment analysis models  
- Text classifiers  
- Topic models  
- Recommender systems  
- Search engines  

---

## 📦 Technologies Used

- Python  
- scikit-learn  
- pandas  

---


## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Lakhan-Bhutiya/Text_Vectorization
   ```
2. Install required libraries:
   ```
   pip install nltk
   ```
   
3.Open the notebook:
   ```
   jupyter notebook
   ```
