# Sentimental-Analysis
## IMDB Sentiment Analysis

### Project Overview
This project implements sentiment analysis on the IMDB movie reviews dataset using natural language processing (NLP) techniques and machine learning. The goal is to classify movie reviews as positive or negative based on the text content.

### Objective
- Perform sentiment analysis on IMDB movie reviews.
- Preprocess text data by cleaning and vectorizing the reviews.
- Train and evaluate a classification model using logistic regression and TF-IDF vectorization.
- Visualize the results and analyze model performance.

### Dataset
The dataset used is the **IMDB Reviews** dataset, which contains movie reviews labeled as positive or negative.

### Features:
- **Review** - The actual text of the review.
- **Sentiment** - The label indicating if the review is positive (1) or negative (0).

### Technologies and Libraries Used:
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, NLTK

### Implementation

#### Key Steps:
1. **Data Loading and Exploration**
   - The IMDB dataset is loaded using `pd.read_csv()`. Basic data analysis is conducted to understand the distribution of positive and negative reviews.

2. **Text Preprocessing**
   - Removal of HTML tags, special characters, and stopwords.
   - Tokenization and lemmatization of the review text.

3. **Feature Extraction**
   - Text data is converted into numerical form using **TF-IDF vectorization**.

4. **Model Training and Evaluation**
   - Train-test split (80-20 split) is applied.
   - A logistic regression model is trained on the TF-IDF vectors.
   - Model performance is evaluated using accuracy, precision, recall, and F1-score.

5. **Visualization**
   - A confusion matrix is plotted to show classification performance.
   - Bar plots display the distribution of sentiments.

### Results
- **Accuracy:** Displays the percentage of correctly classified reviews.
- **Confusion Matrix:** Visualizes the performance of the classifier by showing true positives, true negatives, false positives, and false negatives.
- **Classification Report:** Includes precision, recall, and F1-score.

### Project Structure
```
|-- imdb_sentiment_analysis
    |-- data
        |-- IMDB Dataset.csv
    |-- notebooks
        |-- data_exploration.ipynb
        |-- sentiment_analysis.ipynb
    |-- results
        |-- confusion_matrix.png
    |-- README.md
```

### How to Run the Project
1. Clone the repository:
```bash
git clone https://github.com/username/imdb_sentiment_analysis.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook
```

4. Open `sentiment_analysis.ipynb` and execute all cells.

### Future Enhancements
- Use more complex models such as LSTM and BERT for improved accuracy.
- Perform hyperparameter tuning to enhance model performance.
- Deploy the model as a web application using Flask or FastAPI.

