# Brainwave Matrix Intern - Fake News Classification

## Project Overview

This project was developed during an internship at **Brainwave Matrix** and focuses on building a machine learning model to classify news articles as real or fake. With the increasing spread of misinformation, it's crucial to have tools that can help in detecting and preventing the dissemination of fake news. This repository contains a comprehensive Jupyter Notebook that details the entire process, from data preprocessing to model evaluation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Challenges Faced](#challenges-faced)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

---

## Dataset Description

The **WELFake** dataset is used for this project. It is a comprehensive collection of news articles labeled as fake or real, merged from four popular datasets to provide a robust dataset for training and evaluating machine learning models.

- **Total Entries:** 72,134 articles
  - **Real News:** 35,028 articles (Label = 1)
  - **Fake News:** 37,106 articles (Label = 0)
- **Dataset Features:**
  - `title`: The headline of the news article
  - `text`: The main content of the news article
  - `label`: Binary label indicating fake (0) or real (1) news

*Dataset Reference: [IEEE Transactions on Computational Social Systems](https://doi.org/10.1109/TCSS.2021.3068519)*

---

## Project Structure

- `notebook.ipynb`: Jupyter Notebook containing all code for data preprocessing, model training, evaluation, and prediction.
- `README.md`: Project documentation (this file).
- `WELFake_Dataset.csv`: The dataset file (not included due to size constraints; instructions provided to obtain it).[Dataset](https://zenodo.org/records/4561253)

---

## Installation

To run this project locally, please follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/Brainwave_Matrix_Intern_Fake_News_Classification.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Brainwave_Matrix_Intern_Fake_News_Classification
   ```

3. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv venv
   # Activate the virtual environment:
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *Note: If `requirements.txt` is not available, install the dependencies manually:*

   ```bash
   pip install pandas numpy matplotlib nltk scikit-learn
   ```

5. **Download the NLTK data:**

   Open a Python shell or include the following in your code:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

6. **Obtain the WELFake dataset:**

   - Due to size constraints, the dataset is not included in this repository.
   - Download the dataset from the [IEEE DataPort](https://ieee-dataport.org/open-access/welfake-dataset) or any other reliable source.
   - Place the `WELFake_Dataset.csv` file in the project directory.

---

## Usage

1. **Open the Jupyter Notebook:**

   ```bash
   jupyter notebook notebook.ipynb
   ```

2. **Run the notebook cells:**

   - The notebook is structured sequentially, starting from data loading to model evaluation.
   - Ensure that you run each cell in order to avoid any errors.

3. **Make Predictions:**

   - At the end of the notebook, there is a section where you can input your own news text.
   - The model will predict whether the input text is real or fake news.

---

## Methodology

### Data Preprocessing

- **Handling Missing Values:**
  - Identified and filled missing values with empty strings to maintain data integrity.

- **Text Cleaning:**
  - Converted all text to lowercase.
  - Tokenized text using NLTK's `word_tokenize`.
  - Removed punctuation and stopwords to reduce noise in the data.
  - Applied Porter Stemming to reduce words to their root forms.

- **Exploratory Data Analysis:**
  - Analyzed the distribution of text lengths between fake and real news articles.
  - Visualized the data to identify patterns and anomalies.

### Feature Extraction

- **TF-IDF Vectorization:**
  - Transformed the cleaned text data into numerical feature vectors using `TfidfVectorizer`.
  - This method considers the importance of words in a document relative to the entire corpus.

### Model Training

- **Data Splitting:**
  - Split the dataset into training and testing sets using an 75-25 split.

- **Models Used:**
  - **Multinomial Naive Bayes Classifier:**
    - Suitable for classification with discrete features.
  - **Random Forest Classifier:**
    - Used with 300 estimators to improve accuracy.

### Model Evaluation

- **Performance Metrics:**
  - Generated classification reports including precision, recall, f1-score, and support.
  - Evaluated both models on the test dataset.

---

## Results

- **Multinomial Naive Bayes Classifier:**
  - **Accuracy:** Approximately 91%
  - Showed good performance but was outperformed by the Random Forest Classifier.

- **Random Forest Classifier:**
  - **Accuracy:** Approximately 94%
  - Provided better accuracy and overall performance.

*The Random Forest Classifier was selected as the preferred model due to its higher accuracy and robustness.*

---

## Challenges Faced

- **Memory Limitations:**
  - The large size of the dataset and high dimensionality of the feature space led to memory errors.
  - Resolved by using `TfidfVectorizer` and converting feature arrays to sparse matrices.

- **Processing Time:**
  - Data splitting and model training were time-consuming, taking several hours to complete.
  - Limited the use of resource-intensive processes and optimized code where possible.

- **Data Preprocessing Decisions:**
  - Initially planned to retain stopwords, but had to remove them to reduce dimensionality and prevent memory issues.
  - Opted for stemming instead of lemmatization to reduce complexity.

---

## Conclusion

The project successfully demonstrates the use of machine learning techniques for fake news detection. By preprocessing the data effectively and selecting appropriate models, we achieved high accuracy in classifying news articles. The Random Forest Classifier, in particular, showed superior performance and can be considered a reliable model for this task.

---

## Future Work

- **Model Optimization:**
  - Implement dimensionality reduction techniques such as PCA to further reduce feature space.
  - Experiment with deep learning models like LSTM and Transformers for better context understanding.

- **Feature Engineering:**
  - Use word embeddings (e.g., Word2Vec, GloVe) to capture semantic relationships.
  - Incorporate metadata features such as publication date, author credibility, and source reliability.

- **Deployment:**
  - Develop a web application or API for real-time fake news detection.
  - Integrate the model into browser extensions or social media platforms for wider accessibility.

---

## References

- **WELFake Dataset Publication:**
  - *H. Nozari, M. Salehi, and S. M. Hashemi, "A Comprehensive Analysis of WELFake: A Benchmark Dataset for Fake News Detection," IEEE Transactions on Computational Social Systems, vol. 8, no. 4, pp. 983-996, Aug. 2021.*
  - [DOI: 10.1109/TCSS.2021.3068519](https://doi.org/10.1109/TCSS.2021.3068519)

- **Libraries and Frameworks:**
  - [NLTK Documentation](https://www.nltk.org/)
  - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
  - [Pandas Documentation](https://pandas.pydata.org/docs/)
  - [NumPy Documentation](https://numpy.org/doc/)
  - [Matplotlib Documentation](https://matplotlib.org/)

- **Additional Resources:**
  - [Understanding TF-IDF](https://www.tfidf.com/)
  - [Machine Learning Mastery: Text Data Preprocessing](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/)

---

*For any questions, suggestions, or contributions, please feel free to open an issue or submit a pull request.*

---

**Note:** This project was developed as part of an internship at **Brainwave Matrix**. The code and methodologies used are for educational purposes and can be further enhanced for production-level applications.