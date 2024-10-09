# Brainwave Matrix Intern Fake News Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.6.2-green)](https://www.nltk.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-orange)](https://scikit-learn.org/stable/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

## Table of Contents

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Overview

In an era where misinformation spreads rapidly, detecting fake news is more critical than ever. This project, **Brainwave Matrix Intern Fake News Classification**, focuses on building machine learning models to accurately classify news articles as real or fake using the WELFake dataset. By leveraging natural language processing (NLP) techniques and powerful classifiers like Multinomial Naive Bayes and Random Forest, the project aims to contribute to the fight against misinformation.

---

## Dataset Description

The **WELFake** dataset is a comprehensive collection of news articles designed to facilitate the detection of fake news through machine learning.

- **Total Entries:** 72,134 news articles
  - **Fake News:** 37,106 articles (Label = 0)
  - **Real News:** 35,028 articles (Label = 1)
- **Columns:**
  - `Serial number`: Unique identifier for each article
  - `Title`: Headline of the news article
  - `Text`: Main content of the news article
  - `Label`: Indicates whether the news is real (1) or fake (0)
- **Source:** Merged from four popular datasets—Kaggle, McIntire, Reuters, and BuzzFeed Political—to prevent overfitting and provide extensive text data.
- **Publication:**
  - IEEE Transactions on Computational Social Systems: pp. 1-13
  - DOI: [10.1109/TCSS.2021.3068519](https://ieeexplore.ieee.org/document/9395133)

---

## Project Structure

```
Brainwave_Matrix_Intern_Fake_News_Classification/
├── data/
│   └── WELFake_Dataset.csv          # Dataset file (not included due to size)
├── models/
│   ├── CV_FRN.pkl                   # Saved TF-IDF Vectorizer
│   ├── MNB_FRN.pkl                  # Saved Multinomial Naive Bayes model
│   └── RF_FRN.pkl                   # Saved Random Forest model
├── notebooks/
│   └── Fake_News_Detection.ipynb    # Jupyter Notebook with code and analysis
├── src/
│   ├── data_preprocessing.py        # Script for data cleaning and preprocessing
│   ├── feature_extraction.py        # Script for vectorization
│   ├── model_training.py            # Script for training models
│   └── prediction.py                # Script for loading models and predicting
├── README.md                        # Project README file
├── requirements.txt                 # Python dependencies
└── LICENSE                          # Project license
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (optional but recommended)

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Brainwave_Matrix_Intern_Fake_News_Classification.git
   cd Brainwave_Matrix_Intern_Fake_News_Classification
   ```

2. **Create a virtual environment (optional)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the WELFake Dataset**

   - Due to file size constraints, the dataset is not included in the repository.
   - Download the dataset from [IEEE Dataport](https://zenodo.org/records/4561253).

5. **Place the dataset**

   - Move the `WELFake_Dataset.csv` file into the `data/` directory.

---

## Usage

### Running the Jupyter Notebook

1. **Start Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

2. **Open and run**

   - Navigate to `notebooks/Fake_News_Detection.ipynb`.
   - Run the cells sequentially to reproduce the analysis.

### Running Scripts Individually

- **Data Preprocessing**

  ```bash
  python src/data_preprocessing.py
  ```

- **Feature Extraction**

  ```bash
  python src/feature_extraction.py
  ```

- **Model Training**

  ```bash
  python src/model_training.py
  ```

- **Prediction**

  ```bash
  python src/prediction.py
  ```

### Making Predictions

You can use the `prediction.py` script to input a news article and get predictions from both models.

```bash
python src/prediction.py
```

*Example:*

```bash
Enter news text: "The government has announced a new policy to improve healthcare."
Prediction using MultinomialNB: Real News
Prediction using Random Forest: Real News
```

---

## Results

After training and evaluating both models, we achieved the following performance:

- **Multinomial Naive Bayes Classifier**
  - **Accuracy:** ~91%
  - Suitable for quick predictions with reasonable accuracy.
- **Random Forest Classifier**
  - **Accuracy:** ~94%
  - Provides higher accuracy at the cost of increased computation time.

---

## Challenges and Solutions

### Memory Limitations

- **Issue:** Memory errors when processing n-grams and bigrams with `CountVectorizer`, resulting in over 600,000 features.
- **Solution:** Switched to `TfidfVectorizer` and used a sparse matrix representation to handle the large feature set efficiently.

### Processing Time

- **Issue:** Data splitting and model training were time-consuming, taking several hours.
- **Solution:** Limited the use of resource-intensive processes and opted for efficient algorithms.

### Data Preprocessing Decisions

- **Issue:** Including stopwords increased dimensionality and memory usage.
- **Solution:** Removed stopwords to reduce the feature space and avoid memory constraints.

---

## Future Work

- **Optimize Memory Usage**
  - Implement dimensionality reduction techniques like PCA.
  - Utilize distributed computing for handling large datasets.

- **Model Enhancement**
  - Experiment with deep learning models such as RNNs or Transformers.
  - Implement cross-validation to improve model robustness.

- **Feature Engineering**
  - Incorporate word embeddings like Word2Vec or GloVe.
  - Use additional metadata features, such as publication date or author credibility.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a new branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit your changes**

   ```bash
   git commit -m "Add your message"
   ```

4. **Push to the branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## References

- **WELFake Dataset Publication**
  - M. Zubiaga et al., "WELFake: Word Embedding-based Learning Approach for Fake News Detection," *IEEE Transactions on Computational Social Systems*, vol. XX, no. XX, pp. 1-13, 2021.
  - DOI: [10.1109/TCSS.2021.3068519](https://doi.org/10.1109/TCSS.2021.3068519)

- **Libraries and Tools**
  - [NLTK Documentation](https://www.nltk.org/)
  - [Scikit-learn Documentation](https://scikit-learn.org/stable/)
  - [Pandas Documentation](https://pandas.pydata.org/docs/)
  - [NumPy Documentation](https://numpy.org/doc/)
  - [Matplotlib Documentation](https://matplotlib.org/)

---

*Developed by [Your Name](https://github.com/yourusername) as part of the Brainwave Matrix Internship.*