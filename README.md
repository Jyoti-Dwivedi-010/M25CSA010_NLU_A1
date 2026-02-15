# Question 4: Sports vs. Politics News Classifier 

This repository contains the implementation and report for a binary text classifier designed to distinguish between **Sports** and **Politics** news. The project explores feature representation, model comparison, and the iterative process of dataset validation to ensure model robustness

##  Project Overview
The core objective was to move beyond simplistic datasets to address the "Perfect Accuracy Trap" found in highly curated benchmarks The final model utilizes the **20 Newsgroups dataset**, which offers a higher degree of noise and real-world complexity

##  Key Features
* **Expanded Classification**: Includes diverse sub-groups such as Baseball and Hockey for Sports, and Middle-East, Guns, and Religion for Politics to simulate real-world discourse
* **Feature Engineering**: Implements **TF-IDF** with both unigrams and bigrams to capture semantic phrases
* **Model Comparison**: Benchmarks three major algorithms: Multinomial Naive Bayes, Linear SVM, and Random Forest
##  Quantitative Comparison
The models were evaluated using an 80/20 train-test split on 1,050 samples

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | **0.9552** | 0.96 | 0.94 | 0.95 |
| **Linear SVM** | 0.9457 | 0.95 | 0.93 | 0.94 |
| **Random Forest** | 0.9305 | 0.94 | 0.91 | 0.92 |





##  Interpretability & Analysis
* **Predictive Features**: The model identifies key tokens like "israel," "government," and "guns" for Politics, and "baseball," "nhl," and "hockey" for Sports
* **Temporal Bias**: Analysis reveals that statistical models are historical snapshots; the 20 Newsgroups data contains 1990s-specific terms like "Koresh" and "BATF"
* **Ablation Study**: While bigrams add semantic light, unigram models performed marginally better (0.9562 vs 0.9552), suggesting individual keywords are highly discriminative for this dataset

##  Limitations
The system follows a **Bag-of-Words** approach, meaning it lacks syntactic hierarchy It is excellent at domain detection but can be misled by mixed-context sentences where specific keywords outweigh the overall intent

## Repository Contents
* `M25CSA010_prob4.pdf`: Detailed Technical Report
* Source code for TF-IDF Vectorization and Model Training.
* Dataset preprocessing scripts.

---
**Author**: Dwivedi Jyoti Rajeshbhai (M25CSA010)   
**Date**: January 19, 2026 
