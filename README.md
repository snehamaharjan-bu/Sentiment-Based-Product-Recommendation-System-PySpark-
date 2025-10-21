# Sentiment-Based Product Recommendation System (PySpark)
**Author:** Sneha Maharjan  
**Course:** MET CS 777 – Big Data Analytics  

--------------------------------------------------------------------------------------------------------------

## Overview
This project builds a sentiment-based hybrid recommendation system using PySpark.  
It processes and analyzes product reviews, classifies them as positive or negative using Logistic Regression, and creates recommendations of similar, highly rated products.

-----
## Repository files

| File | Description |
|------|--------------|
| `Maharjan_Sneha_term_project.py` | Main Python script that has the model pipeline for the recommendation system |
| `sample30.csv` | Dataset of product reviews to be used as input |
| `DataAttributeDescription.csv` | Description of all dataset columns |
| `Sentiment-Based Product Recommendation System using PySpark.pdf` | Powerpoint presentation |
| `Term project_ Customer recommendation system - Sneha Maharjan.pdf` | Comprehensive report|
---

## How to run the project

After downloading the python file and the input csv file, run the following command on your terminal (replace <username> with your actual laptop username)->

```bash
spark-submit /Users/<username>/Downloads/Maharjan_Sneha_term_project.py /Users/<username>/Downloads/sample30.csv
