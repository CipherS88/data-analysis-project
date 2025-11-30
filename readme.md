<div align="center">

# 📊 Fayha College - Student Dashboard
### *Data Analysis & Predictive Modeling*

<br>

> *A comprehensive GUI tool transforming raw student data into actionable academic and financial insights. Powered by Python & Machine Learning.* 🔮

<br>

![Python](https://img.shields.io/badge/Python-Data_Science-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Sklearn](https://img.shields.io/badge/scikit--learn-Regression-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Analytics-150458?style=for-the-badge&logo=pandas&logoColor=white)

</div>

---

## 🧐 Overview
This application is designed to help administrators visualize student performance and financial data. By uploading a simple `.csv` or `.xlsx` file, the dashboard automatically cleans the data, fills missing values, and generates interactive reports.

It features a modern **"Superhero" Dark Theme** interface and uses Machine Learning to forecast future trends. 🌃

---

## 🚀 Key Features

### 🎓 1. Demographics Analysis
* **Gender & Age Distribution:** Visual breakdown of the student body.
* **Major Popularity:** Tracks which courses are attracting the most students.

### 📚 2. Academic Performance
* **GPA vs. Major:** Boxplots to identify difficult majors.
* **Semester Trends:** Analyze how GPA fluctuates as students progress.
* **Correlation:** See how semesters impact academic success.

### 💰 3. Financial Insights
* **Debt Analysis:** Stacked bar charts showing Tuition vs. Debt.
* **Scholarship Impact:** Compares financial standing of scholarship vs. non-scholarship students.
* **Heatmaps:** Correlation matrix to find hidden relationships between Age, GPA, and Debt.

### 🔮 4. Predictive Modeling (Machine Learning)
* **GPA Forecast:** Uses **Linear Regression** to predict future average GPA trends.
* **Debt Projection:** Polynomial curve fitting to estimate future student debt loads.
* **Graduation Rate:** Predictive model estimating graduation probability based on current GPA.
* **Major Trends:** Projects which majors will grow in the next 3 years.

---

## 🛠️ The Tech Stack

| Library | Purpose |
| :--- | :--- |
| **Tkinter & TtkBootstrap** | The modern, responsive GUI framework. |
| **Pandas & NumPy** | Data manipulation, cleaning, and preprocessing. |
| **Matplotlib & Seaborn** | Generating the static and statistical graphs. |
| **Scikit-Learn** | Linear Regression models for the "Predictions" tab. |

---

## 💻 How to Run

### 1. Prerequisites
Ensure you have Python installed and the required data science libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn ttkbootstrap openpyxl
