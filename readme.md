<div align="center">

# 📊 Fayha College · Student Dashboard

### *Raw student data in, academic and financial insight out*

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-4B8BBE?style=for-the-badge&logo=python&logoColor=white)

> *A desktop dashboard that cleans a student dataset, draws the charts nobody has time to draw, and forecasts what comes next.*

</div>

---

## 🧐 Overview

This application helps administrators visualise student performance and financial data. Feed it a `.csv` or `.xlsx` file and it cleans the data, fills missing values, and generates interactive reports.

It ships with a modern dark-theme interface and uses machine learning to project future trends.

---

## 🚀 المميزات

### 🎓 1. Demographics

- **Gender & age distribution** — visual breakdown of the student body.
- **Major popularity** — which programmes attract the most students.

### 📚 2. Academic performance

- **GPA vs. major** — boxplots that show which majors are hard.
- **Semester trends** — how GPA moves as students progress.
- **Correlation** — how semesters impact academic success.

### 💰 3. Financial insights

- **Debt analysis** — stacked bars of tuition vs. debt.
- **Scholarship impact** — scholarship vs. non-scholarship standing.
- **Heatmaps** — correlation matrix across age, GPA and debt.

### 🔮 4. Predictive modelling

- **GPA forecast** — linear regression on future average GPA.
- **Debt projection** — polynomial curve fitting for future debt load.
- **Graduation rate** — probability from current GPA.
- **Major trends** — which majors grow in the next 3 years.

---

## 🛠️ التقنيات

| المكتبة | الدور |
| :--- | :--- |
| **Tkinter & TtkBootstrap** | الواجهة الرسومية |
| **Pandas & NumPy** | تنظيف البيانات ومعالجتها |
| **Matplotlib & Seaborn** | الرسوم البيانية والإحصائية |
| **scikit-learn** | نماذج الانحدار في تبويب التوقعات |

---

## 💻 التشغيل

```bash
pip install pandas numpy seaborn matplotlib scikit-learn ttkbootstrap openpyxl
python beta-app.py      # النسخة الكاملة
python modren-app.py    # نسخة الواجهة الحديثة
python generator.py     # توليد بيانات تجريبية
```

الملفات في `sample/` جاهزة للتجربة، و`pdf.html` يعرض التقرير بصيغة قابلة للطباعة.

---

## 📂 البنية

```text
beta-app.py          التطبيق الرئيسي
modren-app.py        نسخة الواجهة الحديثة
generator.py         مولّد بيانات تجريبية
sample/              ملفات CSV/XLSX للتجربة
student_dashboard_doc.html   التوثيق
pdf.html             نسخة التقرير للطباعة
```

---

<div align="center">

<sub>Built by <b>Abdullah Aljuaid</b> · <a href="https://github.com/CipherS88">@CipherS88</a></sub>

</div>
