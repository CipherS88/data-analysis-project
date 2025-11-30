
import pandas as pd
import numpy as np
import random
from faker import Faker

# Initialize
fake = Faker('en')
np.random.seed(42)

# Configuration
n_students = 10000
enrollment_years = [2020, 2021, 2022, 2023, 2024, 2025]
majors = ['Computer Science', 'Computer Engineering', 'Chemical Engineering', 
          'Electrical Engineering', 'HR Management', 'Supply Chain']
major_weights = [0.35, 0.25, 0.15, 0.10, 0.10, 0.05]

# Generate data
data = []
for i in range(n_students):
    enroll_year = random.choice(enrollment_years)
    sex = 'Male' if random.random() <= 0.8 else 'Female'
    
    student = {
        'ID': f"{enroll_year}{str(i+1).zfill(5)}",
        'Name': fake.name_male() if sex == 'Male' else fake.name_female(),
        'Sex': sex,
        'Age': int(np.random.normal(21, 1.5)),
        'Enrollment Year': enroll_year,
        'Semesters': random.randint(1, 10),
        'Major': random.choices(majors, weights=major_weights)[0],
        'GPA': round(np.random.normal(2.8, 0.8), 1),
        'Residence': 'Yes' if i < 2 else 'No',
        'Study Time': 'Male Evening' if (sex == 'Male' and random.random() <= 0.3) 
                      else f"{sex} Morning",
        'Scholarship': f"{random.randint(10, 30)}%" if random.random() <= 0.15 else 'None'
    }
    
    # Calculate financials
    total_tuition = student['Semesters'] * 20000
    owes_tuition = random.random() <= 0.12
    student['Tuition Owed'] = f"{total_tuition if owes_tuition else 0} SAR"
    student['Debt Amount'] = f"{round(total_tuition * random.uniform(0.1, 1)) if owes_tuition else 0} SAR"
    
    data.append(student)

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_excel("fayha_college_students.xlsx", index=False)