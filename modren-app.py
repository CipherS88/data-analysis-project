import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import os
from ttkbootstrap import Style

class StudentDashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Fayha College - Student Dashboard")
        self.style = Style(theme="superhero")  # Modern dark theme
        self.file_path = None
        self.df = pd.DataFrame()

        self.create_widgets()

    def create_widgets(self):
        frame_top = ttk.Frame(self.root)
        frame_top.pack(pady=10, fill='x')

        self.file_label = ttk.Label(frame_top, text="📁 Upload .csv or .xlsx file")
        self.file_label.pack(side='left', padx=10)

        self.upload_btn = ttk.Button(frame_top, text="Browse", command=self.load_file, bootstyle="primary")
        self.upload_btn.pack(side='left')

        self.tabs = ttk.Notebook(self.root, bootstyle="dark")
        self.tabs.pack(fill='both', expand=True)

        self.tab_demo = ttk.Frame(self.tabs)
        self.tab_academic = ttk.Frame(self.tabs)
        self.tab_financial = ttk.Frame(self.tabs)
        self.tab_query = ttk.Frame(self.tabs)

        self.tabs.add(self.tab_demo, text='🎓 Demographics')
        self.tabs.add(self.tab_academic, text='📚 Academic')
        self.tabs.add(self.tab_financial, text='💰 Financial')
        self.tabs.add(self.tab_query, text='🔍 Query')

    def load_file(self):
        filetypes = [("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
        self.file_path = filedialog.askopenfilename(title="Open file", filetypes=filetypes)

        if self.file_path:
            try:
                if self.file_path.endswith('.csv'):
                    self.df = pd.read_csv(self.file_path)
                elif self.file_path.endswith('.xlsx'):
                    self.df = pd.read_excel(self.file_path)
                else:
                    messagebox.showerror("Invalid File", "Unsupported file format")
                    return

                self.preprocess_data()
                self.populate_tabs()
                messagebox.showinfo("Success", "File loaded and processed successfully ✅")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file: {e}")

    def preprocess_data(self):
        df = self.df

        for col in df.select_dtypes(include=[np.number]):
            df[col].fillna(df[col].mean(), inplace=True)

        for col in df.select_dtypes(include=[object]):
            df[col].fillna(df[col].mode()[0], inplace=True)

        df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
        df['Residence'] = df['Residence'].map({'Yes': 1, 'No': 0})
        df['Scholarship'] = df['Scholarship'].apply(lambda x: 1 if x != 'None' else 0)

        df['Tuition Owed'] = df['Tuition Owed'].replace('[^0-9.]', '', regex=True).astype(float)
        df['Debt Amount'] = df['Debt Amount'].replace('[^0-9.]', '', regex=True).astype(float)

        self.df = df

    def plot_to_tab(self, tab, fig):
        for widget in tab.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def populate_tabs(self):
        df = self.df

        # Demographics Tab
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        sns.countplot(data=df, x='Sex', ax=axs[0])
        axs[0].set_title("Gender Distribution")
        axs[0].set_xticklabels(['Female', 'Male'])

        sns.histplot(data=df, x='Age', bins=range(18, 26), ax=axs[1])
        axs[1].set_title("Age Distribution")

        sns.countplot(data=df, x='Major', ax=axs[2])
        axs[2].set_title("Major Popularity")
        axs[2].tick_params(axis='x', rotation=30)

        fig.suptitle("🎓 Demographics Overview", fontsize=16)
        plt.tight_layout()
        self.plot_to_tab(self.tab_demo, fig)

        # Academic Performance Tab
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        sns.boxplot(data=df, x='Major', y='GPA', ax=axs[0])
        axs[0].set_title("GPA by Major")
        axs[0].tick_params(axis='x', rotation=30)

        scatter = sns.scatterplot(data=df, x='Semesters', y='GPA', hue='Debt Amount', ax=axs[1])
        axs[1].set_title("GPA vs Semesters")

        avg_gpa = df.groupby('Enrollment Year')['GPA'].mean().reset_index()
        sns.lineplot(data=avg_gpa, x='Enrollment Year', y='GPA', marker='o', ax=axs[2])
        axs[2].set_title("Avg GPA per Year")

        fig.suptitle("📚 Academic Performance", fontsize=16)
        plt.tight_layout()
        self.plot_to_tab(self.tab_academic, fig)

        # Financial Analysis Tab (clean and clearer)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        df_fin = df.groupby('Scholarship')[['Tuition Owed', 'Debt Amount']].sum().reset_index()
        df_fin['Scholarship'] = df_fin['Scholarship'].map({1: 'With Scholarship', 0: 'Without Scholarship'})
        df_fin.set_index('Scholarship')[['Tuition Owed', 'Debt Amount']].plot(
            kind='bar', stacked=True, ax=axs[0], color=['#4CAF50', '#FF5733'])
        axs[0].set_title("Tuition vs Debt by Scholarship")
        axs[0].set_ylabel("Amount (SAR)")
        axs[0].legend(title="Category")

        corr = df[['Age', 'GPA', 'Debt Amount', 'Tuition Owed']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axs[1])
        axs[1].set_title("Correlation Matrix")

        fig.suptitle("💰 Financial Insights", fontsize=16)
        plt.tight_layout()
        self.plot_to_tab(self.tab_financial, fig)




if __name__ == '__main__':
    root = tk.Tk()
    app = StudentDashboardApp(root)
    root.mainloop()
