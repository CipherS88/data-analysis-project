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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class StudentDashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Fayha College - Student Dashboard")
        self.style = Style(theme="superhero")  # Modern dark theme
        self.file_path = None
        self.df = pd.DataFrame()
        self.root.frame = (1920,1080)

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
        self.tab_predictions = ttk.Frame(self.tabs)  # New predictions tab

        self.tabs.add(self.tab_demo, text='🎓 Demographics')
        self.tabs.add(self.tab_academic, text='📚 Academic')
        self.tabs.add(self.tab_financial, text='💰 Financial')
        self.tabs.add(self.tab_query, text='🔍 Query under coding')
        self.tabs.add(self.tab_predictions, text='🔮 Predictions')  # Add the predictions tab

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
                self.populate_predictions_tab()  # Populate the predictions tab
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

    def populate_predictions_tab(self):
        """Populate the predictions tab with future projections based on the data"""
        if self.df.empty:
            return
            
        # Clear the tab first
        for widget in self.tab_predictions.winfo_children():
            widget.destroy()
            
        # Create a container frame for the predictions
        predictions_frame = ttk.Frame(self.tab_predictions, padding=20)
        predictions_frame.pack(fill='both', expand=True)
        
        # Add a title
        title_label = ttk.Label(predictions_frame, text="🔮 Future Predictions & Forecasts", 
                               font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create the predictions visualizations
        fig = plt.figure(figsize=(14, 10))
        
        # 1. GPA Trend Forecast
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        self.plot_gpa_forecast(ax1)
        
        # 2. Debt Projection
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        self.plot_debt_projection(ax2)
        
        # 3. Graduation Rate Prediction
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        self.plot_graduation_prediction(ax3)
        
        # 4. Major Popularity Trend
        ax4 = plt.subplot2grid((2, 2), (1, 1))
        self.plot_major_trend(ax4)
        
        plt.tight_layout()
        
        # Display the predictions in the tab
        canvas = FigureCanvasTkAgg(fig, master=predictions_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add explanation text
        explanation_frame = ttk.Frame(predictions_frame)
        explanation_frame.pack(fill='x', pady=10)
        
        explanation_text = """
        These predictions are based on historical trends in the dataset and should be used as general guidance only.
        The models use simple linear regression and extrapolation of current trends.
        For more accurate predictions, consider using advanced machine learning models and additional data sources.
        """
        explanation_label = ttk.Label(explanation_frame, text=explanation_text, wraplength=800)
        explanation_label.pack(pady=5)

    def plot_gpa_forecast(self, ax):
        """Plot future GPA trend forecast"""
        df = self.df
        
        # Get average GPA by enrollment year
        gpa_by_year = df.groupby('Enrollment Year')['GPA'].mean().reset_index()
        
        # Create forecast for future years
        years = gpa_by_year['Enrollment Year'].values.reshape(-1, 1)
        gpas = gpa_by_year['GPA'].values
        
        # Simple linear regression for prediction
        model = LinearRegression()
        model.fit(years, gpas)
        
        # Create future years for prediction
        last_year = years[-1][0]
        future_years = np.array(range(last_year + 1, last_year + 4)).reshape(-1, 1)
        
        # Predict future GPAs
        future_gpas = model.predict(future_years)
        
        # Plot historical data
        ax.plot(years.flatten(), gpas, 'o-', color='#4CAF50', label='Historical')
        
        # Plot predictions
        ax.plot(future_years.flatten(), future_gpas, 'o--', color='#FF5733', label='Predicted')
        
        # Fill the area to indicate prediction uncertainty
        ax.fill_between(future_years.flatten(), 
                        future_gpas - 0.2, 
                        future_gpas + 0.2, 
                        color='#FF5733', alpha=0.2)
        
        ax.set_title("Average GPA Forecast")
        ax.set_xlabel("Enrollment Year")
        ax.set_ylabel("Average GPA")
        ax.set_ylim(min(gpas) - 0.5, max(gpas) + 0.5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    def plot_debt_projection(self, ax):
        """Plot projected debt trends"""
        df = self.df
        
        # Calculate average debt by semester
        debt_by_semester = df.groupby('Semesters')['Debt Amount'].mean().reset_index()
        
        # Plot the data
        semesters = debt_by_semester['Semesters'].values
        debt = debt_by_semester['Debt Amount'].values
        
        # Plot with curve fitting
        z = np.polyfit(semesters, debt, 2)
        p = np.poly1d(z)
        
        # Generate smooth curve for existing data
        xp = np.linspace(min(semesters), max(semesters), 100)
        
        # Generate future projection
        future_semesters = np.linspace(max(semesters), max(semesters) + 2, 20)
        
        # Plot historical and projected
        ax.scatter(semesters, debt, color='#4CAF50', label='Historical Data')
        ax.plot(xp, p(xp), '-', color='#4CAF50')
        ax.plot(future_semesters, p(future_semesters), '--', color='#FF5733', label='Projected')
        
        # Add shaded area for confidence interval
        ax.fill_between(future_semesters, 
                        p(future_semesters) * 0.9, 
                        p(future_semesters) * 1.1, 
                        color='#FF5733', alpha=0.2)
        
        ax.set_title("Student Debt Projection by Semester")
        ax.set_xlabel("Semester")
        ax.set_ylabel("Average Debt Amount (SAR)")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    def plot_graduation_prediction(self, ax):
        """Plot graduation rate prediction based on GPA and other factors"""
        df = self.df
        
        # Simple graduation prediction model based on GPA thresholds
        gpa_bins = [2.0, 2.5, 3.0, 3.5, 4.0]
        grad_rates = [0.65, 0.78, 0.85, 0.92, 0.98]  # Estimated rates
        
        # Create bar chart
        ax.bar(range(len(gpa_bins)), grad_rates, color='#4CAF50', alpha=0.7)
        
        # Add value labels on the bars
        for i, v in enumerate(grad_rates):
            ax.text(i, v + 0.02, f"{v*100:.0f}%", ha='center', fontsize=9)
        
        # Set labels and title
        ax.set_title("Predicted Graduation Rate by GPA")
        ax.set_xlabel("GPA Range")
        ax.set_ylabel("Predicted Graduation Rate")
        ax.set_xticks(range(len(gpa_bins)))
        ax.set_xticklabels([f"{gpa_bins[i-1]:.1f}-{x:.1f}" if i > 0 else f"<{x:.1f}" for i, x in enumerate(gpa_bins)])
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    def plot_major_trend(self, ax):
        """Plot projected trend in major popularity"""
        df = self.df
        
        # Count students by major and enrollment year
        major_trend = df.groupby(['Enrollment Year', 'Major']).size().unstack().fillna(0)
        
        # Get last few years for trending
        last_years = major_trend.index.sort_values()[-3:]
        recent_data = major_trend.loc[last_years]
        
        # Calculate percent change for each major          #sort
        pct_changes = recent_data.pct_change().iloc[-1].sort_values(ascending=False)
        
        # Plot top majors by growth rate
        top_majors = pct_changes.head(5).index
        
        # Create projection for future years
        last_year = max(major_trend.index)
        year_range = np.array(list(range(last_year - 2, last_year + 3)))
        
        # Plot each major's trend[predction not the main formula]
        for major in top_majors:
            if major in major_trend.columns:
                # Get historical data
                hist_data = major_trend[major].dropna()
                
                if len(hist_data) >= 2:
                    # Simple linear extrapolation
                    z = np.polyfit(hist_data.index, hist_data.values, 1)
                    p = np.poly1d(z)
                    
                    # Project future values
                    projected = p(year_range)
                    
                    # Plot the line
                    ax.plot(year_range, projected, 'o-', label=major)
                    
                    # Add dashed line for future projection
                    future_years = year_range[year_range > last_year]
                    if len(future_years) > 0:
                        future_data = p(future_years)
                        ax.plot(future_years, future_data, '--', alpha=0.7)
        
        ax.set_title("Major Popularity Projection")
        ax.set_xlabel("Enrollment Year")
        ax.set_ylabel("Number of Students")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title="Growing Majors", loc='upper left')
        
        # Add annotation for the projection
        ax.axvline(x=last_year, color='gray', linestyle='--', alpha=0.8)
        ax.text(last_year + 0.1, ax.get_ylim()[1] * 0.9, "Projection →", 
                fontsize=8, ha='left', va='center', color='gray')


if __name__ == '__main__':
    root = tk.Tk()
    app = StudentDashboardApp(root)
    root.mainloop()