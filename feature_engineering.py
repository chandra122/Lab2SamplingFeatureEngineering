import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk

# Phase 1: Handling Missing Data

def load_data():
    # Load dataset
    california = fetch_california_housing()
    data = pd.DataFrame(california.data, columns=california.feature_names)
    data['MedHouseVal'] = california.target  # Add target variable
    return data

def introduce_nulls(data):
    # Introduce missing values in 10% of the cells
    for col in data.columns:
        data.loc[data.sample(frac=0.1).index, col] = np.nan
    return data

def show_nulls(data):
    # Identify columns with missing data
    print("Missing values: " + str(data.isnull().sum()))

def replace_nulls(data):
    # Impute missing values
    no_null_data = pd.DataFrame()
    for column in data.columns:
        if data[column].dtype == 'int64' or data[column].dtype == 'float64':
            no_null_data[column] = data[column].fillna(data[column].median())
        else:
            no_null_data[column] = data[column]
    return no_null_data

# Load the dataset
california_data = load_data()

# Introduce missing values
california_data = introduce_nulls(california_data)

# Replace missing values
no_null_california = replace_nulls(california_data)

# Phase 2: Scaling Features
print("\nStarting Phase 2: Scaling Features")

# Separate features and target
california_features = no_null_california.drop(columns=['MedHouseVal'])
california_target = no_null_california['MedHouseVal']

def scale_features(features):
    # Select only the numeric columns for scaling
    numeric_columns = features.columns

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler to the numeric columns
    features[numeric_columns] = scaler.fit_transform(features[numeric_columns])
    return features

# Scale features
print("Scaling features...")
scaled_california = scale_features(california_features)

# Add scaled features back to the dataset
scaled_california['MedHouseVal'] = california_target
print("Feature scaling completed.\n")

# Tkinter GUI for Visualization

class CaliforniaHousingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("California Housing Dataset Visualization")
        self.root.geometry("1000x800")

        # Dataset
        self.data = scaled_california

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Buttons for visualizations
        ttk.Button(button_frame, text="Show Feature Distributions", command=self.show_distributions).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="Show Correlation Heatmap", command=self.show_heatmap).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="Show Scatter Plot", command=self.show_scatter_plot).pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas for plots
        self.plot_canvas = ttk.Frame(self.root)
        self.plot_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_distributions(self):
        # Clear previous plot
        for widget in self.plot_canvas.winfo_children():
            widget.destroy()

        # Plot feature distributions
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        fig.suptitle("Feature Distributions")
        axes = axes.ravel()

        for i, column in enumerate(self.data.columns[:-1]):  # Exclude target variable
            sns.histplot(self.data[column], ax=axes[i], kde=True)
            axes[i].set_title(column)

        plt.tight_layout()

        # Embed the plot in the tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_heatmap(self):
        # Clear previous plot
        for widget in self.plot_canvas.winfo_children():
            widget.destroy()

        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")

        # Embed the plot in the tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_scatter_plot(self):
        # Clear previous plot
        for widget in self.plot_canvas.winfo_children():
            widget.destroy()

        # Plot scatter plot of a selected feature vs. target
        fig, ax = plt.subplots(figsize=(8, 6))
        feature = 'MedInc'  # Example feature
        sns.scatterplot(x=self.data[feature], y=self.data['MedHouseVal'], ax=ax)
        ax.set_title(f"{feature} vs. MedHouseVal")
        ax.set_xlabel(feature)
        ax.set_ylabel("MedHouseVal")

        # Embed the plot in the tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Run the application
if __name__ == '__main__':
    root = tk.Tk()
    app = CaliforniaHousingApp(root)
    root.mainloop()