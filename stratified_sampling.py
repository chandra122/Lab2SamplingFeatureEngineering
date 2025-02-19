import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import tkinter as tk
from tkinter import ttk, messagebox

# Set the matplotlib backend to 'TkAgg' for tkinter compatibility
import matplotlib
matplotlib.use('TkAgg')

# Phase 1: Stratified Sampling
def load_and_prep_data():
    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    diabetes_data = pd.read_csv(url, header=None, names=column_names)
    return diabetes_data

def strat_split(diabetes_data):
    # Stratified Shuffle Split
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    diabetes_features = diabetes_data.drop('Outcome', axis=1)
    diabetes_target = diabetes_data['Outcome']
    for train_index, test_index in strat_split.split(diabetes_features, diabetes_target):
        strat_train_set = diabetes_data.loc[train_index]
        strat_test_set = diabetes_data.loc[test_index]
    return strat_train_set, strat_test_set

# Phase 2: Additional Sampling Techniques
def random_samp(diabetes_data): 
    return train_test_split(diabetes_data, test_size=0.2, random_state=42)

def systematic_step_samp(diabetes_data):
    step = 5
    indices = list(range(0, len(diabetes_data), step))
    return diabetes_data.iloc[indices]

def cluster_sampling(data):
    n_clusters = 3
    clusters = np.array_split(data, n_clusters)
    sampled_clusters = np.random.choice(range(n_clusters), size=n_clusters//2, replace=False)
    return pd.concat([clusters[i] for i in sampled_clusters])

# Function to plot class distribution
def plot_class_distribution(train_set, test_set, title, canvas):
    # Clear previous plot
    for widget in canvas.winfo_children():
        widget.destroy()

    # Plot class distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x='Outcome', data=train_set, ax=ax1)
    ax1.set_title(f'{title} - Training Set')

    sns.countplot(x='Outcome', data=test_set, ax=ax2)
    ax2.set_title(f'{title} - Testing Set')

    # Embed the plot in the tkinter canvas
    plot_canvas = FigureCanvasTkAgg(fig, master=canvas)
    plot_canvas.draw()
    plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Function to compare sampling results
def compare_sampling_results(phase1_train, phase1_test, phase2_train, phase2_test, phase2_name, canvas):
    # Clear previous plot
    for widget in canvas.winfo_children():
        widget.destroy()

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x='Outcome', data=phase1_train, color='blue', alpha=0.5, label='Phase 1', ax=ax1)
    sns.countplot(x='Outcome', data=phase2_train, color='red', alpha=0.5, label=f'Phase 2 ({phase2_name})', ax=ax1)
    ax1.set_title('Training Set Comparison')
    ax1.legend()

    sns.countplot(x='Outcome', data=phase1_test, color='blue', alpha=0.5, label='Phase 1', ax=ax2)
    sns.countplot(x='Outcome', data=phase2_test, color='red', alpha=0.5, label=f'Phase 2 ({phase2_name})', ax=ax2)
    ax2.set_title('Testing Set Comparison')
    ax2.legend()

    # Embed the plot in the tkinter canvas
    plot_canvas = FigureCanvasTkAgg(fig, master=canvas)
    plot_canvas.draw()
    plot_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# GUI Application
class SamplingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sampling Methods Comparison")
        self.root.geometry("1200x800")

        # Load data
        self.diabetes_data = load_and_prep_data()
        self.strat_train_set, self.strat_test_set = strat_split(self.diabetes_data)

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(side=tk.TOP, fill=tk.X)

        # Buttons for sampling methods
        ttk.Button(button_frame, text="Stratified Sampling", command=self.show_stratified).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="Random Sampling", command=self.show_random).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="Systematic Step Sampling", command=self.show_systematic).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="Cluster Sampling", command=self.show_cluster).pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas for plots
        self.plot_canvas = ttk.Frame(self.root)
        self.plot_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_stratified(self):
        plot_class_distribution(self.strat_train_set, self.strat_test_set, "Stratified Sampling", self.plot_canvas)

    def show_random(self):
        random_train, random_test = random_samp(self.diabetes_data)
        plot_class_distribution(random_train, random_test, "Random Sampling", self.plot_canvas)
        compare_sampling_results(self.strat_train_set, self.strat_test_set, random_train, random_test, "Random Sampling", self.plot_canvas)

    def show_systematic(self):
        systematic_train = systematic_step_samp(self.diabetes_data)
        systematic_test = systematic_step_samp(self.diabetes_data)
        plot_class_distribution(systematic_train, systematic_test, "Systematic Step Sampling", self.plot_canvas)
        compare_sampling_results(self.strat_train_set, self.strat_test_set, systematic_train, systematic_test, "Systematic Step Sampling", self.plot_canvas)

    def show_cluster(self):
        cluster_train = cluster_sampling(self.diabetes_data)
        cluster_test = cluster_sampling(self.diabetes_data)
        plot_class_distribution(cluster_train, cluster_test, "Cluster Sampling", self.plot_canvas)
        compare_sampling_results(self.strat_train_set, self.strat_test_set, cluster_train, cluster_test, "Cluster Sampling", self.plot_canvas)

# Run the application
if __name__ == '__main__':
    root = tk.Tk()
    app = SamplingApp(root)
    root.mainloop()