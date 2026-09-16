import tkinter as tk
from tkinter import filedialog, messagebox

from pages.dataset import DatasetPage
from pages.training import TrainingPage
from pages.analytics import AnalyticsPage


class HomePage(tk.Frame):
    """Landing page for choosing or uploading a dataset."""

    def __init__(self, parent, controller):
        super().__init__(parent, bg="#F4F6F8")
        self.controller = controller

        tk.Label(
            self,
            text="Explainable Customer Churn\nPrediction System",
            font=("Arial", 28, "bold"),
            bg="#F4F6F8",
            fg="#1F2937",
        ).pack(pady=(70, 15))

        tk.Label(
            self,
            text="Machine Learning Based Telecom Customer Churn Analysis",
            font=("Arial", 13),
            bg="#F4F6F8",
            fg="#555555",
        ).pack(pady=(0, 40))

        button_frame = tk.Frame(self, bg="#F4F6F8")
        button_frame.pack()

        buttons = (
            ("Use IBM Dataset", controller.use_ibm_dataset),
            ("Upload Dataset", controller.upload_dataset),
            ("About Project", controller.show_about),
            ("Exit", controller.destroy),
        )

        for text, command in buttons:
            tk.Button(
                button_frame,
                text=text,
                font=("Arial", 13, "bold"),
                width=25,
                height=2,
                command=command,
            ).pack(pady=8)


class ChurnApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Explainable Customer Churn Prediction System")
        self.geometry("1200x750")
        self.minsize(1000, 650)

        self.dataset = None
        self.target_column = None
        self.dataset_name = None
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None

        container = tk.Frame(self)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for page_name, page_class in (
            ("HomePage", HomePage),
            ("DatasetPage", DatasetPage),
            ("TrainingPage", TrainingPage),
            ("AnalyticsPage", AnalyticsPage),
        ):
            page = page_class(container, self)
            self.frames[page_name] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_page("HomePage")

    def show_page(self, page_name):
        if page_name not in self.frames:
            raise ValueError(f"Unknown page: {page_name}")
        self.frames[page_name].tkraise()

    def use_ibm_dataset(self):
        dataset_page = self.frames["DatasetPage"]
        dataset_page.load_ibm_dataset()
        if dataset_page.df is not None:
            self.show_page("DatasetPage")

    def upload_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=(
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx;*.xls"),
                ("All Files", "*.*"),
            ),
        )
        if not file_path:
            return

        dataset_page = self.frames["DatasetPage"]
        dataset_page.load_uploaded_dataset(file_path)
        if dataset_page.df is not None:
            self.show_page("DatasetPage")

    def show_about(self):
        messagebox.showinfo(
            "About Project",
            "Explainable Customer Churn Prediction System\n\n"
            "This application compares machine-learning models to predict "
            "telecom customer churn and provides dataset analytics.",
        )


# ==============================================
# RUN APPLICATION
# ==============================================

if __name__ == "__main__":
    app = ChurnApp()
    app.mainloop()
