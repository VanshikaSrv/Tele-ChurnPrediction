import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class ChurnApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Explainable Customer Churn Prediction System")
        self.geometry("900x600")
        self.resizable(False, False)

        self.show_home()

    # ---------------- HOME PAGE ----------------
    def show_home(self):

        # Clear existing widgets
        for widget in self.winfo_children():
            widget.destroy()

        # Main frame
        main_frame = tk.Frame(self, bg="#F4F6F8")
        main_frame.pack(fill="both", expand=True)

        # ---------------- TITLE ----------------
        title = tk.Label(
            main_frame,
            text="Explainable Customer Churn\nPrediction System",
            font=("Arial", 28, "bold"),
            bg="#F4F6F8",
            fg="#1F2937"
        )
        title.pack(pady=(70, 15))

        # Subtitle
        subtitle = tk.Label(
            main_frame,
            text="Machine Learning Based Telecom Customer Churn Analysis",
            font=("Arial", 13),
            bg="#F4F6F8",
            fg="#555555"
        )
        subtitle.pack(pady=(0, 40))

        # ---------------- BUTTON FRAME ----------------
        button_frame = tk.Frame(
            main_frame,
            bg="#F4F6F8"
        )
        button_frame.pack()

        # IBM Dataset button
        ibm_button = tk.Button(
            button_frame,
            text="Use IBM Dataset",
            font=("Arial", 13, "bold"),
            width=25,
            height=2,
            command=self.use_ibm_dataset
        )
        ibm_button.pack(pady=8)

        # Upload Dataset button
        upload_button = tk.Button(
            button_frame,
            text="Upload Dataset",
            font=("Arial", 13, "bold"),
            width=25,
            height=2,
            command=self.upload_dataset
        )
        upload_button.pack(pady=8)

        # About button
        about_button = tk.Button(
            button_frame,
            text="About Project",
            font=("Arial", 13, "bold"),
            width=25,
            height=2,
            command=self.about_project
        )
        about_button.pack(pady=8)

        # Exit button
        exit_button = tk.Button(
            button_frame,
            text="Exit",
            font=("Arial", 13, "bold"),
            width=25,
            height=2,
            command=self.destroy
        )
        exit_button.pack(pady=8)

    # ---------------- IBM DATASET ----------------
    def use_ibm_dataset(self):

        messagebox.showinfo(
            "IBM Dataset",
            "IBM Telco Customer Churn Dataset selected!"
        )

        # Part 2 will open Dataset Information page here.

    # ---------------- UPLOAD DATASET ----------------
    def upload_dataset(self):

        file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx"),
                ("Excel Files", "*.xls"),
                ("All Files", "*.*")
            ]
        )

        if file_path:

            messagebox.showinfo(
                "Dataset Selected",
                "Dataset selected successfully!\n\n" + file_path
            )

            # Part 2 will process the dataset here.

    # ---------------- ABOUT PROJECT ----------------
    def about_project(self):

        messagebox.showinfo(
            "About Project",
            "Explainable Customer Churn Prediction System\n\n"
            "This project uses Machine Learning to predict "
            "whether a telecom customer is likely to churn.\n\n"
            "The system compares multiple ML models and "
            "uses SHAP to explain predictions.\n\n"
            "Developed as a Machine Learning / Data Mining Project."
        )


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    app = ChurnApp()
    app.mainloop()