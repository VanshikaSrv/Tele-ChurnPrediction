
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os


class DatasetPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg="#F4F6F8")

        self.controller = controller
        self.df = None
        self.dataset_name = ""
        self.uploaded = False

        # ================= TITLE =================

        title = tk.Label(
            self,
            text="Dataset Information",
            font=("Arial", 25, "bold"),
            bg="#F4F6F8",
            fg="#1F2937"
        )
        title.pack(pady=20)

        # ================= INFORMATION FRAME =================

        info_frame = tk.Frame(
            self,
            bg="white",
            padx=20,
            pady=15
        )
        info_frame.pack(fill="x", padx=40)

        self.name_label = tk.Label(
            info_frame,
            text="Dataset Name: -",
            font=("Arial", 12, "bold"),
            bg="white",
            anchor="w"
        )
        self.name_label.pack(fill="x", pady=3)

        self.rows_label = tk.Label(
            info_frame,
            text="Number of Rows: -",
            font=("Arial", 12),
            bg="white",
            anchor="w"
        )
        self.rows_label.pack(fill="x", pady=3)

        self.columns_label = tk.Label(
            info_frame,
            text="Number of Columns: -",
            font=("Arial", 12),
            bg="white",
            anchor="w"
        )
        self.columns_label.pack(fill="x", pady=3)

        self.missing_label = tk.Label(
            info_frame,
            text="Missing Values: -",
            font=("Arial", 12),
            bg="white",
            anchor="w"
        )
        self.missing_label.pack(fill="x", pady=3)

        # ================= TARGET COLUMN =================

        target_frame = tk.Frame(
            self,
            bg="#F4F6F8"
        )
        target_frame.pack(pady=10)

        self.target_label = tk.Label(
            target_frame,
            text="Target Column:",
            font=("Arial", 12, "bold"),
            bg="#F4F6F8"
        )
        self.target_label.pack(side="left", padx=10)

        self.target_combo = ttk.Combobox(
            target_frame,
            state="readonly",
            width=25
        )
        self.target_combo.pack(side="left")

        # ================= PREVIEW LABEL =================

        preview_label = tk.Label(
            self,
            text="Dataset Preview (First 10 Rows)",
            font=("Arial", 14, "bold"),
            bg="#F4F6F8"
        )
        preview_label.pack(pady=(5, 5))

        # ================= TABLE =================

        table_frame = tk.Frame(self)
        table_frame.pack(
            fill="both",
            expand=True,
            padx=40,
            pady=5
        )

        # Treeview
        self.table = ttk.Treeview(
            table_frame,
            show="headings"
        )

        # Scrollbars
        vertical_scroll = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.table.yview
        )

        horizontal_scroll = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.table.xview
        )

        self.table.configure(
            yscrollcommand=vertical_scroll.set,
            xscrollcommand=horizontal_scroll.set
        )

        self.table.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        vertical_scroll.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        horizontal_scroll.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # ================= BUTTONS =================

        button_frame = tk.Frame(
            self,
            bg="#F4F6F8"
        )
        button_frame.pack(pady=15)

        tk.Button(
            button_frame,
            text="Back",
            width=15,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.go_back
        ).pack(side="left", padx=10)

        tk.Button(
            button_frame,
            text="Continue to Training",
            width=20,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.continue_training
        ).pack(side="left", padx=10)

    # ==================================================
    # LOAD IBM DATASET
    # ==================================================

    def load_ibm_dataset(self):

        file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

        if not os.path.exists(file_path):

            messagebox.showerror(
                "File Not Found",
                "IBM dataset was not found.\n\n"
                "Please put the CSV file inside the data folder."
            )

            return

        try:

            self.df = pd.read_csv(file_path)

            self.dataset_name = os.path.basename(file_path)
            self.uploaded = False

            # IBM target column
            self.target_combo["values"] = ["Churn"]
            self.target_combo.current(0)

            self.update_page()

        except Exception as e:

            messagebox.showerror(
                "Error",
                f"Could not load dataset.\n\n{e}"
            )

    # ==================================================
    # LOAD UPLOADED DATASET
    # ==================================================

    def load_uploaded_dataset(self, file_path):

        try:

            # Read CSV
            if file_path.lower().endswith(".csv"):
                self.df = pd.read_csv(file_path)

            # Read Excel
            elif file_path.lower().endswith((".xlsx", ".xls")):
                self.df = pd.read_excel(file_path)

            else:
                messagebox.showerror(
                    "Invalid File",
                    "Please select CSV, XLS or XLSX file."
                )
                return

            self.dataset_name = os.path.basename(file_path)
            self.uploaded = True

            # All columns can potentially be target
            columns = self.df.columns.tolist()

            self.target_combo["values"] = columns

            if columns:
                self.target_combo.current(0)

            self.update_page()

        except Exception as e:

            messagebox.showerror(
                "Error",
                f"Could not load dataset.\n\n{e}"
            )

    # ==================================================
    # UPDATE INFORMATION
    # ==================================================

    def update_page(self):

        # Dataset name
        self.name_label.config(
            text=f"Dataset Name: {self.dataset_name}"
        )

        # Rows
        self.rows_label.config(
            text=f"Number of Rows: {self.df.shape[0]}"
        )

        # Columns
        self.columns_label.config(
            text=f"Number of Columns: {self.df.shape[1]}"
        )

        # Missing values
        missing_values = self.df.isnull().sum().sum()

        self.missing_label.config(
            text=f"Missing Values: {missing_values}"
        )

        # Update table
        self.display_table()

    # ==================================================
    # DISPLAY FIRST 10 ROWS
    # ==================================================

    def display_table(self):

        # Clear old table
        for item in self.table.get_children():
            self.table.delete(item)

        # Clear columns
        self.table["columns"] = list(self.df.columns)

        # Create headings
        for column in self.df.columns:

            self.table.heading(
                column,
                text=column
            )

            self.table.column(
                column,
                width=120,
                minwidth=80
            )

        # Insert first 10 rows
        for _, row in self.df.head(10).iterrows():

            values = []

            for value in row:

                values.append(str(value))

            self.table.insert(
                "",
                "end",
                values=values
            )

    # ==================================================
    # BACK
    # ==================================================

    def go_back(self):

        self.controller.show_page(
            "HomePage"
        )

    # ==================================================
    # CONTINUE
    # ==================================================

    def continue_training(self):

        if self.df is None:

            messagebox.showwarning(
                "No Dataset",
                "Please select a dataset first."
            )

            return

        target = self.target_combo.get()

        if target == "":

            messagebox.showwarning(
                "Target Column",
                "Please select the target column."
            )

            return

        # Save information for next page
        self.controller.dataset = self.df
        self.controller.target_column = target
        self.controller.dataset_name = self.dataset_name

        messagebox.showinfo(
            "Dataset Ready",
            f"Target Column: {target}\n\n"
            "Ready to continue to model training."
        )

        # Training page will be connected in Part 3.

