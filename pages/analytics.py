
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class AnalyticsPage(tk.Frame):

    def __init__(self, parent, controller):

        super().__init__(
            parent,
            bg="#F4F6F8"
        )

        self.controller = controller
        self.df = None

        # ================= TITLE =================

        title = tk.Label(
            self,
            text="Dataset Analytics Dashboard",
            font=("Arial", 25, "bold"),
            bg="#F4F6F8",
            fg="#1F2937"
        )

        title.pack(pady=15)

        # ================= GRAPH AREA =================

        self.graph_frame = tk.Frame(
            self,
            bg="white"
        )

        self.graph_frame.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=5
        )

        # ================= BUTTONS =================

        button_frame = tk.Frame(
            self,
            bg="#F4F6F8"
        )

        button_frame.pack(
            pady=10
        )

        # Back
        tk.Button(
            button_frame,
            text="Back",
            width=15,
            height=2,
            font=("Arial", 11, "bold"),
            command=lambda:
                controller.show_page("TrainingPage")
        ).pack(
            side="left",
            padx=10
        )

        # Refresh
        tk.Button(
            button_frame,
            text="Refresh Analytics",
            width=18,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.load_analytics
        ).pack(
            side="left",
            padx=10
        )

        # Prediction
        tk.Button(
            button_frame,
            text="Prediction",
            width=15,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.go_to_prediction
        ).pack(
            side="left",
            padx=10
        )

    # ==================================================
    # LOAD ANALYTICS
    # ==================================================

    def load_analytics(self):

        self.df = self.controller.dataset

        if self.df is None:

            messagebox.showwarning(
                "No Dataset",
                "Please select a dataset first."
            )

            return

        # Clear previous graphs
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Create notebook
        notebook = ttk.Notebook(
            self.graph_frame
        )

        notebook.pack(
            fill="both",
            expand=True
        )

        # Create six graph tabs
        self.create_churn_chart(notebook)
        self.create_contract_chart(notebook)
        self.create_internet_chart(notebook)
        self.create_monthly_charges_chart(notebook)
        self.create_tenure_chart(notebook)
        self.create_correlation_chart(notebook)

    # ==================================================
    # GRAPH 1 - CHURN DISTRIBUTION
    # ==================================================

    def create_churn_chart(self, notebook):

        frame = tk.Frame(
            notebook,
            bg="white"
        )

        notebook.add(
            frame,
            text="Churn Distribution"
        )

        target = self.controller.target_column

        if target not in self.df.columns:
            return

        values = self.df[target].astype(str)

        counts = values.value_counts()

        figure = Figure(
            figsize=(7, 4.5),
            dpi=100
        )

        ax = figure.add_subplot(111)

        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title(
            "Churn Distribution"
        )

        canvas = FigureCanvasTkAgg(
            figure,
            master=frame
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    # ==================================================
    # GRAPH 2 - CONTRACT TYPE
    # ==================================================

    def create_contract_chart(self, notebook):

        frame = tk.Frame(
            notebook,
            bg="white"
        )

        notebook.add(
            frame,
            text="Contract Type"
        )

        # Find contract column
        contract_column = self.find_column(
            [
                "Contract",
                "contract",
                "Contract Type",
                "contract_type"
            ]
        )

        if contract_column is None:

            self.show_no_data(
                frame,
                "Contract column not found."
            )

            return

        counts = self.df[
            contract_column
        ].astype(str).value_counts()

        figure = Figure(
            figsize=(7, 4.5),
            dpi=100
        )

        ax = figure.add_subplot(111)

        ax.bar(
            counts.index,
            counts.values
        )

        ax.set_title(
            "Contract Type Distribution"
        )

        ax.set_xlabel(
            "Contract Type"
        )

        ax.set_ylabel(
            "Number of Customers"
        )

        ax.tick_params(
            axis="x",
            rotation=15
        )

        figure.tight_layout()

        canvas = FigureCanvasTkAgg(
            figure,
            master=frame
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    # ==================================================
    # GRAPH 3 - INTERNET SERVICE
    # ==================================================

    def create_internet_chart(self, notebook):

        frame = tk.Frame(
            notebook,
            bg="white"
        )

        notebook.add(
            frame,
            text="Internet Service"
        )

        internet_column = self.find_column(
            [
                "InternetService",
                "Internet Service",
                "internet_service"
            ]
        )

        if internet_column is None:

            self.show_no_data(
                frame,
                "Internet Service column not found."
            )

            return

        counts = self.df[
            internet_column
        ].astype(str).value_counts()

        figure = Figure(
            figsize=(7, 4.5),
            dpi=100
        )

        ax = figure.add_subplot(111)

        ax.bar(
            counts.index,
            counts.values
        )

        ax.set_title(
            "Internet Service Distribution"
        )

        ax.set_xlabel(
            "Internet Service"
        )

        ax.set_ylabel(
            "Number of Customers"
        )

        ax.tick_params(
            axis="x",
            rotation=15
        )

        figure.tight_layout()

        canvas = FigureCanvasTkAgg(
            figure,
            master=frame
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    # ==================================================
    # GRAPH 4 - MONTHLY CHARGES
    # ==================================================

    def create_monthly_charges_chart(self, notebook):

        frame = tk.Frame(
            notebook,
            bg="white"
        )

        notebook.add(
            frame,
            text="Monthly Charges"
        )

        column = self.find_column(
            [
                "MonthlyCharges",
                "Monthly Charges",
                "monthly_charges"
            ]
        )

        if column is None:

            self.show_no_data(
                frame,
                "Monthly Charges column not found."
            )

            return

        data = pd.to_numeric(
            self.df[column],
            errors="coerce"
        ).dropna()

        figure = Figure(
            figsize=(7, 4.5),
            dpi=100
        )

        ax = figure.add_subplot(111)

        ax.hist(
            data,
            bins=20
        )

        ax.set_title(
            "Monthly Charges Distribution"
        )

        ax.set_xlabel(
            "Monthly Charges"
        )

        ax.set_ylabel(
            "Number of Customers"
        )

        figure.tight_layout()

        canvas = FigureCanvasTkAgg(
            figure,
            master=frame
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    # ==================================================
    # GRAPH 5 - TENURE
    # ==================================================

    def create_tenure_chart(self, notebook):

        frame = tk.Frame(
            notebook,
            bg="white"
        )

        notebook.add(
            frame,
            text="Tenure"
        )

        column = self.find_column(
            [
                "tenure",
                "Tenure"
            ]
        )

        if column is None:

            self.show_no_data(
                frame,
                "Tenure column not found."
            )

            return

        data = pd.to_numeric(
            self.df[column],
            errors="coerce"
        ).dropna()

        figure = Figure(
            figsize=(7, 4.5),
            dpi=100
        )

        ax = figure.add_subplot(111)

        ax.hist(
            data,
            bins=20
        )

        ax.set_title(
            "Customer Tenure Distribution"
        )

        ax.set_xlabel(
            "Tenure (Months)"
        )

        ax.set_ylabel(
            "Number of Customers"
        )

        figure.tight_layout()

        canvas = FigureCanvasTkAgg(
            figure,
            master=frame
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    # ==================================================
    # GRAPH 6 - CORRELATION HEATMAP
    # ==================================================

    def create_correlation_chart(self, notebook):

        frame = tk.Frame(
            notebook,
            bg="white"
        )

        notebook.add(
            frame,
            text="Correlation Heatmap"
        )

        # Select numerical columns
        numeric_df = self.df.select_dtypes(
            include=np.number
        )

        if numeric_df.shape[1] < 2:

            self.show_no_data(
                frame,
                "Not enough numerical columns for correlation."
            )

            return

        correlation = numeric_df.corr()

        figure = Figure(
            figsize=(7, 5),
            dpi=100
        )

        ax = figure.add_subplot(111)

        image = ax.imshow(
            correlation,
            interpolation="nearest"
        )

        ax.set_title(
            "Correlation Heatmap"
        )

        # Axis labels
        ax.set_xticks(
            range(len(correlation.columns))
        )

        ax.set_yticks(
            range(len(correlation.columns))
        )

        ax.set_xticklabels(
            correlation.columns,
            rotation=45,
            ha="right"
        )

        ax.set_yticklabels(
            correlation.columns
        )

        # Display correlation values
        for i in range(
            len(correlation.columns)
        ):

            for j in range(
                len(correlation.columns)
            ):

                ax.text(
                    j,
                    i,
                    f"{correlation.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

        figure.colorbar(
            image,
            ax=ax
        )

        figure.tight_layout()

        canvas = FigureCanvasTkAgg(
            figure,
            master=frame
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    # ==================================================
    # FIND COLUMN
    # ==================================================

    def find_column(self, possible_names):

        for name in possible_names:

            if name in self.df.columns:

                return name

        # Case-insensitive search
        for column in self.df.columns:

            for name in possible_names:

                if column.lower() == name.lower():

                    return column

        return None

    # ==================================================
    # NO DATA MESSAGE
    # ==================================================

    def show_no_data(self, frame, text):

        label = tk.Label(
            frame,
            text=text,
            font=("Arial", 14),
            bg="white",
            fg="red"
        )

        label.pack(
            expand=True
        )

    # ==================================================
    # PREDICTION
    # ==================================================

    def go_to_prediction(self):

        prediction_page = self.controller.frames[
            "PredictionPage"
        ]

        prediction_page.load_prediction_form()

        self.controller.show_page(
            "PredictionPage"
        )


