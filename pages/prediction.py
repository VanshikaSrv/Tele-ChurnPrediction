import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import numpy as np

from .explainability import get_top_reasons, generate_explanation


class PredictionPage(tk.Frame):

    def __init__(self, parent, controller):

        super().__init__(
            parent,
            bg="#F4F6F8"
        )

        self.controller = controller

        self.df = None
        self.entries = {}
        self.input_columns = []

        # ==========================================
        # TITLE
        # ==========================================

        title = tk.Label(
            self,
            text="Customer Churn Prediction",
            font=("Arial", 25, "bold"),
            bg="#F4F6F8",
            fg="#1F2937"
        )

        title.pack(pady=15)

        # ==========================================
        # FORM FRAME
        # ==========================================

        self.form_frame = tk.Frame(
            self,
            bg="white"
        )

        self.form_frame.pack(
            fill="both",
            expand=True,
            padx=40,
            pady=10
        )

        # ==========================================
        # BUTTON FRAME
        # ==========================================

        button_frame = tk.Frame(
            self,
            bg="#F4F6F8"
        )

        button_frame.pack(
            pady=15
        )

        # ==========================================
        # PREDICT BUTTON
        # ==========================================

        tk.Button(
            button_frame,
            text="Predict",
            width=15,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.predict
        ).pack(
            side="left",
            padx=10
        )

        # ==========================================
        # EXPLAIN PREDICTION BUTTON
        # ==========================================

        tk.Button(
            button_frame,
            text="Explain Prediction",
            width=18,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.explain_prediction
        ).pack(
            side="left",
            padx=10
        )

        # ==========================================
        # RESET BUTTON
        # ==========================================

        tk.Button(
            button_frame,
            text="Reset",
            width=15,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.reset_form
        ).pack(
            side="left",
            padx=10
        )

        # ==========================================
        # BACK BUTTON
        # ==========================================

        tk.Button(
            button_frame,
            text="Back",
            width=15,
            height=2,
            font=("Arial", 11, "bold"),
            command=lambda:
                controller.show_page("AnalyticsPage")
        ).pack(
            side="left",
            padx=10
        )

    # ==========================================
    # LOAD PREDICTION FORM
    # ==========================================

    def load_prediction_form(self):

        self.df = self.controller.dataset

        if self.df is None:

            messagebox.showwarning(
                "No Dataset",
                "Please select a dataset first."
            )

            return

        # Clear old form
        for widget in self.form_frame.winfo_children():
            widget.destroy()

        self.entries = {}

        target = self.controller.target_column

        # Remove target column from input columns
        self.input_columns = [
            column
            for column in self.df.columns
            if column != target
        ]

        # ==========================================
        # CREATE SCROLLABLE AREA
        # ==========================================

        canvas = tk.Canvas(
            self.form_frame,
            bg="white"
        )

        scrollbar = ttk.Scrollbar(
            self.form_frame,
            orient="vertical",
            command=canvas.yview
        )

        scroll_frame = tk.Frame(
            canvas,
            bg="white"
        )

        scroll_frame.bind(
            "<Configure>",
            lambda event:
            canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window(
            (0, 0),
            window=scroll_frame,
            anchor="nw"
        )

        canvas.configure(
            yscrollcommand=scrollbar.set
        )

        canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        # ==========================================
        # CREATE INPUT FIELDS
        # ==========================================

        for row, column in enumerate(self.input_columns):

            label = tk.Label(
                scroll_frame,
                text=column,
                font=("Arial", 11, "bold"),
                bg="white"
            )

            label.grid(
                row=row,
                column=0,
                padx=20,
                pady=8,
                sticky="w"
            )

            # ======================================
            # NUMERICAL COLUMN
            # ======================================

            if pd.api.types.is_numeric_dtype(
                self.df[column]
            ):

                entry = tk.Entry(
                    scroll_frame,
                    width=35,
                    font=("Arial", 11)
                )

                entry.grid(
                    row=row,
                    column=1,
                    padx=20,
                    pady=8
                )

                # Put median value as default
                median_value = self.df[column].median()

                if pd.notna(median_value):

                    entry.insert(
                        0,
                        str(round(median_value, 2))
                    )

                self.entries[column] = entry

            # ======================================
            # CATEGORICAL COLUMN
            # ======================================

            else:

                values = (
                    self.df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                values.sort()

                combo = ttk.Combobox(
                    scroll_frame,
                    values=values,
                    width=32,
                    state="readonly"
                )

                combo.grid(
                    row=row,
                    column=1,
                    padx=20,
                    pady=8
                )

                if len(values) > 0:

                    combo.current(0)

                self.entries[column] = combo

    # ==========================================
    # PREDICT
    # ==========================================

    def predict(self):

        model = self.controller.best_model

        if model is None:

            messagebox.showwarning(
                "Model Not Trained",
                "Please train the models first."
            )

            return

        if self.df is None:

            messagebox.showwarning(
                "No Dataset",
                "Please load a dataset first."
            )

            return

        # ==========================================
        # COLLECT USER INPUT
        # ==========================================

        input_data = {}

        for column in self.input_columns:

            widget = self.entries[column]

            value = widget.get().strip()

            if value == "":

                messagebox.showwarning(
                    "Missing Value",
                    "Please enter/select a value for "
                    + column
                )

                return

            input_data[column] = value

        # ==========================================
        # CREATE DATAFRAME
        # ==========================================

        input_df = pd.DataFrame(
            [input_data]
        )

        # ==========================================
        # CONVERT NUMERICAL COLUMNS
        # ==========================================

        for column in self.input_columns:

            if pd.api.types.is_numeric_dtype(
                self.df[column]
            ):

                input_df[column] = pd.to_numeric(
                    input_df[column],
                    errors="coerce"
                )

        # ==========================================
        # MAKE PREDICTION
        # ==========================================

        try:

            prediction = model.predict(
                input_df
            )[0]

            # ======================================
            # PROBABILITY
            # ======================================

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    input_df
                )[0]

                # Find probability of class 1
                if hasattr(model, "classes_"):

                    classes = list(
                        model.classes_
                    )

                    if 1 in classes:

                        churn_index = classes.index(1)

                        probability = float(
                            probabilities[churn_index]
                        )

                    else:

                        probability = float(
                            np.max(probabilities)
                        )

                else:

                    probability = float(
                        np.max(probabilities)
                    )

            else:

                probability = 0.0

            # ======================================
            # DETERMINE RESULT
            # ======================================

            if prediction == 1:

                result = "CHURN"

            else:

                result = "NO CHURN"

            # ======================================
            # RISK LEVEL
            # ======================================

            if probability >= 0.75:

                risk = "High"

            elif probability >= 0.50:

                risk = "Medium"

            else:

                risk = "Low"

            # ======================================
            # SAVE RESULT
            # ======================================

            self.controller.prediction_result = result

            self.controller.prediction_probability = probability

            self.controller.prediction_risk = risk

            self.controller.prediction_input = input_df

            # ======================================
            # SHOW RESULT
            # ======================================

            messagebox.showinfo(
                "Prediction Result",
                "Prediction: "
                + result
                + "\n\nProbability: "
                + str(
                    round(
                        probability * 100,
                        2
                    )
                )
                + "%"
                + "\n\nRisk Level: "
                + risk
            )

        except Exception as e:

            messagebox.showerror(
                "Prediction Error",
                "Could not make prediction.\n\n"
                + str(e)
            )

    # ==========================================
    # EXPLAIN PREDICTION
    # ==========================================

    def explain_prediction(self):

        # ==========================================
        # CHECK WHETHER PREDICTION WAS MADE
        # ==========================================

        if self.controller.prediction_input is None:

            messagebox.showwarning(
                "No Prediction",
                "Please make a prediction first."
            )

            return

        try:

            # ======================================
            # GET CUSTOMER DATA
            # ======================================

            customer_data = (
                self.controller
                .prediction_input
                .iloc[0]
                .to_dict()
            )

            # ======================================
            # GET TOP FEATURES
            # ======================================

            top_features = get_top_reasons(
                customer_data,
                top_n=5
            )

            # ======================================
            # GENERATE EXPLANATION
            # ======================================

            explanation = generate_explanation(
                top_features
            )

            # ======================================
            # CREATE EXPLANATION WINDOW
            # ======================================

            explanation_window = tk.Toplevel(self)

            explanation_window.title(
                "Prediction Explanation"
            )

            explanation_window.geometry(
                "750x600"
            )

            explanation_window.configure(
                bg="#F4F6F8"
            )

            # ======================================
            # TITLE
            # ======================================

            tk.Label(
                explanation_window,
                text="Prediction Explanation",
                font=("Arial", 22, "bold"),
                bg="#F4F6F8",
                fg="#1F2937"
            ).pack(
                pady=20
            )

            # ======================================
            # RESULT
            # ======================================

            result = (
                self.controller
                .prediction_result
            )

            probability = (
                self.controller
                .prediction_probability
            )

            risk = (
                self.controller
                .prediction_risk
            )

            result_text = (
                "Prediction: "
                + str(result)
                + "\n"
                + "Probability: "
                + str(
                    round(
                        probability * 100,
                        2
                    )
                )
                + "%\n"
                + "Risk Level: "
                + str(risk)
            )

            tk.Label(
                explanation_window,
                text=result_text,
                font=("Arial", 13, "bold"),
                bg="white",
                justify="left",
                padx=20,
                pady=15
            ).pack(
                fill="x",
                padx=30
            )

            # ======================================
            # TOP FEATURES
            # ======================================

            tk.Label(
                explanation_window,
                text="Top Influencing Factors",
                font=("Arial", 16, "bold"),
                bg="#F4F6F8",
                fg="#1F2937"
            ).pack(
                pady=(20, 10)
            )

            # ======================================
            # TREEVIEW
            # ======================================

            table_frame = tk.Frame(
                explanation_window,
                bg="#F4F6F8"
            )

            table_frame.pack(
                fill="both",
                expand=False,
                padx=30
            )

            columns = (
                "Feature",
                "Value",
                "Impact",
                "Direction"
            )

            tree = ttk.Treeview(
                table_frame,
                columns=columns,
                show="headings",
                height=6
            )

            tree.heading(
                "Feature",
                text="Feature"
            )

            tree.heading(
                "Value",
                text="Value"
            )

            tree.heading(
                "Impact",
                text="Impact"
            )

            tree.heading(
                "Direction",
                text="Direction"
            )

            tree.column(
                "Feature",
                width=230
            )

            tree.column(
                "Value",
                width=100
            )

            tree.column(
                "Impact",
                width=100
            )

            tree.column(
                "Direction",
                width=180
            )

            tree.pack(
                fill="x"
            )

            # ======================================
            # INSERT TOP FEATURES
            # ======================================

            for _, row in top_features.iterrows():

                tree.insert(
                    "",
                    "end",
                    values=(
                        str(row["Feature"]),
                        str(
                            round(
                                float(row["Value"]),
                                3
                            )
                        ),
                        str(
                            round(
                                float(row["Impact"]),
                                3
                            )
                        ),
                        str(row["Direction"])
                    )
                )

            # ======================================
            # HUMAN READABLE EXPLANATION
            # ======================================

            tk.Label(
                explanation_window,
                text="Explanation",
                font=("Arial", 16, "bold"),
                bg="#F4F6F8",
                fg="#1F2937"
            ).pack(
                pady=(20, 8)
            )

            explanation_text = tk.Text(
                explanation_window,
                height=8,
                width=75,
                font=("Arial", 11),
                wrap="word"
            )

            explanation_text.pack(
                padx=30,
                pady=(0, 20)
            )

            explanation_text.insert(
                "1.0",
                explanation
            )

            explanation_text.config(
                state="disabled"
            )

            # ======================================
            # CLOSE BUTTON
            # ======================================

            tk.Button(
                explanation_window,
                text="Close",
                width=15,
                height=2,
                font=("Arial", 11, "bold"),
                command=explanation_window.destroy
            ).pack(
                pady=10
            )

        except Exception as e:

            messagebox.showerror(
                "Explanation Error",
                "Could not generate explanation.\n\n"
                + str(e)
            )

    # ==========================================
    # RESET
    # ==========================================

    def reset_form(self):

        for column, widget in self.entries.items():

            if isinstance(
                widget,
                ttk.Combobox
            ):

                if len(widget["values"]) > 0:

                    widget.current(0)

            else:

                widget.delete(
                    0,
                    tk.END
                )

                # Put median again
                median_value = (
                    self.df[column].median()
                )

                if pd.notna(median_value):

                    widget.insert(
                        0,
                        str(
                            round(
                                median_value,
                                2
                            )
                        )
                    )

        # Clear previous prediction
        self.controller.prediction_result = None
        self.controller.prediction_probability = None
        self.controller.prediction_risk = None
        self.controller.prediction_input = None