
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class TrainingPage(tk.Frame):

    def __init__(self, parent, controller):

        super().__init__(
            parent,
            bg="#F4F6F8"
        )

        self.controller = controller

        # Store trained models
        self.trained_models = {}

        # Best model
        self.best_model = None
        self.best_model_name = None

        # ================= TITLE =================

        title = tk.Label(
            self,
            text="Model Training & Comparison",
            font=("Arial", 25, "bold"),
            bg="#F4F6F8",
            fg="#1F2937"
        )

        title.pack(pady=20)

        # ================= STATUS =================

        self.status_label = tk.Label(
            self,
            text="Click 'Train Models' to start training.",
            font=("Arial", 12),
            bg="#F4F6F8",
            fg="#555555"
        )

        self.status_label.pack(pady=5)

        # ================= TABLE FRAME =================

        table_frame = tk.Frame(
            self,
            bg="white"
        )

        table_frame.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=20
        )

        # ================= TABLE =================

        columns = (
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Training Time"
        )

        self.table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=10
        )

        for column in columns:

            self.table.heading(
                column,
                text=column
            )

            self.table.column(
                column,
                width=130,
                anchor="center"
            )

        self.table.pack(
            fill="both",
            expand=True
        )

        # ================= BEST MODEL =================

        self.best_frame = tk.Frame(
            self,
            bg="white",
            padx=20,
            pady=15
        )

        self.best_frame.pack(
            fill="x",
            padx=30,
            pady=5
        )

        self.best_label = tk.Label(
            self.best_frame,
            text="Best Model: -",
            font=("Arial", 14, "bold"),
            bg="white",
            fg="#1F2937"
        )

        self.best_label.pack()

        self.best_accuracy_label = tk.Label(
            self.best_frame,
            text="F1 Score: -",
            font=("Arial", 12),
            bg="white"
        )

        self.best_accuracy_label.pack()

        # ================= BUTTONS =================

        button_frame = tk.Frame(
            self,
            bg="#F4F6F8"
        )

        button_frame.pack(
            pady=15
        )

        # Back button
        tk.Button(
            button_frame,
            text="Back",
            width=15,
            height=2,
            font=("Arial", 11, "bold"),
            command=lambda: controller.show_page("DatasetPage")
        ).pack(
            side="left",
            padx=10
        )

        # Train button
        tk.Button(
            button_frame,
            text="Train Models",
            width=18,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.train_models
        ).pack(
            side="left",
            padx=10
        )

        # Save button
        tk.Button(
            button_frame,
            text="Save Model",
            width=15,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.save_model
        ).pack(
            side="left",
            padx=10
        )

        # Continue button
        tk.Button(
            button_frame,
            text="Continue to Analytics",
            width=22,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.continue_to_analytics
        ).pack(
            side="left",
            padx=10
        )

    # ==================================================
    # PREPARE TARGET
    # ==================================================

    def prepare_target(self, y):

        """
        Convert target values into 0 and 1.
        """

        # If target is numeric
        if pd.api.types.is_numeric_dtype(y):

            unique_values = y.dropna().unique()

            if len(unique_values) == 2:

                values = sorted(unique_values)

                return y.map({
                    values[0]: 0,
                    values[1]: 1
                })

        # Convert to string
        y = y.astype(str).str.strip().str.lower()

        # Common churn labels
        positive_values = [
            "yes",
            "y",
            "true",
            "1",
            "churn",
            "churned",
            "exit",
            "exited",
            "attrition",
            "left"
        ]

        result = y.apply(
            lambda x: 1 if x in positive_values else 0
        )

        return result

    # ==================================================
    # TRAIN MODELS
    # ==================================================

    def train_models(self):

        df = self.controller.dataset
        target_column = self.controller.target_column

        if df is None:

            messagebox.showwarning(
                "No Dataset",
                "Please select a dataset first."
            )

            return

        if target_column not in df.columns:

            messagebox.showerror(
                "Error",
                "Target column was not found."
            )

            return

        try:

            self.status_label.config(
                text="Preparing dataset..."
            )

            self.update_idletasks()

            # ------------------------------------------
            # Separate X and y
            # ------------------------------------------

            X = df.drop(
                columns=[target_column]
            ).copy()

            y = df[target_column].copy()

            # Remove ID columns that are not useful
            id_columns = []

            for column in X.columns:

                if column.lower() in [
                    "customerid",
                    "customer_id",
                    "id"
                ]:

                    id_columns.append(column)

            if id_columns:

                X = X.drop(
                    columns=id_columns
                )

            # ------------------------------------------
            # Convert target
            # ------------------------------------------

            y = self.prepare_target(y)

            # Remove rows where target is missing
            valid_rows = y.notna()

            X = X.loc[valid_rows]
            y = y.loc[valid_rows]

            # ------------------------------------------
            # Check target
            # ------------------------------------------

            if y.nunique() != 2:

                messagebox.showerror(
                    "Invalid Target",
                    "The target column must contain exactly "
                    "two classes for binary churn prediction."
                )

                return

            # ------------------------------------------
            # Identify columns
            # ------------------------------------------

            numerical_columns = X.select_dtypes(
                include=["int64", "float64", "int32", "float32"]
            ).columns.tolist()

            categorical_columns = X.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

            # ------------------------------------------
            # Preprocessing
            # ------------------------------------------

            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numeric",
                        numerical_pipeline,
                        numerical_columns
                    ),
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_columns
                    )
                ]
            )

            # ------------------------------------------
            # Train / Test Split
            # ------------------------------------------

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

            # ------------------------------------------
            # Models
            # ------------------------------------------

            models = {

                "Logistic Regression":
                    LogisticRegression(
                        max_iter=1000
                    ),

                "Decision Tree":
                    DecisionTreeClassifier(
                        random_state=42
                    ),

                "Random Forest":
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    ),

                "KNN":
                    KNeighborsClassifier(
                        n_neighbors=5
                    )
            }

            # Clear previous table
            for item in self.table.get_children():

                self.table.delete(item)

            self.trained_models = {}

            results = []

            # ------------------------------------------
            # Train each model
            # ------------------------------------------

            for model_name, model in models.items():

                self.status_label.config(
                    text=f"Training {model_name}..."
                )

                self.update_idletasks()

                start_time = time.time()

                pipeline = Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            preprocessor
                        ),
                        (
                            "model",
                            model
                        )
                    ]
                )

                pipeline.fit(
                    X_train,
                    y_train
                )

                training_time = time.time() - start_time

                # Prediction
                y_pred = pipeline.predict(X_test)

                # Probability
                if hasattr(
                    pipeline,
                    "predict_proba"
                ):

                    y_probability = pipeline.predict_proba(
                        X_test
                    )[:, 1]

                    roc_auc = roc_auc_score(
                        y_test,
                        y_probability
                    )

                else:

                    roc_auc = 0

                # Metrics
                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                precision = precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                # Store
                self.trained_models[
                    model_name
                ] = pipeline

                results.append(
                    {
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1,
                        "ROC-AUC": roc_auc,
                        "Time": training_time
                    }
                )

            # ------------------------------------------
            # Find best model
            # ------------------------------------------

            results.sort(
                key=lambda x: x["F1"],
                reverse=True
            )

            best = results[0]

            self.best_model_name = best["Model"]

            self.best_model = self.trained_models[
                self.best_model_name
            ]

            # ------------------------------------------
            # Display results
            # ------------------------------------------

            for result in results:

                self.table.insert(
                    "",
                    "end",
                    values=(
                        result["Model"],
                        f"{result['Accuracy'] * 100:.2f}%",
                        f"{result['Precision'] * 100:.2f}%",
                        f"{result['Recall'] * 100:.2f}%",
                        f"{result['F1'] * 100:.2f}%",
                        f"{result['ROC-AUC']:.3f}",
                        f"{result['Time']:.3f}s"
                    )
                )

            # Best model information
            self.best_label.config(
                text=f"Best Model: {self.best_model_name}"
            )

            self.best_accuracy_label.config(
                text=f"F1 Score: {best['F1'] * 100:.2f}%"
            )

            self.status_label.config(
                text="Model training completed successfully."
            )

            # Save information
            self.controller.best_model = self.best_model
            self.controller.best_model_name = self.best_model_name

            self.controller.preprocessor = (
                self.best_model.named_steps["preprocessor"]
            )

            messagebox.showinfo(
                "Training Complete",
                f"Training completed successfully!\n\n"
                f"Best Model: {self.best_model_name}\n"
                f"F1 Score: {best['F1'] * 100:.2f}%"
            )

        except Exception as e:

            messagebox.showerror(
                "Training Error",
                f"An error occurred while training.\n\n{e}"
            )

    # ==================================================
    # SAVE MODEL
    # ==================================================

    def save_model(self):

        if self.best_model is None:

            messagebox.showwarning(
                "No Model",
                "Please train the models first."
            )

            return

        try:

            import joblib
            import os

            os.makedirs(
                "models",
                exist_ok=True
            )

            joblib.dump(
                self.best_model,
                "models/best_model.pkl"
            )

            messagebox.showinfo(
                "Model Saved",
                "Best model saved successfully!\n\n"
                "models/best_model.pkl"
            )

        except Exception as e:

            messagebox.showerror(
                "Save Error",
                str(e)
            )

    # ==================================================
    # CONTINUE
    # ==================================================

    def continue_to_analytics(self):

        if self.best_model is None:

            messagebox.showwarning(
                "Train Models",
                "Please train the models before continuing."
            )

            return

        messagebox.showinfo(
            "Next Step",
            "Model training completed.\n\n"
            "Analytics Dashboard will be connected in Part 4."
        )
