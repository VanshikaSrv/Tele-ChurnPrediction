
import tkinter as tk

from pages.home import HomePage
from pages.dataset import DatasetPage


class ChurnApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title(
            "Explainable Customer Churn Prediction System"
        )

        self.geometry("1200x750")
        self.resizable(False, False)

        # Shared dataset information
        self.dataset = None
        self.target_column = None
        self.dataset_name = None

        # Container
        container = tk.Frame(self)

        container.pack(
            side="top",
            fill="both",
            expand=True
        )

        # Store pages
        self.frames = {}

        # Create Home Page
        home_page = HomePage(
            container,
            self
        )

        self.frames["HomePage"] = home_page

        home_page.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        # Create Dataset Page
        dataset_page = DatasetPage(
            container,
            self
        )

        self.frames["DatasetPage"] = dataset_page

        dataset_page.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        # Show Home
        self.show_page("HomePage")

    # ==========================================
    # CHANGE PAGE
    # ==========================================

    def show_page(self, page_name):

        frame = self.frames[page_name]

        frame.tkraise()


if __name__ == "__main__":

    app = ChurnApp()

    app.mainloop()
