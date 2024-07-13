"""
This is the main file from which the whole project runs.
"""
import tkinter as tk
from GUI import EmotionRecognitionApp


if __name__ == "__main__":
    """
    From here the project starts running
    """

    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()


