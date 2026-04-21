"""
This is the main file from which the whole project runs.
"""
import tkinter as tk
from GUI import EmotionRecognitionApp


if __name__ == "__main__":
    root: tk.Tk = tk.Tk()
    app: EmotionRecognitionApp = EmotionRecognitionApp(root)
    root.mainloop()
