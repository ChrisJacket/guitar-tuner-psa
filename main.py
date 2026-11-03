"""
main.py
=======
Application entry point for the Python Guitar Tuner.
Launches the GUI, not much else.
"""
import os
import sys

from gui.main_window import MainWindow
from tunings_manager import TuningsManager

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tunings_file = os.path.join(base_dir, "tunings.json")
    tunings_mgr = TuningsManager(tunings_file)
    app = MainWindow(tunings_mgr)
    app.mainloop()


if __name__ == "__main__":
    main()
