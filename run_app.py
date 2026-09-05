"""
Cosmic Orbital Change Detection Launcher
Starts Flask Backend and Launches Web UI
"""

import os
import sys
import time
import webbrowser
import threading

def open_browser():
    time.sleep(1.5)
    print("🛰️ Opening Mission Control HUD in your default browser...")
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if curr_dir not in sys.path:
        sys.path.insert(0, curr_dir)

    print("="*65)
    print("🌌  LAUNCHING COSMOS ORBITAL CHANGE DETECTION STATION  🌌")
    print("="*65)

    threading.Thread(target=open_browser, daemon=True).start()

    from app import app
    app.run(host='0.0.0.0', port=5000, debug=False)
