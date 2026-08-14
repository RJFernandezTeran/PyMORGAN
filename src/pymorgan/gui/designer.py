import os
import subprocess
import sys


def main():
    # Find main_window.ui relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.normpath(os.path.join(current_dir, "main_window.ui"))

    # If a file is specified as an argument, use that instead
    if len(sys.argv) > 1:
        ui_path = sys.argv[1]

    cmd = ["uvx", "--from", "pyside6-essentials", "pyside6-designer", ui_path]
    print(f"Running: {' '.join(cmd)}")

    try:
        # Launching the designer
        subprocess.Popen(cmd, shell=True)
    except Exception as e:
        print(f"Error running command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
