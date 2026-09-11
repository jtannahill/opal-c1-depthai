"""Keep Opal Composer from holding the camera while the service runs.

Only quits the app; the OpalLauncher login item and camera extension are left
alone, so reopening Composer restores stock behavior.
"""
import subprocess
import time


def running():
    return subprocess.run(["pgrep", "-x", "Opal Composer"], capture_output=True).returncode == 0


def quit_composer(timeout=10):
    if not running():
        return False
    subprocess.run(["osascript", "-e", 'tell application "Opal Composer" to quit'], capture_output=True)
    deadline = time.time() + timeout
    while running() and time.time() < deadline:
        time.sleep(0.5)
    return not running()


if __name__ == "__main__":
    print("was running:", running(), "| quit ok:", quit_composer())
