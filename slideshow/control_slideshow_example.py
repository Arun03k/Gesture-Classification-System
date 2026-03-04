import requests
import random
import time

BASE_URL = "http://127.0.0.1:8800/event"


def send_command(cmd: str) -> None:
    try:
        resp = requests.post(BASE_URL, json={"command": cmd}, timeout=2.0)
        resp.raise_for_status()
        print(f"--> sent: {cmd}")
    except Exception as e:
        print(f"[Error] Could not send command: {e}")


def self_control() -> None:
    print("Slideshow Controller")
    print("--------------------")
    print("Available commands (examples):")
    print("  swipe_left, swipe_right, swipe_up, swipe_down, first")
    print("  move_left, move_right, move_up, move_down")
    print("  rotate, rotate_counter_clock, zoom_in, zoom_out, reset")
    print("Quit with: quit or q")
    print()

    while True:
        cmd = input("command> ").strip()
        if cmd.lower() in {"q", "quit", "exit"}:
            break
        if not cmd:
            continue
        send_command(cmd)


def auto_control() -> None:
    gestures = ["swipe_left", "swipe_right", "rotate"]
    while True:
        random_gesture = gestures[random.randrange(3)]
        match random_gesture:
            case "swipe_left":
                send_command("swipe_left")
                print("swipe_left")
            case "swipe_right":
                send_command("swipe_right")
                print("swipe_right")

            case "rotate":
                send_command("rotate")
                print("rotate")

        time.sleep(3)


if __name__ == "__main__":
    # auto_control()
    self_control()
