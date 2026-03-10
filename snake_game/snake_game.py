"""Grid Collector — Gesture Controlled Game (O10).

Move a box around an 8x8 grid to collect apples and avoid bombs.

Gestures
--------
  sl  (swipe left)  → move LEFT
  sr  (swipe right) → move RIGHT
  su  (swipe up)    → move UP
  sd  (swipe down)  → move DOWN

Keyboard fallback: A / ← = left,  D / → = right,  W / ↑ = up,  S / ↓ = down
                   R = restart,   Q / ESC = quit

Usage
-----
    python snake_game/snake_game.py               # gesture + webcam
    python snake_game/snake_game.py --flip        # mirror mode
    python snake_game/snake_game.py --keyboard    # keyboard-only (no model needed)
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import random
import time
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import yaml
import sys

SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from pipeline.gesture_pipeline import (
    MODEL_GAME, extract_features, forward_pass,
    load_model_artifacts, get_model_dir,
)

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_MODEL_DIR = PROJECT_DIR / "data" / "processed"
KP_YAML        = PROJECT_DIR / "notebooks" / "process_videos" / "keypoint_mapping.yml"

# ── Gesture pipeline constants (game-tuned) ───────────────────────────────
WINDOW_SIZE   = 18
HISTORY_LEN   = 5
MIN_CONF      = 0.75   # sl / sr
MIN_CONF_SU   = 0.85   # su — slightly strict to avoid false up during sd prep
MIN_CONF_SD   = 0.82   # sd — workable threshold
MIN_CONSEC    = 5
MIN_CONSEC_SU = 6
MIN_CONSEC_SD = 6
# After su or sd fires, the OPPOSITE gesture needs this higher confidence
# for OPPOSITE_LOCKOUT_SEC to block the natural "arm return" motion.
OPPOSITE_LOCKOUT_CONF = 0.94
OPPOSITE_LOCKOUT_SEC  = 1.4
BUFFER_SIZE   = WINDOW_SIZE * 3
COOLDOWN_SEC  = 0.45

# ── Grid layout ───────────────────────────────────────────────────────────
GRID_ROWS = 8
GRID_COLS = 8
CELL_SIZE = 60                # pixels per cell
GRID_PAD  = 10                # padding around grid
BOARD_W   = GRID_COLS * CELL_SIZE + 2 * GRID_PAD
BOARD_H   = GRID_ROWS * CELL_SIZE + 2 * GRID_PAD + 60
CAM_W     = 400
CAM_H     = BOARD_H

# ── Game rules ────────────────────────────────────────────────────────────
LIVES_INIT     = 3
APPLES_INIT    = 4
BOMBS_INIT     = 3
APPLES_PER_LVL = 1
BOMBS_PER_LVL  = 1
MAX_BOMBS      = 12
MAX_APPLES     = 10

# ── Colours (BGR) ─────────────────────────────────────────────────────────
COL_BG        = (25, 22, 18)
COL_GRID_LINE = (55, 50, 42)
COL_CELL      = (40, 36, 30)
COL_PLAYER    = (0, 220, 120)
COL_PLAYER_BD = (0, 180, 90)
COL_APPLE     = (30, 50, 215)
COL_APPLE_LF  = (35, 140, 60)
COL_BOMB      = (40, 40, 40)
COL_BOMB_FUSE = (50, 100, 255)
COL_BOMB_SPARK= (0, 200, 255)
COL_TEXT       = (240, 240, 240)
COL_DIM        = (130, 130, 130)
COL_GAMEOVER   = (50, 50, 220)
COL_FLASH_GOOD = (0, 255, 120)
COL_FLASH_BAD  = (0, 0, 255)

GESTURE_DISPLAY = {
    "idle": "idle",
    "sl":   "<- Left",
    "sr":   "Right ->",
    "su":   "^ Up",
    "sd":   "v Down",
}
GESTURE_COLS = {
    "sl": (0, 200, 255),
    "sr": (0, 255, 100),
    "su": (255, 255,   0),
    "sd": (100, 100, 255),
}


# ════════════════════════════════════════════════════════════════════════
#  Column name loader
# ════════════════════════════════════════════════════════════════════════
def load_column_names(yaml_path: pathlib.Path) -> list:
    with open(yaml_path, "r") as f:
        m = yaml.safe_load(f)
    all_kp = m["face"] + m["body"]
    return [f"{kp}_{d}" for kp in all_kp for d in ("x", "y", "z", "confidence")]


# ════════════════════════════════════════════════════════════════════════
#  Gesture Controller  (with per-gesture confidence / consec thresholds)
# ════════════════════════════════════════════════════════════════════════
class GestureController:
    def __init__(self):
        model_dir = get_model_dir(BASE_MODEL_DIR, MODEL_GAME)
        self.weights, self.biases, self.mean_, self.std_, self.idx_to_label = \
            load_model_artifacts(model_dir)
        self._col_names = load_column_names(KP_YAML)

        self._buffer: collections.deque = collections.deque(maxlen=BUFFER_SIZE)
        self._history: list = ["idle"] * HISTORY_LEN
        self._consec:  int  = 0
        self._last_gesture   = "idle"
        self._last_event_time  = 0.0
        self._last_event_label = "idle"
        self._last_su_fired: float = 0.0   # timestamp when su last fired
        self._last_sd_fired: float = 0.0   # timestamp when sd last fired

        self.display_gesture: str   = "idle"
        self.display_until:   float = 0.0

        print(f"[GestureController] Loaded game model — classes: "
              f"{list(self.idx_to_label.values())}")

    def process_landmarks(self, landmarks) -> str | None:
        row = {}
        joint_names_flat = [c[:-2] for c in self._col_names[::4]]
        for ji, kp_name in enumerate(joint_names_flat):
            lm = landmarks.landmark[ji]
            row[f"{kp_name}_x"]          = lm.x
            row[f"{kp_name}_y"]          = lm.y
            row[f"{kp_name}_z"]          = lm.z
            row[f"{kp_name}_confidence"] = lm.visibility
        self._buffer.append(row)

        if len(self._buffer) < WINDOW_SIZE:
            return None

        df          = pd.DataFrame(list(self._buffer))
        features    = extract_features(df)
        window_vec  = features[-WINDOW_SIZE:].flatten().reshape(1, -1)
        window_std  = (window_vec - self.mean_) / self.std_

        probs    = forward_pass(window_std, self.weights, self.biases)
        pred_idx = int(np.argmax(probs))
        conf     = float(probs[0, pred_idx])

        raw = self.idx_to_label.get(pred_idx, "idle")

        # Per-gesture confidence gate.
        # After su fires, sd (arm returning to neutral) needs very high
        # confidence for OPPOSITE_LOCKOUT_SEC — blocks false down after up.
        # Same logic in reverse for su after sd.
        now = time.time()
        if raw == "sd":
            if (now - self._last_su_fired) < OPPOSITE_LOCKOUT_SEC:
                threshold = OPPOSITE_LOCKOUT_CONF
            else:
                threshold = MIN_CONF_SD
        elif raw == "su":
            if (now - self._last_sd_fired) < OPPOSITE_LOCKOUT_SEC:
                threshold = OPPOSITE_LOCKOUT_CONF
            else:
                threshold = MIN_CONF_SU
        else:
            threshold = MIN_CONF
        if conf < threshold:
            raw = "idle"

        self._history.pop(0)
        self._history.append(raw)
        voted = Counter(self._history).most_common(1)[0][0]

        if voted != "idle":
            self._consec = self._consec + 1 if voted == self._last_gesture or self._last_gesture == "idle" else 1
            self._last_gesture = voted
            if voted == "su":
                required = MIN_CONSEC_SU
            elif voted == "sd":
                required = MIN_CONSEC_SD
            else:
                required = MIN_CONSEC
            if self._consec < required:
                voted = "idle"
        else:
            self._consec = 0
            self._last_gesture = "idle"

        fired = None
        if voted != "idle" and self._last_event_label == "idle" and \
                (now - self._last_event_time) >= COOLDOWN_SEC:
            fired = voted
            self.display_gesture = voted
            self.display_until   = now + 1.8
            self._last_event_time = now
            if voted == "su":
                self._last_su_fired = now
            elif voted == "sd":
                self._last_sd_fired = now
        self._last_event_label = voted if voted != "idle" else "idle"
        return fired

    def reset(self):
        self._history          = ["idle"] * HISTORY_LEN
        self._consec           = 0
        self._last_gesture     = "idle"
        self._last_event_label = "idle"
        self._last_su_fired    = 0.0
        self._last_sd_fired    = 0.0


# ════════════════════════════════════════════════════════════════════════
#  Grid Collector Game
# ════════════════════════════════════════════════════════════════════════
class GridCollector:
    def __init__(self):
        self.rng = random.Random()
        self.apples: set = set()
        self.bombs:  set = set()
        self.reset()

    def reset(self):
        self.score     = 0
        self.lives     = LIVES_INIT
        self.level     = 1
        self.game_over = False
        self.player_r  = GRID_ROWS // 2
        self.player_c  = GRID_COLS // 2
        self._flash: list = []
        self.apples = set()
        self.bombs  = set()
        self._place_items()

    def _place_items(self):
        """Place apples and bombs on empty cells (not on the player)."""
        occupied = {(self.player_r, self.player_c)} | self.apples | self.bombs
        n_apples = min(APPLES_INIT + (self.level - 1) * APPLES_PER_LVL, MAX_APPLES)
        n_bombs  = min(BOMBS_INIT  + (self.level - 1) * BOMBS_PER_LVL,  MAX_BOMBS)
        while len(self.apples) < n_apples:
            cell = self._random_free_cell(occupied)
            if cell is None:
                break
            self.apples.add(cell)
            occupied.add(cell)
        while len(self.bombs) < n_bombs:
            cell = self._random_free_cell(occupied)
            if cell is None:
                break
            self.bombs.add(cell)
            occupied.add(cell)

    def _random_free_cell(self, occupied: set):
        free = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)
                if (r, c) not in occupied]
        return self.rng.choice(free) if free else None

    # ── Movement ──────────────────────────────────────────────────────
    def move(self, dr: int, dc: int):
        if self.game_over:
            return
        nr = self.player_r + dr
        nc = self.player_c + dc
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
            self.player_r = nr
            self.player_c = nc
            self._check_cell()

    def _check_cell(self):
        pos = (self.player_r, self.player_c)
        now = time.time()
        if pos in self.apples:
            self.apples.discard(pos)
            self.score += 1
            self._flash.append((*pos, COL_FLASH_GOOD, now + 0.4))
            print(f"  Apple collected! Score: {self.score}")
            if not self.apples:
                self.level += 1
                print(f"  Level up! -> {self.level}")
                self._place_items()
        elif pos in self.bombs:
            self.bombs.discard(pos)
            self.lives -= 1
            self._flash.append((*pos, COL_FLASH_BAD, now + 0.5))
            print(f"  BOMB! Lives left: {self.lives}")
            if self.lives <= 0:
                self.lives = 0
                self.game_over = True

    def update(self):
        now = time.time()
        self._flash = [f for f in self._flash if f[3] > now]

    # ── Render ────────────────────────────────────────────────────────
    def render(self, canvas: np.ndarray):
        canvas[:] = COL_BG
        ox, oy = GRID_PAD, GRID_PAD

        # Draw cells
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x1 = ox + c * CELL_SIZE
                y1 = oy + r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                cv2.rectangle(canvas, (x1, y1), (x2, y2), COL_CELL, -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), COL_GRID_LINE, 1)

        # Draw apples
        for (r, c) in self.apples:
            cx = ox + c * CELL_SIZE + CELL_SIZE // 2
            cy = oy + r * CELL_SIZE + CELL_SIZE // 2
            rad = CELL_SIZE // 3
            cv2.circle(canvas, (cx, cy), rad, COL_APPLE, -1)
            cv2.circle(canvas, (cx, cy), rad, (100, 100, 220), 1)
            cv2.line(canvas, (cx, cy - rad), (cx + 5, cy - rad - 7), COL_APPLE_LF, 2)
            cv2.line(canvas, (cx, cy - rad), (cx, cy - rad - 4), (40, 80, 40), 2)

        # Draw bombs
        spark_on = int(time.time() * 6) % 2 == 0
        for (r, c) in self.bombs:
            cx = ox + c * CELL_SIZE + CELL_SIZE // 2
            cy = oy + r * CELL_SIZE + CELL_SIZE // 2
            rad = CELL_SIZE // 3
            cv2.circle(canvas, (cx, cy), rad, COL_BOMB, -1)
            cv2.circle(canvas, (cx, cy), rad, (120, 120, 120), 1)
            cv2.line(canvas, (cx, cy - rad), (cx + 4, cy - rad - 8), COL_BOMB_FUSE, 2)
            if spark_on:
                cv2.circle(canvas, (cx + 4, cy - rad - 9), 3, COL_BOMB_SPARK, -1)

        # Draw player box
        px = ox + self.player_c * CELL_SIZE
        py = oy + self.player_r * CELL_SIZE
        m = 4
        cv2.rectangle(canvas, (px + m, py + m),
                      (px + CELL_SIZE - m, py + CELL_SIZE - m), COL_PLAYER, -1)
        cv2.rectangle(canvas, (px + m, py + m),
                      (px + CELL_SIZE - m, py + CELL_SIZE - m), COL_PLAYER_BD, 2)
        eye_y = py + CELL_SIZE // 2 - 4
        cv2.circle(canvas, (px + CELL_SIZE // 2 - 8, eye_y), 3, (20, 20, 20), -1)
        cv2.circle(canvas, (px + CELL_SIZE // 2 + 8, eye_y), 3, (20, 20, 20), -1)
        cv2.ellipse(canvas, (px + CELL_SIZE // 2, py + CELL_SIZE // 2 + 6),
                    (7, 4), 0, 0, 180, (20, 20, 20), 1)

        # Flash effects
        for (fr, fc, col, _) in self._flash:
            fx = ox + fc * CELL_SIZE + CELL_SIZE // 2
            fy = oy + fr * CELL_SIZE + CELL_SIZE // 2
            cv2.circle(canvas, (fx, fy), CELL_SIZE // 2 + 4, col, 3)

        # HUD
        hud_y = oy + GRID_ROWS * CELL_SIZE + 8
        cv2.putText(canvas, f"Score: {self.score}", (ox, hud_y + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (80, 220, 80), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Lv {self.level}", (ox + 170, hud_y + 22),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, COL_DIM, 1, cv2.LINE_AA)
        for i in range(LIVES_INIT):
            col = (60, 60, 220) if i < self.lives else (60, 60, 60)
            cv2.circle(canvas, (BOARD_W - GRID_PAD - 18 - i * 28, hud_y + 16), 9, col, -1)
        cv2.putText(canvas, f"Apples: {len(self.apples)}  Bombs: {len(self.bombs)}",
                    (ox + 230, hud_y + 22), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    COL_DIM, 1, cv2.LINE_AA)

        # Game over overlay
        if self.game_over:
            ov = canvas.copy()
            cv2.rectangle(ov, (0, 0), (BOARD_W, BOARD_H), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.65, canvas, 0.35, 0, canvas)
            cv2.putText(canvas, "GAME  OVER",
                        (BOARD_W // 2 - 130, BOARD_H // 2 - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.7, COL_GAMEOVER, 3, cv2.LINE_AA)
            cv2.putText(canvas, f"Score: {self.score}  |  Level: {self.level}",
                        (BOARD_W // 2 - 130, BOARD_H // 2 + 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, COL_TEXT, 2, cv2.LINE_AA)
            cv2.putText(canvas, "R = restart    Q = quit",
                        (BOARD_W // 2 - 110, BOARD_H // 2 + 70),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, COL_DIM, 1, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════════════
#  Webcam panel
# ════════════════════════════════════════════════════════════════════════
def render_cam_panel(cam_frame: np.ndarray | None,
                     gesture_ctrl: GestureController | None) -> np.ndarray:
    panel = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

    if cam_frame is not None:
        h, w  = cam_frame.shape[:2]
        scale = min(CAM_W / w, (CAM_H - 100) / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(cam_frame, (nw, nh))
        x_off = (CAM_W - nw) // 2
        panel[:nh, x_off:x_off + nw] = resized

    bar_top = CAM_H - 100
    cv2.rectangle(panel, (0, bar_top), (CAM_W, CAM_H), (15, 15, 25), -1)
    cv2.line(panel, (0, bar_top), (CAM_W, bar_top), (70, 70, 90), 1)

    if gesture_ctrl is not None:
        now  = time.time()
        show = gesture_ctrl.display_gesture if now < gesture_ctrl.display_until else "idle"
        text = GESTURE_DISPLAY.get(show, show)
        col  = GESTURE_COLS.get(show, (180, 180, 180))
        cv2.putText(panel, f"Gesture: {text}", (10, bar_top + 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.72, col, 1, cv2.LINE_AA)

    cv2.putText(panel, "sl/sr = left/right   su/sd = up/down",
                (10, bar_top + 55), cv2.FONT_HERSHEY_PLAIN, 1.0,
                (100, 180, 100), 1, cv2.LINE_AA)
    cv2.putText(panel, "WASD / Arrows = move   R = restart",
                (10, bar_top + 75), cv2.FONT_HERSHEY_PLAIN, 0.95,
                (80, 80, 100), 1, cv2.LINE_AA)
    cv2.putText(panel, "Q / ESC = quit",
                (10, bar_top + 93), cv2.FONT_HERSHEY_PLAIN, 0.9,
                (70, 70, 70), 1, cv2.LINE_AA)

    return panel


# ════════════════════════════════════════════════════════════════════════
#  Main loop
# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Grid Collector — Gesture Controlled (O10)")
    parser.add_argument("--camera",   type=int, default=0, help="Webcam index")
    parser.add_argument("--flip",     action="store_true", help="Mirror webcam")
    parser.add_argument("--keyboard", action="store_true",
                        help="Keyboard-only mode (no gesture model needed)")
    args = parser.parse_args()

    # ── Gesture controller ────────────────────────────────────────────
    gesture_ctrl: GestureController | None = None
    if not args.keyboard:
        try:
            gesture_ctrl = GestureController()
        except Exception as e:
            print(f"[Warning] Could not load game model: {e}")
            print("  Falling back to keyboard-only mode.")
            print("  (Train the game model first: run gesture_recognition_game.ipynb)")

    # ── Camera ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}. Keyboard-only mode.")
        cap = None

    mp_pose   = mp.solutions.pose
    mp_draw   = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) if cap else None

    # ── Game init ─────────────────────────────────────────────────────
    game         = GridCollector()
    board_canvas = np.zeros((BOARD_H, BOARD_W, 3), dtype=np.uint8)
    win_name     = "Grid Collector — Gesture Controlled"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    high_score = 0

    print("\n  Grid Collector started!")
    print("  Gestures: sl/sr = left/right  |  su/sd = up/down")
    print("  Keyboard: WASD / Arrow keys  |  R = restart  |  Q = quit\n")

    while True:
        # ── Camera + gesture ────────────────────────────────────────
        cam_display = None
        if cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if ok:
                if args.flip:
                    frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                if pose:
                    results = pose.process(rgb)
                    rgb.flags.writeable = True
                    cam_display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if results.pose_landmarks:
                        mp_draw.draw_landmarks(
                            cam_display, results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                        )
                        if gesture_ctrl and not game.game_over:
                            g = gesture_ctrl.process_landmarks(results.pose_landmarks)
                            if g:
                                if   g == "sl": game.move(0, -1)
                                elif g == "sr": game.move(0,  1)
                                elif g == "su": game.move(-1, 0)
                                elif g == "sd": game.move( 1, 0)
                else:
                    cam_display = frame

        # ── Keyboard ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("r"), ord("R")):
            high_score = max(high_score, game.score)
            game.reset()
            if gesture_ctrl:
                gesture_ctrl.reset()
            continue

        if not game.game_over:
            if key in (ord("a"), ord("A"), 81):    game.move(0, -1)
            elif key in (ord("d"), ord("D"), 83):  game.move(0,  1)
            elif key in (ord("w"), ord("W"), 82):  game.move(-1, 0)
            elif key in (ord("s"), ord("S"), 84):  game.move( 1, 0)

        # ── Game tick ────────────────────────────────────────────────
        game.update()

        # ── Render ───────────────────────────────────────────────────
        game.render(board_canvas)
        if high_score > 0:
            cv2.putText(board_canvas, f"Best: {high_score}",
                        (BOARD_W - 130, BOARD_H - 8), cv2.FONT_HERSHEY_PLAIN, 1.1,
                        (160, 160, 0), 1, cv2.LINE_AA)

        cam_panel = render_cam_panel(cam_display, gesture_ctrl)
        sep       = np.full((BOARD_H, 2, 3), (70, 70, 90), dtype=np.uint8)
        combined  = np.hstack([board_canvas, sep, cam_panel])
        cv2.imshow(win_name, combined)

    # ── Cleanup ───────────────────────────────────────────────────────
    if pose:  pose.close()
    if cap:   cap.release()
    cv2.destroyAllWindows()
    final = max(high_score, game.score)
    print(f"Final score: {game.score}  |  High score: {final}")


if __name__ == "__main__":
    main()
