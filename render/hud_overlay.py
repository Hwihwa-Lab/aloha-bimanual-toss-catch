"""
60fps Telemetry Cockpit HUD Overlay for Aloha Toss & Catch Simulator.
Integrates real-time phase banners, parabolic trajectory ribbons,
tactical PiP camera views, and dynamic flight telemetry gauges.
"""
import cv2
import numpy as np

class AlohaTelemetryHUD:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        # Color Palette (Dark Cyber / Tactical Lab)
        self.C_BG_OVERLAY = (15, 18, 22)
        self.C_CYAN = (230, 190, 0)       # BGR: Neon Cyan
        self.C_MAGENTA = (180, 40, 240)   # BGR: Neon Magenta
        self.C_GREEN = (60, 220, 40)      # BGR: Neon Green
        self.C_AMBER = (30, 160, 255)     # BGR: Bright Amber
        self.C_WHITE = (240, 245, 250)
        self.C_GRAY = (120, 130, 140)
        self.C_DARK_PANEL = (25, 28, 35)

    def draw(self, frame_bgr, telemetry):
        """
        frame_bgr: Base rendered scene (H, W, 3) in BGR
        telemetry: dict containing:
          - step: int
          - phase: int (0: GRASP, 1: TOSS, 2: FLIGHT, 3: CATCH)
          - proj_pos: np.ndarray (3,)
          - proj_vel: np.ndarray (3,)
          - dist_to_catch: float
          - left_pip: optional (H_pip, W_pip, 3) image
          - right_pip: optional (H_pip, W_pip, 3) image
          - trajectory: list of 3D positions
          - reward: float
          - is_success: bool
        """
        # Ensure correct canvas dimensions
        if frame_bgr.shape[0] != self.height or frame_bgr.shape[1] != self.width:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        canvas = frame_bgr.copy()

        # 1. Top HUD Bar
        self._draw_top_bar(canvas, telemetry)

        # 2. Phase Banner (Top-Center)
        self._draw_phase_banner(canvas, telemetry.get("phase", 0), telemetry.get("is_success", False))

        # 3. Left Telemetry Dashboard (Ballistic Dynamics)
        self._draw_ballistics_panel(canvas, telemetry)

        # 4. Right Telemetry Dashboard (Contact & Catcher Intercept)
        self._draw_intercept_panel(canvas, telemetry)

        # 5. Dual Wrist PiP Insets (Bottom Left & Bottom Right)
        left_pip = telemetry.get("left_pip")
        right_pip = telemetry.get("right_pip")
        if left_pip is not None:
            self._render_pip(canvas, left_pip, x=24, y=self.height - 180, title="L-ARM WRIST CAM (THROWER)")
        if right_pip is not None:
            self._render_pip(canvas, right_pip, x=self.width - 240, y=self.height - 180, title="R-ARM WRIST CAM (CATCHER)")

        # 6. Bottom Status Footer
        self._draw_footer(canvas, telemetry)

        return canvas

    def _draw_top_bar(self, canvas, telemetry):
        # Header banner overlay
        cv2.rectangle(canvas, (0, 0), (self.width, 42), self.C_BG_OVERLAY, -1)
        cv2.line(canvas, (0, 42), (self.width, 42), self.C_CYAN, 1)

        # Brand Title
        cv2.putText(canvas, "ALOHA 14-DOF BIMANUAL // DYNAMIC TOSS & CATCH RL", (20, 27),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, self.C_CYAN, 1, cv2.LINE_AA)

        # Step & Frequency
        step_text = f"STEP: {telemetry.get('step', 0):04d} / 250"
        cv2.putText(canvas, step_text, (self.width - 320, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.C_WHITE, 1, cv2.LINE_AA)

        fps_text = "ENGINE: MuJoCo 3.x (50Hz)"
        cv2.putText(canvas, fps_text, (self.width - 150, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_GRAY, 1, cv2.LINE_AA)

    def _draw_phase_banner(self, canvas, phase, is_success):
        phase_names = [
            ("PHASE 0: SECURE GRASP", self.C_AMBER),
            ("PHASE 1: BALLISTIC TOSS", self.C_CYAN),
            ("PHASE 2: AIR INTERCEPT", self.C_MAGENTA),
            ("PHASE 3: CUSHION LOCK", self.C_GREEN)
        ]
        text, color = phase_names[min(phase, 3)]
        if is_success:
            text = "MISSION SUCCESS: CATCH SECURED"
            color = self.C_GREEN

        cx = self.width // 2
        w_box = 320
        cv2.rectangle(canvas, (cx - w_box//2, 52), (cx + w_box//2, 86), self.C_DARK_PANEL, -1)
        cv2.rectangle(canvas, (cx - w_box//2, 52), (cx + w_box//2, 86), color, 2)
        
        # Center text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.putText(canvas, text, (cx - tw//2, 75), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)

    def _draw_ballistics_panel(self, canvas, telemetry):
        x, y, w, h = 24, 60, 240, 160
        # Semi-transparent background
        sub = canvas[y:y+h, x:x+w]
        bg = np.full(sub.shape, self.C_DARK_PANEL, dtype=np.uint8)
        cv2.addWeighted(sub, 0.35, bg, 0.65, 0, canvas[y:y+h, x:x+w])
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.C_CYAN, 1)

        cv2.putText(canvas, "BALLISTIC DYNAMICS", (x+10, y+22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, self.C_CYAN, 1, cv2.LINE_AA)

        proj_pos = telemetry.get("proj_pos", np.zeros(3))
        proj_vel = telemetry.get("proj_vel", np.zeros(3))
        speed = float(np.linalg.norm(proj_vel))

        stats = [
            f"Altitude (Z):  {proj_pos[2]:.3f} m",
            f"Ground Range:  {proj_pos[0]:.3f} m",
            f"Velocity Vxyz: {speed:.2f} m/s",
            f"Apex Delta:    {abs(proj_pos[2] - 1.25):.3f} m"
        ]
        for i, s in enumerate(stats):
            cv2.putText(canvas, s, (x+12, y+50 + i*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_WHITE, 1, cv2.LINE_AA)

    def _draw_intercept_panel(self, canvas, telemetry):
        x, y, w, h = self.width - 264, 60, 240, 160
        sub = canvas[y:y+h, x:x+w]
        bg = np.full(sub.shape, self.C_DARK_PANEL, dtype=np.uint8)
        cv2.addWeighted(sub, 0.35, bg, 0.65, 0, canvas[y:y+h, x:x+w])
        cv2.rectangle(canvas, (x, y), (x+w, y+h), self.C_MAGENTA, 1)

        cv2.putText(canvas, "INTERCEPT RADAR", (x+10, y+22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, self.C_MAGENTA, 1, cv2.LINE_AA)

        dist_catch = telemetry.get("dist_to_catch", 0.0)
        reward = telemetry.get("reward", 0.0)

        stats = [
            f"Dist to Catcher: {dist_catch*100:.1f} cm",
            f"Step Reward:     {reward:+.2f}",
            f"Contact Sensor:  {'ACTIVE' if telemetry.get('contact', False) else 'READY'}",
            f"Catch Lock:      {'LOCKED' if telemetry.get('is_success', False) else 'TRACKING'}"
        ]
        for i, s in enumerate(stats):
            color = self.C_GREEN if "ACTIVE" in s or "LOCKED" in s else self.C_WHITE
            cv2.putText(canvas, s, (x+12, y+50 + i*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _render_pip(self, canvas, pip_img, x, y, title):
        w_pip, h_pip = 216, 135
        resized = cv2.resize(pip_img, (w_pip, h_pip))
        canvas[y:y+h_pip, x:x+w_pip] = resized
        cv2.rectangle(canvas, (x, y), (x+w_pip, y+h_pip), self.C_CYAN, 1)
        
        # Subtitle badge
        cv2.rectangle(canvas, (x, y-18), (x+w_pip, y), self.C_DARK_PANEL, -1)
        cv2.putText(canvas, title, (x+6, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, self.C_CYAN, 1, cv2.LINE_AA)

    def _draw_footer(self, canvas, telemetry):
        y = self.height - 24
        cv2.rectangle(canvas, (0, y), (self.width, self.height), self.C_BG_OVERLAY, -1)
        cv2.line(canvas, (0, y), (self.width, y), (50, 60, 70), 1)
        cv2.putText(canvas, "[Hwihwa Lab] Bimanual Toss-and-Catch Autonomous Dynamic RL Benchmark",
                    (20, y+16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_GRAY, 1, cv2.LINE_AA)

