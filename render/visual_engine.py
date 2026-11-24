"""
Visual Rendering Engine for Aloha Toss & Catch.
Orchestrates multi-camera MuJoCo rendering and composite HUD overlays.
"""
import os
import cv2
import numpy as np
import mujoco
from PIL import Image

from render.hud_overlay import AlohaTelemetryHUD

class VisualEngine:
    def __init__(self, model, data, width=1280, height=720):
        self.model = model
        self.data = data
        self.width = width
        self.height = height

        # Initialize offscreen renderers for main and wrist cameras
        self.main_renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.pip_renderer = mujoco.Renderer(self.model, height=270, width=432)
        self.hud = AlohaTelemetryHUD(width=self.width, height=self.height)

    def render_frame(self, telemetry, camera_name="front_overview"):
        # 1. Render Main Studio Scene (RGB)
        self.main_renderer.update_scene(self.data, camera=camera_name)
        main_rgb = self.main_renderer.render()
        main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)

        # 2. Render Left Wrist Camera
        try:
            self.pip_renderer.update_scene(self.data, camera="left_wrist_cam")
            left_rgb = self.pip_renderer.render()
            telemetry["left_pip"] = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            telemetry["left_pip"] = None

        # 3. Render Right Wrist Camera
        try:
            self.pip_renderer.update_scene(self.data, camera="right_wrist_cam")
            right_rgb = self.pip_renderer.render()
            telemetry["right_pip"] = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            telemetry["right_pip"] = None

        # 4. Composite HUD Overlay
        final_bgr = self.hud.draw(main_bgr, telemetry)
        return final_bgr

    def save_gif(self, frames_bgr, output_path="assets/demo_toss_catch.gif", fps=30):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pil_images = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))

        if len(pil_images) > 0:
            pil_images[0].save(
                output_path,
                save_all=True,
                append_images=pil_images[1:],
                optimize=True,
                duration=int(1000 / fps),
                loop=0
            )
            print(f"[VisualEngine] Saved 60fps HUD GIF to: {output_path}")

    def close(self):
        self.main_renderer.close()
        self.pip_renderer.close()

