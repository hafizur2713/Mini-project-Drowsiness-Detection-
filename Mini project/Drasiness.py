import math
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2 as cv
import mediapipe as mp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from pygame import mixer


BASE_DIR = Path(__file__).resolve().parent
ALERT_SOUND = BASE_DIR / "audio" / "alert.wav"

EAR_THRESHOLD = 0.21
ALERT_FRAME_THRESHOLD = 30
GRAPH_WINDOW = 120
AUTO_LOG_INTERVAL_SECONDS = 0.5

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LABELS = ("Awake", "Drowsy")


def euclidean_distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


def compute_EAR(landmarks, left_indices, right_indices):
    left_eye = [landmarks[i] for i in left_indices]
    right_eye = [landmarks[i] for i in right_indices]

    left_EAR = (
        euclidean_distance(left_eye[1], left_eye[5])
        + euclidean_distance(left_eye[2], left_eye[4])
    ) / (2.0 * euclidean_distance(left_eye[0], left_eye[3]))

    right_EAR = (
        euclidean_distance(right_eye[1], right_eye[5])
        + euclidean_distance(right_eye[2], right_eye[4])
    ) / (2.0 * euclidean_distance(right_eye[0], right_eye[3]))

    return (left_EAR + right_EAR) / 2.0


class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detection Dashboard")
        self.root.geometry("1180x760")
        self.root.minsize(1050, 680)
        self.root.configure(bg="#101418")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_capture = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True
        )

        self.sound_ready = self.setup_sound()
        self.alert_active = False
        self.detection_running = False
        self.eye_closed_frame_count = 0
        self.running = True

        self.ear_values = deque(maxlen=GRAPH_WINDOW)
        self.status_values = deque(maxlen=GRAPH_WINDOW)
        self.confusion = Counter()
        self.actual_label = tk.StringVar(value="Awake")
        self.auto_evaluate = tk.BooleanVar(value=True)
        self.predicted_label = tk.StringVar(value="No face")
        self.ear_text = tk.StringVar(value="EAR: --")
        self.alert_text = tk.StringVar(value="Status: Stopped")
        self.accuracy_text = tk.StringVar(value="Accuracy: --")
        self.samples_text = tk.StringVar(value="Samples: 0")
        self.cm_awake_awake = tk.StringVar(value="0")
        self.cm_awake_drowsy = tk.StringVar(value="0")
        self.cm_drowsy_awake = tk.StringVar(value="0")
        self.cm_drowsy_drowsy = tk.StringVar(value="0")
        self.last_auto_log_time = 0

        self.build_ui()
        self.update_frame()

    def setup_sound(self):
        if not ALERT_SOUND.exists():
            return False

        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            mixer.init()
            mixer.music.load(str(ALERT_SOUND))
            return True
        except Exception:
            return False

    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#101418")
        style.configure("Panel.TFrame", background="#171d24")
        style.configure("Panel.TLabelframe", background="#171d24", foreground="#eef3f8")
        style.configure(
            "Panel.TLabelframe.Label", background="#171d24", foreground="#eef3f8"
        )
        style.configure("TLabel", background="#101418", foreground="#eef3f8")
        style.configure("Panel.TLabel", background="#171d24", foreground="#eef3f8")
        style.configure("Muted.TLabel", background="#171d24", foreground="#9aa8b6")
        style.configure(
            "Matrix.TLabel",
            background="#202832",
            foreground="#eef3f8",
            padding=5,
            anchor="center",
            font=("Segoe UI", 9, "bold"),
        )
        style.configure("Accent.TButton", padding=8)
        style.configure("TRadiobutton", background="#171d24", foreground="#eef3f8")

        container = ttk.Frame(self.root, padding=16)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(1, weight=1)

        title = ttk.Label(
            container,
            text="Drowsiness Detection Dashboard",
            font=("Segoe UI", 22, "bold"),
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        video_panel = ttk.Frame(container, style="Panel.TFrame", padding=12)
        video_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        video_panel.rowconfigure(0, weight=1)
        video_panel.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_panel, style="Panel.TLabel")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        side_panel = ttk.Frame(container, style="Panel.TFrame", padding=14)
        side_panel.grid(row=1, column=1, sticky="nsew")
        side_panel.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(
            side_panel,
            textvariable=self.alert_text,
            style="Panel.TLabel",
            font=("Segoe UI", 17, "bold"),
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        stats = ttk.Frame(side_panel, style="Panel.TFrame")
        stats.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        stats.columnconfigure((0, 1), weight=1)

        self.make_stat(stats, "Current", self.ear_text, 0, 0)
        self.make_stat(stats, "Prediction", self.predicted_label, 0, 1)
        self.make_stat(stats, "Evaluation", self.accuracy_text, 1, 0)
        self.make_stat(stats, "Logged", self.samples_text, 1, 1)

        matrix_panel = ttk.LabelFrame(
            side_panel,
            text="Confusion Matrix",
            padding=6,
            style="Panel.TLabelframe",
        )
        matrix_panel.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        for column in range(3):
            matrix_panel.columnconfigure(column, weight=1, uniform="matrix")

        ttk.Label(matrix_panel, text="", style="Panel.TLabel").grid(row=0, column=0)
        ttk.Label(matrix_panel, text="Pred Awake", style="Muted.TLabel").grid(
            row=0, column=1, sticky="ew", padx=2, pady=1
        )
        ttk.Label(matrix_panel, text="Pred Drowsy", style="Muted.TLabel").grid(
            row=0, column=2, sticky="ew", padx=2, pady=1
        )
        ttk.Label(matrix_panel, text="Actual Awake", style="Muted.TLabel").grid(
            row=1, column=0, sticky="ew", padx=2, pady=1
        )
        ttk.Label(
            matrix_panel, textvariable=self.cm_awake_awake, style="Matrix.TLabel"
        ).grid(row=1, column=1, sticky="ew", padx=2, pady=1)
        ttk.Label(
            matrix_panel, textvariable=self.cm_awake_drowsy, style="Matrix.TLabel"
        ).grid(row=1, column=2, sticky="ew", padx=2, pady=1)
        ttk.Label(matrix_panel, text="Actual Drowsy", style="Muted.TLabel").grid(
            row=2, column=0, sticky="ew", padx=2, pady=1
        )
        ttk.Label(
            matrix_panel, textvariable=self.cm_drowsy_awake, style="Matrix.TLabel"
        ).grid(row=2, column=1, sticky="ew", padx=2, pady=1)
        ttk.Label(
            matrix_panel, textvariable=self.cm_drowsy_drowsy, style="Matrix.TLabel"
        ).grid(row=2, column=2, sticky="ew", padx=2, pady=1)

        label_panel = ttk.LabelFrame(
            side_panel,
            text="Ground Truth Label",
            padding=8,
            style="Panel.TLabelframe",
        )
        label_panel.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        label_panel.columnconfigure((0, 1, 2), weight=1)
        ttk.Radiobutton(
            label_panel,
            text="Actual Awake",
            variable=self.actual_label,
            value="Awake",
        ).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Radiobutton(
            label_panel,
            text="Actual Drowsy",
            variable=self.actual_label,
            value="Drowsy",
        ).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(
            label_panel,
            text="Auto update metrics",
            variable=self.auto_evaluate,
        ).grid(row=0, column=2, sticky="w", padx=4, pady=2)

        detection_controls = ttk.Frame(side_panel, style="Panel.TFrame")
        detection_controls.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        detection_controls.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(
            detection_controls,
            text="Start Detection",
            command=self.start_detection,
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.stop_button = ttk.Button(
            detection_controls,
            text="Stop Detection",
            command=self.stop_detection,
            state=tk.DISABLED,
        )
        self.stop_button.grid(row=0, column=1, sticky="ew")

        graph_panel = ttk.Frame(side_panel, style="Panel.TFrame")
        graph_panel.grid(row=5, column=0, sticky="nsew")
        side_panel.rowconfigure(5, weight=1)
        graph_panel.rowconfigure(0, weight=1)
        graph_panel.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(5, 2.0), dpi=100, facecolor="#171d24")
        self.ear_ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_panel)
        graph_widget = self.canvas.get_tk_widget()
        graph_widget.configure(height=130)
        graph_widget.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(container, style="TFrame")
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(footer, text="Reset Metrics", command=self.reset_metrics).pack(
            side=tk.LEFT
        )
        ttk.Button(footer, text="Quit", command=self.on_close).pack(
            side=tk.RIGHT
        )

    def make_stat(self, parent, title, variable, row, column):
        card = ttk.Frame(parent, style="Panel.TFrame", padding=(0, 8))
        card.grid(row=row, column=column, sticky="ew", padx=(0, 12), pady=4)
        ttk.Label(card, text=title, style="Muted.TLabel", font=("Segoe UI", 9)).pack(
            anchor="w"
        )
        ttk.Label(
            card,
            textvariable=variable,
            style="Panel.TLabel",
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w")

    def update_frame(self):
        if not self.running:
            return

        if not self.detection_running:
            self.update_stats(None, "Stopped")
            self.update_placeholder_video()
            self.update_graphs()
            self.root.after(100, self.update_frame)
            return

        if self.video_capture is None:
            self.root.after(100, self.update_frame)
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.root.after(20, self.update_frame)
            return

        frame = cv.flip(frame, 1)
        frame, ear, prediction = self.process_frame(frame)
        self.update_stats(ear, prediction)
        self.auto_log_prediction(prediction)
        self.update_video(frame)
        self.update_graphs()
        self.root.after(15, self.update_frame)

    def draw_stopped_overlay(self, frame):
        cv.putText(
            frame,
            "Detection Stopped",
            (24, 42),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 190, 255),
            2,
        )

    def process_frame(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        prediction = "No face"
        ear = None

        if results.multi_face_landmarks:
            mesh_points = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            landmarks = [(int(p.x * w), int(p.y * h)) for p in mesh_points]
            ear = compute_EAR(landmarks, LEFT_EYE_IDX, RIGHT_EYE_IDX)

            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                cv.circle(frame, landmarks[idx], 2, (24, 210, 125), -1)

            if ear < EAR_THRESHOLD:
                self.eye_closed_frame_count += 1
            else:
                self.eye_closed_frame_count = 0

            prediction = (
                "Drowsy"
                if self.eye_closed_frame_count > ALERT_FRAME_THRESHOLD
                else "Awake"
            )
        else:
            self.eye_closed_frame_count = 0
            self.stop_alert()

        if prediction == "Drowsy":
            self.start_alert()
            color = (0, 0, 255)
        elif prediction == "Awake":
            self.stop_alert()
            color = (24, 210, 125)
        else:
            color = (0, 190, 255)

        cv.putText(
            frame,
            f"Prediction: {prediction}",
            (24, 42),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
        if ear is not None:
            cv.putText(
                frame,
                f"EAR: {ear:.3f}",
                (24, 78),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        return frame, ear, prediction

    def update_stats(self, ear, prediction):
        self.predicted_label.set(prediction)
        self.ear_text.set(f"EAR: {ear:.3f}" if ear is not None else "EAR: --")
        if not self.detection_running:
            self.alert_text.set("Status: Stopped")
        else:
            self.alert_text.set(
                "Status: Drowsiness Alert"
                if prediction == "Drowsy"
                else f"Status: {prediction}"
            )

        if ear is not None:
            self.ear_values.append(ear)
            self.status_values.append(1 if prediction == "Drowsy" else 0)

    def update_video(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((720, 540))
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def update_placeholder_video(self):
        image = Image.new("RGB", (720, 540), "#111820")
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def update_graphs(self):
        self.ear_ax.clear()

        self.ear_ax.set_facecolor("#171d24")
        self.ear_ax.plot(list(self.ear_values), color="#38bdf8", linewidth=2)
        self.ear_ax.axhline(EAR_THRESHOLD, color="#f97316", linestyle="--", linewidth=1)
        self.ear_ax.set_title("EAR Trend", color="#eef3f8", fontsize=11)
        self.ear_ax.set_ylim(0, 0.45)
        self.ear_ax.tick_params(colors="#9aa8b6", labelsize=8)
        for spine in self.ear_ax.spines.values():
            spine.set_color("#303946")

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

    def get_matrix(self):
        return [
            [self.confusion[("Awake", "Awake")], self.confusion[("Awake", "Drowsy")]],
            [self.confusion[("Drowsy", "Awake")], self.confusion[("Drowsy", "Drowsy")]],
        ]

    def auto_log_prediction(self, prediction):
        if not self.auto_evaluate.get() or prediction not in LABELS:
            return

        now = time.time()
        if now - self.last_auto_log_time < AUTO_LOG_INTERVAL_SECONDS:
            return

        self.last_auto_log_time = now
        self.record_prediction(self.actual_label.get(), prediction)

    def record_prediction(self, actual, prediction):
        self.confusion[(actual, prediction)] += 1
        self.update_confusion_text()
        total = sum(self.confusion.values())
        correct = self.confusion[("Awake", "Awake")] + self.confusion[
            ("Drowsy", "Drowsy")
        ]
        accuracy = (correct / total) * 100 if total else 0
        self.accuracy_text.set(f"Accuracy: {accuracy:.1f}%")
        self.samples_text.set(f"Samples: {total}")

    def update_confusion_text(self):
        self.cm_awake_awake.set(str(self.confusion[("Awake", "Awake")]))
        self.cm_awake_drowsy.set(str(self.confusion[("Awake", "Drowsy")]))
        self.cm_drowsy_awake.set(str(self.confusion[("Drowsy", "Awake")]))
        self.cm_drowsy_drowsy.set(str(self.confusion[("Drowsy", "Drowsy")]))

    def reset_metrics(self):
        self.confusion.clear()
        self.update_confusion_text()
        self.last_auto_log_time = 0
        self.accuracy_text.set("Accuracy: --")
        self.samples_text.set("Samples: 0")
        self.update_graphs()

    def start_detection(self):
        if self.video_capture is None:
            self.video_capture = cv.VideoCapture(0)
            if not self.video_capture.isOpened():
                self.video_capture.release()
                self.video_capture = None
                messagebox.showerror("Camera Error", "Could not open webcam.")
                return

        self.detection_running = True
        self.eye_closed_frame_count = 0
        self.last_auto_log_time = 0
        self.alert_text.set("Status: Monitoring")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)

    def stop_detection(self):
        self.detection_running = False
        self.eye_closed_frame_count = 0
        self.predicted_label.set("Stopped")
        self.ear_text.set("EAR: --")
        self.alert_text.set("Status: Stopped")
        self.stop_alert()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)

    def start_alert(self):
        if self.alert_active:
            return

        self.alert_active = True
        if self.sound_ready:
            threading.Thread(target=self.play_alert, daemon=True).start()

    def stop_alert(self):
        self.alert_active = False
        if self.sound_ready:
            mixer.music.stop()

    def play_alert(self):
        while self.alert_active:
            if not mixer.music.get_busy():
                mixer.music.play()
            time.sleep(1)

    def on_close(self):
        self.running = False
        self.stop_alert()
        time.sleep(0.1)
        if self.video_capture is not None:
            self.video_capture.release()
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()
        cv.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.mainloop()
