import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk

from model import ChangeDetectionModel


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "best_model.pth"
IMAGE_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# LOAD MODEL
# ============================================================

model = ChangeDetectionModel().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Support different checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully!")
print("Using device:", device)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

image1_path = None
image2_path = None

image1_display = None
image2_display = None
mask_display = None


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(path):

    image = cv2.imread(path)

    if image is None:
        raise ValueError("Could not read image.")

    # BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Same resizing as training
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Same normalization as training
    image = image.astype("float32") / 255.0

    # HWC → CHW
    image = torch.tensor(
        image,
        dtype=torch.float32
    ).permute(2, 0, 1)

    # Add batch dimension
    image = image.unsqueeze(0)

    return image.to(device)


# ============================================================
# SELECT IMAGE 1
# ============================================================

def upload_image1():

    global image1_path
    global image1_display

    path = filedialog.askopenfilename(
        title="Select Time 1 Image",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg"),
            ("All Files", "*.*")
        ]
    )

    if not path:
        return

    image1_path = path

    image = Image.open(path)
    image.thumbnail((350, 350))

    image1_display = ImageTk.PhotoImage(image)

    label_image1.config(
        image=image1_display
    )

    status_label.config(
        text="Time 1 image selected."
    )


# ============================================================
# SELECT IMAGE 2
# ============================================================

def upload_image2():

    global image2_path
    global image2_display

    path = filedialog.askopenfilename(
        title="Select Time 2 Image",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg"),
            ("All Files", "*.*")
        ]
    )

    if not path:
        return

    image2_path = path

    image = Image.open(path)
    image.thumbnail((350, 350))

    image2_display = ImageTk.PhotoImage(image)

    label_image2.config(
        image=image2_display
    )

    status_label.config(
        text="Time 2 image selected."
    )


# ============================================================
# RUN PREDICTION
# ============================================================

def predict_change():

    global mask_display

    if image1_path is None:
        messagebox.showwarning(
            "Missing Image",
            "Please select Time 1 image."
        )
        return

    if image2_path is None:
        messagebox.showwarning(
            "Missing Image",
            "Please select Time 2 image."
        )
        return

    try:

        img1 = preprocess_image(image1_path)
        img2 = preprocess_image(image2_path)

        with torch.no_grad():

            output = model(
                img1,
                img2
            )

            # Convert logits → probability
            probability = torch.sigmoid(output)

            # Binary mask
            prediction = (
                probability > 0.5
            ).float()

        # Remove batch/channel dimensions
        mask = prediction[0, 0].cpu().numpy()

        # Convert to 0-255
        mask = (
            mask * 255
        ).astype(np.uint8)

        # PIL image
        mask_image = Image.fromarray(
            mask
        )

        mask_image = mask_image.resize(
            (350, 350)
        )

        mask_display = ImageTk.PhotoImage(
            mask_image
        )

        label_mask.config(
            image=mask_display
        )

        status_label.config(
            text="Prediction completed!"
        )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )


# ============================================================
# SAVE MASK
# ============================================================

def save_mask():

    if mask_display is None:
        messagebox.showwarning(
            "No Prediction",
            "Run prediction first."
        )
        return

    path = filedialog.asksaveasfilename(
        title="Save Change Mask",
        defaultextension=".png",
        filetypes=[
            ("PNG Image", "*.png")
        ]
    )

    if not path:
        return

    # Re-run prediction to get original 256x256 mask
    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)

    with torch.no_grad():

        output = model(
            img1,
            img2
        )

        probability = torch.sigmoid(output)

        prediction = (
            probability > 0.5
        ).float()

    mask = prediction[0, 0].cpu().numpy()

    mask = (
        mask * 255
    ).astype(np.uint8)

    cv2.imwrite(
        path,
        mask
    )

    messagebox.showinfo(
        "Saved",
        "Change mask saved successfully!"
    )


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "SIH1518 - Change Detection"
)

root.geometry(
    "1200x700"
)

root.configure(
    bg="white"
)


# ============================================================
# TITLE
# ============================================================

title = tk.Label(
    root,
    text="Satellite Image Change Detection",
    font=("Arial", 22, "bold"),
    bg="white"
)

title.pack(
    pady=15
)


# ============================================================
# IMAGE FRAME
# ============================================================

main_frame = tk.Frame(
    root,
    bg="white"
)

main_frame.pack(
    pady=10
)


# ============================================================
# TIME 1
# ============================================================

frame1 = tk.Frame(
    main_frame,
    bg="white",
    width=350,
    height=450
)

frame1.grid(
    row=0,
    column=0,
    padx=15
)

label1_title = tk.Label(
    frame1,
    text="Time 1 (Before)",
    font=("Arial", 16, "bold"),
    bg="white"
)

label1_title.pack(
    pady=10
)

label_image1 = tk.Label(
    frame1,
    text="No image selected",
    width=35,
    height=18,
    bg="black",
    fg="white"
)

label_image1.pack()


button1 = tk.Button(
    frame1,
    text="Upload Image 1",
    command=upload_image1,
    font=("Arial", 12),
    padx=15,
    pady=5
)

button1.pack(
    pady=15
)


# ============================================================
# TIME 2
# ============================================================

frame2 = tk.Frame(
    main_frame,
    bg="white",
    width=350,
    height=450
)

frame2.grid(
    row=0,
    column=1,
    padx=15
)

label2_title = tk.Label(
    frame2,
    text="Time 2 (After)",
    font=("Arial", 16, "bold"),
    bg="white"
)

label2_title.pack(
    pady=10
)

label_image2 = tk.Label(
    frame2,
    text="No image selected",
    width=35,
    height=18,
    bg="black",
    fg="white"
)

label_image2.pack()


button2 = tk.Button(
    frame2,
    text="Upload Image 2",
    command=upload_image2,
    font=("Arial", 12),
    padx=15,
    pady=5
)

button2.pack(
    pady=15
)


# ============================================================
# MASK
# ============================================================

frame3 = tk.Frame(
    main_frame,
    bg="white",
    width=350,
    height=450
)

frame3.grid(
    row=0,
    column=2,
    padx=15
)

label3_title = tk.Label(
    frame3,
    text="Change Mask (Prediction)",
    font=("Arial", 16, "bold"),
    bg="white"
)

label3_title.pack(
    pady=10
)

label_mask = tk.Label(
    frame3,
    text="Prediction will appear here",
    width=35,
    height=18,
    bg="black",
    fg="white"
)

label_mask.pack()


button_predict = tk.Button(
    frame3,
    text="Predict Change",
    command=predict_change,
    font=("Arial", 12, "bold"),
    padx=15,
    pady=5
)

button_predict.pack(
    pady=8
)

button_save = tk.Button(
    frame3,
    text="Save Result Mask",
    command=save_mask,
    font=("Arial", 12),
    padx=15,
    pady=5
)

button_save.pack()


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(
    root,
    text="Select two satellite images to begin.",
    font=("Arial", 11),
    bg="white"
)

status_label.pack(
    pady=10
)


# ============================================================
# START GUI
# ============================================================

root.mainloop()