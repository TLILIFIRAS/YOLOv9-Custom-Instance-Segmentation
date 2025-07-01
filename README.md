
# 🎥 YOLOv9 Segmentation on Local Video with Mask Overlay and Output Saving

This project performs real-time **object detection and segmentation** on a **local input video** using a custom-trained **YOLOv9 segmentation model** (`best.pt`). It overlays class labels, bounding boxes, and red segmentation masks, then saves the output as a new annotated video file.

---

## 🚀 Features

- 🧠 YOLOv9 segmentation using a custom model
- 🖼️ Bounding boxes + class label overlays in white
- 🎭 Red masks for segmented objects
- 🎥 Output saved as `output.mp4`
- ⚡ Frame skipping option for performance optimization

---

## 🛠️ Install Python Dependencies

First, it is recommended to create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔧 Setup

1. Place your custom YOLOv9 segmentation model file as `best.pt` in the project directory.
2. Create a `coco1.txt` file containing your class names, one per line.
3. Add your input video file named `input.mp4` (or modify filenames in the script).

Directory structure:

```
.
├── best.pt
├── coco1.txt
├── input.mp4
├── main.py
├── requirements.txt
└── README.md
```

---

## ▶️ Usage

Run the main script:

```bash
python app.py
```

- The script will process the input video, apply detection and segmentation, then save the output video as `output.mp4`.
- Press `q` during the video window to exit early.

---

## 📦 Output

The output video will be saved as:

```
output.mp4
```

Features of the output:

- White bounding boxes and class labels overlay
- Red segmentation masks overlay
- Video resized to 1020x600 (modifiable in script)
- Preserves the original FPS of the input video

---

## 📄 requirements.txt

Contents of `requirements.txt`:

```
ultralytics>=8.0.20
opencv-python
numpy
pandas
cvzone
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## ✨ Author

**Firas Tlili**  
[LinkedIn](https://www.linkedin.com/in/firastlili)  
Full Stack Machine Learning Engineer | Computer Vision Enthusiast

---

## 📄 License

This project is licensed under the MIT License.
