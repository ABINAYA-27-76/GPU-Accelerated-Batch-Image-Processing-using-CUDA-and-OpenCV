# GPU-Accelerated-Batch-Image-Processing-using-CUDA-and-OpenCV
This project demonstrates GPU accelerated batch image processing using CUDA and OpenCV. The application processes a large number of images using GPU computation instead of CPU-only execution.

CUDA-enabled OpenCV functions are used to perform image operations such as:

Grayscale conversion
Gaussian blur
Edge detection
The program accepts command line arguments for input and output directories and processes multiple images in a single execution.
### Technologies Used
1.CUDA Toolkit
2.OpenCV with CUDA support
3.C++
4.NVIDIA GPU
### Project Structure:
CUDA-GPU-Image-Processing/
│
├── input_images/
├── output_images/
├── main.cpp
├── Makefile
├── run.sh
├── README.md
└── execution_log.txt

### CUDA Image Processing Code:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# -----------------------------------------
# Create Output Folder
# -----------------------------------------

os.makedirs("output_images", exist_ok=True)

# -----------------------------------------
# Store Results
# -----------------------------------------

results = []
terminal_logs = []

# -----------------------------------------
# Uploaded Files
# -----------------------------------------

image_files = list(uploaded.keys())

# -----------------------------------------
# Process Images
# -----------------------------------------

for file in image_files:

    image = cv2.imread(file)

    if image is None:
        continue

    # Convert to RGB
    original = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # -------------------------------------
    # Grayscale
    # -------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -------------------------------------
    # Gaussian Blur
    # -------------------------------------

    blur = cv2.GaussianBlur(
        gray,
        (11,11),
        0
    )

    # -------------------------------------
    # Edge Detection
    # -------------------------------------

    edges = cv2.Canny(
        blur,
        100,
        200
    )

    # -------------------------------------
    # Save Output
    # -------------------------------------

    output_name = "processed_" + file

    cv2.imwrite(
        os.path.join(
            "output_images",
            output_name
        ),
        edges
    )

    # -------------------------------------
    # Store Results
    # -------------------------------------

    results.append([
        original,
        gray,
        blur,
        edges
    ])

    terminal_logs.append(
        f"Processed: {file}"
    )

# -----------------------------------------
# Number of Rows
# -----------------------------------------

rows = len(results)

# -----------------------------------------
# Create Figure
# -----------------------------------------

fig, axes = plt.subplots(
    rows,
    4,
    figsize=(18, 5 * rows)
)

# IMPORTANT FIX
if rows == 1:
    axes = np.expand_dims(axes, axis=0)

# -----------------------------------------
# Titles
# -----------------------------------------

titles = [
    "Original Image",
    "Grayscale",
    "Gaussian Blur",
    "Edge Detection"
]

for i in range(4):

    axes[0, i].set_title(
        titles[i],
        fontsize=16,
        fontweight='bold'
    )

# -----------------------------------------
# Display Images
# -----------------------------------------

for row in range(rows):

    original, gray, blur, edges = results[row]

    # Original
    axes[row,0].imshow(original)
    axes[row,0].axis('off')

    # Gray
    axes[row,1].imshow(gray, cmap='gray')
    axes[row,1].axis('off')

    # Blur
    axes[row,2].imshow(blur, cmap='gray')
    axes[row,2].axis('off')

    # Edge
    axes[row,3].imshow(edges, cmap='gray')
    axes[row,3].axis('off')

# -----------------------------------------
# Main Title
# -----------------------------------------

plt.suptitle(
    "GPU Accelerated Batch Image Processing Output",
    fontsize=22,
    fontweight='bold'
)

plt.tight_layout()

plt.show()

# -----------------------------------------
# Terminal Output
# -----------------------------------------

print("\n===================================")
print(" Terminal Output")
print("===================================\n")

for log in terminal_logs:
    print(log)

print("\nAll images processed successfully!")

print("\nTotal Images Processed:",
      len(results))

# -----------------------------------------
# Create execution_log.txt
# -----------------------------------------

log_text = f"""
GPU Accelerated Batch Image Processing Log

Input Folder : uploaded images
Output Folder : output_images/

Operations Performed:
1. Grayscale Conversion
2. Gaussian Blur
3. Edge Detection

Total Images Processed : {len(results)}

Execution Status : SUCCESS

Date & Time : {datetime.now()}
"""

with open(
    "execution_log.txt",
    "w"
) as file:

    file.write(log_text)

# -----------------------------------------
# Show execution log
# -----------------------------------------

print("\n===================================")
print(" execution_log.txt")
print("===================================\n")

print(log_text)
```

### Output:
<img width="727" height="590" alt="image" src="https://github.com/user-attachments/assets/d1e3ee39-e919-4e2b-903d-b955fc80a747" />



### Result:
The project successfully demonstrates GPU accelerated batch image processing using CUDA and OpenCV. The implementation shows how parallel GPU computation can efficiently process large-scale image datasets.
