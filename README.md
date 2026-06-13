# NeuralSight // ML Image Classifier Playground

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hosted_on_GitHub_Pages-cyan?style=for-the-badge&logo=github)](https://nkachalia1.github.io/ml-image-classifier/)

NeuralSight is a highly interactive, client-side Machine Learning Image Classifier and Transfer Learning Playground running entirely in the browser using **TensorFlow.js**. 

This application uses GPU-accelerated WebGL bindings to perform high-speed standard classification (via MobileNet) and on-the-fly custom training (via a custom k-Nearest Neighbors head) in real-time.

---

## 🌟 Key Features

1. **Pre-trained MobileNet Inference**
   - Classify images instantly from a live webcam feed, drag-and-drop file uploads, or a preloaded gallery.
   - Uses a COCO-SSD person assist in MobileNet mode so live camera views of people resolve to `person` instead of nearby ImageNet object labels.
   - Beautiful, reactive HUD displaying top predictions with real-time confidence scores and system latency metrics.

2. **In-Browser Custom Transfer Learning (KNN)**
   - Train your own neural categories on the fly with zero remote backend dependencies.
   - Record training examples instantly by holding the recording button against the webcam stream, or uploading files.
   - Export your custom trained model weights as a single JSON file and reload them later to resume inference.

3. **Neural "X-Ray" Preprocessor**
   - Inspect exactly what the neural network "sees" by viewing the normalized `224x224` pixel tensor input.
   - Apply real-time visual preprocessors:
     - **Grayscale**: Visualizes luminance channel extraction.
     - **Sobel Edge Detector**: Convolves 3x3 kernels across pixels to isolate sharp high-contrast contours (neon ultraviolet styling).
     - **Pixel Grid**: Maps the discrete input resolution cells.
   - Real-time RGB Color Distribution Histogram mapping.

4. **Custom Model Validation Lab**
   - Queue labeled holdout images against your trained custom classes.
   - Run an in-browser evaluation pass to measure accuracy, average confidence, and per-image outcomes.
   - Inspect a confusion matrix to spot class overlap and export validation reports as CSV.

5. **Analytics & Performance Tracker**
   - Chart history table tracking past predictions.
   - Interactive system latency monitor showcasing inference latency in milliseconds over time.
   - Export historical telemetry outputs directly to a CSV file.

---

## 🚀 How to Run Locally

Since the project operates as a static site powered by CDN dependencies, running it simply requires a lightweight web server to prevent browser `CORS` file system blockages.

### Option A: NPM / Node.js Dev Server (Recommended)
1. Ensure Node.js is installed.
2. Open your terminal in this directory.
3. Install dependencies:
   ```bash
   npm install
   ```
4. Start the Vite development server:
   ```bash
   npm run dev
   ```
5. Open `http://localhost:5173` in your browser.

### Option B: Python Fallback Web Server (Zero-Dependency)
1. Ensure Python 3 is installed.
2. Double-click or execute the server script in the terminal:
   ```bash
   python server.py
   ```
3. A local server will start on port `8000` and automatically open your default browser to `http://localhost:8000`.

---

## 🧠 Behind the Machine Learning

### 1. Pre-trained Inference (MobileNet v2)
MobileNet is a class of efficient convolutional neural networks designed for mobile and embedded vision applications. NeuralSight loads the MobileNet v2 model. When an image is supplied, it is squeezed down to a `224x224` tensor, normalized between `-1.0` and `1.0`, and convolved through layers to map against 1000 categories from the ImageNet database. Since ImageNet MobileNet does not include a generic `person` class, NeuralSight also runs a cached COCO-SSD person detector in MobileNet mode and promotes `person` when a human is detected in the frame.

### 2. Transfer Learning (KNN Classifier)
Rather than retraining millions of network weights (which requires high-end servers), we use **Feature Extraction (Transfer Learning)**:
- We pass your input feed through MobileNet, but stop at the final pre-classification activation layer.
- This returns a dense **1280-dimensional feature vector** representing highly optimized features (shapes, curves, textures) detected by the network.
- When you click "Record Example", we save this vector and automatically generate **6 augmented vectors** (incorporating horizontal mirroring, rotations, brightness shifts, scale zoom/crop, and low-light variations) to enforce robust visual invariance.
- During custom prediction, we pass the webcam frame through MobileNet to extract its vector, and run a **k-Nearest Neighbors (KNN)** classification with a dynamically adjusted value of `k` based on dataset size for optimal robustness. Custom labels require at least two trained classes and a high similarity score before they blend into standard inference, which prevents a single uploaded label from sticking to unrelated camera frames.

### 3. Edge Detection Matrix Math (Sobel Filter)
In the X-Ray tab, Sobel Edge Detection is implemented via canvas pixel math:
- Grayscale values are extracted for each pixel surrounding a coordinates block.
- Two 3x3 convolution matrices are applied over the surrounding pixels:
  
  $$\mathbf{G}_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * \mathbf{A} \quad \text{and} \quad \mathbf{G}_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * \mathbf{A}$$

- The vertical and horizontal gradient values are squared, added, and square-rooted to get the edge gradient magnitude:

  $$G = \sqrt{G_x^2 + G_y^2}$$

- The pixel is then colored proportionally, outlining sharp boundaries instantly.

---

## 🛠️ File Structure

- `index.html` - Premium UI viewport skeleton, including semantic structures and CDN modules.
- `style.css` - Custom styling tokens, glassmorphism panel properties, and glowing state transitions.
- `classifier.js` - TensorFlow.js and MobileNet loading, features bottleneck extraction and KNN weights handling.
- `ui.js` - Graphic canvas manipulations, webcam connections, drag-drop uploads, RGB charting, and telemetry logging.
- `app.js` - Central loop coordinator, managing model cycles via synchronous requestAnimationFrame.
- `server.py` - Custom Python local hosting.
- `package.json` - Node scripts mapping.
