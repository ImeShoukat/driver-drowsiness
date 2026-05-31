# Driver Drowsiness Detection System
A real-time driver drowsiness  detection app using digital image processing pipeline and CNN.

## Image Processing Pipeline
Before eye regions are fed into the CNN architecture, they undergo several image processing steps to ensure robustness under various lighting conditions
1. The image is converted into a 1-channel grayscale matrix to reduce computational overhead.
2. The contrast is enhanced using CLAHE to sharpen the visibility of eyelid and eye features.
3. The matrix size is resized to a uniform 64x64 pixels using geometric transformation.
4. The pixel values are normalized from the 0-255 range down to a scale of 0.0 to 1.0.

## Project Directory Structure
```text
driver-drowsiness/
│
├── assets/
│   └── alarm.mp3               # Audio warning 
│
├── data/                       # Eye Image Datasets
│   ├── train/                  
│   ├── val/                    
│   └── test/                   
│
├── train_model.ipynb           # DIP Preprocessing & CNN Training
├── app.py                      # Main app
├── drowsiness_model.keras      # Saved weight matrix of the trained AI model
└── README.md                   # Project Documentation
```

## Installation
### 1. Prerequisites & Library Installation
Make sure you are running Python 3.10+. Open your terminal in VS Code, install the required dependencies
```bash
pip install opencv-python tensorflow numpy pygame matplotlib
```
### 2. Model Training
If you want to re-train the CNN architecture or visualize the Before vs After performance graphs of the digital image processing pipeline, execute all cells within:
```
train_model.ipynb
```
### 3. Launch the live webcam detection
To run the live drowsiness monitoring system via your laptop's webcam, execute the following command in your terminal
```bash
python app.py
```
- Position yourself facing the webcam until your face bounding box (Blue) and eye bounding boxes (Green) appear.
- The screen dynamically displays your real-time Awake % or Sleepy % status.
- If your eyes remain closed continuously for more than 2.0 seconds, a warning siren will automatically trigger.
- Press the `q` key on your keyboard to close the live feed window and exit the application safely.

## Dataset
Dataset source: https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset

## Academic Notice & License
This project was developed strictly for academic purposes as part of the Final Project for the Digital Image Processing (Pengolahan Citra Digital) course at Universitas Jenderal Soedirman. 

Anyone is free to use, study, and modify this codebase for educational and research purposes, provided that proper academic credit/citation is given to the original authors. Commercial use of this software is highly discouraged.

Licensed under the [MIT License](LICENSE)