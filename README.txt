TWO-DIGIT HANDWRITTEN NUMBER RECOGNITION SYSTEM
================================================

This project is a full pipeline for recognizing handwritten two-digit numbers (00–99) using a Convolutional Neural Network (CNN) and a real-time Tkinter-based GUI.

------------------------------------------------
SYSTEM REQUIREMENTS
------------------------------------------------
- Python 3.11 (64-bit)
- pip (Python package manager)
- At least 8 GB RAM
- Optional: NVIDIA GPU with CUDA 11.8+ (for faster training)

------------------------------------------------
FILES INCLUDED
------------------------------------------------
- run_project.py                --> One-click setup and launch script
- exporting_images_from_npz.py --> Converts .npz dataset to PNG images
- auto_sort.py                 --> Sorts images into folders by label
- train_cnn.py                 --> Trains CNN and generates evaluation plots
- two_digit_gui.py            --> GUI for drawing and live prediction
- mnist_compressed.npz        --> Compressed MNIST 100 dataset

------------------------------------------------
HOW TO INSTALL AND RUN
------------------------------------------------

1. Install Python 3.11:
   https://www.python.org/downloads/release/python-3110/

2. Open Command Prompt or Terminal in the project folder.

3. Run the setup script:
   python run_project.py

This will:
✔️ Check/install required packages for Python 3.11
✔️ Convert and organize dataset
✔️ Train the CNN model (if needed)
✔️ Launch the digit recognition GUI

------------------------------------------------
NOTES
------------------------------------------------
- All trained models and graphs will be saved locally.
- You can re-run the GUI anytime using: python two_digit_gui.py
- The model will not re-train if 'cnn_model_from_images.h5' already exists.

------------------------------------------------
CONTACT
------------------------------------------------
Author: Wassaf Ahmed Baloch
University: Bahria University Karachi Campus
Course: AI Semester Project Lab 
