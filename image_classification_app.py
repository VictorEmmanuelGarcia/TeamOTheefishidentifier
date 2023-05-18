import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def classify_image():
    # Open a file dialog to choose the image
    file_path = filedialog.askopenfilename(initialdir='/', title='Select Image', filetypes=(('Image Files', '*.png *.jpg *.jpeg'), ('All Files', '*.*')))
    
    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Classify the image
    preds = model.predict(x)
    pred_labels = decode_predictions(preds, top=3)[0]
    
    # Display the results
    result_text.delete(1.0, tk.END)
    for label in pred_labels:
        result_text.insert(tk.END, f"{label[1]}: {label[2] * 100:.2f}%\n")
    

# Create the Tkinter window
window = tk.Tk()
window.title("Fish Classification App")

# Create the image preview area
image_label = tk.Label(window)
image_label.pack()

# Create the button to choose the image
choose_button = tk.Button(window, text="Choose Image", command=classify_image)
choose_button.pack()

# Create the text area for displaying the results
result_text = tk.Text(window)
result_text.pack()

# Start the Tkinter event loop
window.mainloop()
