import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np
import csv

# Define the class names
class_names = []
# class_names = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight', ...]
with open('names.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['name'])

modelNames = []
with open('modelNames.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        modelNames.append(row['name'])

# variable
file_path = None
origin_image = None
# Load the saved model
model = tf.keras.models.load_model('./models/my_trained_model5.h5')
img_height = img_width = 256

# Create a function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of the model
    image = image.resize((img_height, img_width))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the pixel values to be between 0 and 1
    image_array = image_array / 255.0
    # Expand the dimensions of the image to match the input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Create a function to classify the image
def classify_image():
    global file_path
    global origin_image
    global class_names
    if file_path == None:
        # Show an error message if there was a problem loading the image
        messagebox.showerror('Error', 'Please upload image!')
    # Load the image
    image = origin_image.copy()
    # Preprocess the image
    image_array = preprocess_image(image)
    # Use the model to make predictions
    predictions = model.predict(image_array)
    model.evaluate(image_array)
    # # Get the class with the highest probability
    # predicted_class = np.argmax(predictions[0])
    # # Get the class name
    # class_name = class_names[predicted_class]
    # # Get the probability of the predicted class
    # class_prob = np.max(predictions[0])
    # Show a message box with the predicted class and percentage
    canvasInfor.delete('all')
    text = canvasInfor.create_text(90, 80, anchor=tk.CENTER, text = "Predict information \n Black Rot: {0}% \n ESCA: {1}% \n Healthy: {2}% \n Leaf Blight: {3}%".format(round( predictions[0][0]*100, 2),round( predictions[0][1]*100, 2),round( predictions[0][2]*100, 2),round( predictions[0][3]*100, 2)), font=("Arial", 14))
    # messagebox.showinfo('Classification Result', f'The image is classified as {class_name} with {class_prob} probability.')

# Create a function to handle the upload button click
def handle_upload():
    # Open a file dialog to select the image file
    global file_path
    global origin_image
    file_path = filedialog.askopenfilename()
    # Check if a file was selected
    if file_path:
        try:
            # Load the image and display it in the GUI
            origin_image = Image.open(file_path)
            image = origin_image.copy()
            image = image.resize((400, 400))
            image_tk = ImageTk.PhotoImage(image)
            canvas.itemconfig(canvas_image, image=image_tk)
            canvas.image = image_tk
        except:
            # Show an error message if there was a problem loading the image
            messagebox.showerror('Error', 'Could not open the image file.')

# Create the GUI window
window = tk.Tk()
window.title('Image Classifier')
window.geometry("900x650")

# Create a canvas to display banner
canvasBanner = tk.Canvas(window, width=900, height=180)
canvasBanner.place(x=0, y = 0)
canvas_imageBanner = canvasBanner.create_image(0, 0, anchor=tk.NW)
imageBanner = Image.open("./banner.png")
imageBanner = imageBanner.resize((900, 180))
image_tk = ImageTk.PhotoImage(imageBanner)
canvasBanner.itemconfig(canvas_imageBanner, image=image_tk)
canvasBanner.image = image_tk

# Create a canvas to display the image
canvas = tk.Canvas(window, width=400, height=400, bd=2, relief='solid')
canvas.place(x=50, y = 100 + 100)
canvas_image = canvas.create_image(0, 0, anchor=tk.NW)

# Create a canvas to display infor
canvasInfor = tk.Canvas(window, width=200, height=400, bd=1, relief='solid')
canvasInfor.place(x=500, y = 100 + 100)

# Create a button to upload the image
upload_button = tk.Button(window, text='Upload Image', command=handle_upload, width=12)
upload_button.place(x=760, y = 100 + 100)

classify_button = tk.Button(window, text='Classify', command=classify_image, width=12)
classify_button.place(x=760, y = 200 + 100)

# create a label
label = tk.Label(window, text="Select an model:")
label.place(x=760, y = 300 + 100)

# create a variable to hold the selected option
selected_option = tk.StringVar(window)

# set the default value of the variable
selected_option.set("ResNet 50 with dropout")

# create a selection box (option menu)
options = []
# options = ["ResNet 50 with dropout", "ResNet 50", "Basic CNN", "Basic CNN with dropout", "ResNet 50 - imagenet"]
options = modelNames.copy()
option_menu = tk.OptionMenu(window, selected_option, *options)
option_menu.config(width=23)
option_menu.place(x=710, y = 350 + 100)
def on_option_changed(*args):
    global model
    # get the new selected option
    new_option = selected_option.get()
    # do something with the new selected option
    print("Selected option:", new_option)
    if new_option == "ResNet 50 with dropout":
        model = None
        model = tf.keras.models.load_model('./models/my_trained_model5.h5')
    elif new_option == "ResNet 50":
        model = None
        model = tf.keras.models.load_model('./models/my_trained_model3.h5')
    elif new_option == "Basic CNN":
        model = None
        model = tf.keras.models.load_model('./models/my_trained_model.h5')
    elif new_option == "ResNet 50 - imagenet":
        model = None
        model = tf.keras.models.load_model('./models/grape_disease_model.h5')
    else:
        model = None
        model = tf.keras.models.load_model('./models/my_trained_model2.h5')

# attach a trace to the selected_option variable
selected_option.trace("w", on_option_changed)
# Start the GUI main loop
window.mainloop()
