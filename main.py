import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np
import csv
import threading
from models import BasicCNNWithDropout, BasicCNN, ResNet50_imagenet, ResNet50, ResNet50WithDropout

# Define the class names
class_names = []
with open('./resource/names.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['name'])

modelNames = []
with open('./resource/modelNames.csv', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        modelNames.append(row['name'])

# variable
file_path = None
origin_image = None
# Load the saved model
model = tf.keras.models.load_model('./pretrain_models/my_trained_model5.h5')
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
    global origin_image
    global class_names
    if origin_image == None:
        # Show an error message if there was a problem loading the image
        messagebox.showerror('Error', 'Please upload image!')
        return
    # Load the image
    image = origin_image.copy()
    # Preprocess the image
    image_array = preprocess_image(image)
    # Use the model to make predictions
    predictions = model.predict(image_array)
    model.evaluate(image_array)
    canvasInfor.delete('all')
    text = canvasInfor.create_text(90, 50, anchor=tk.CENTER, text = "Predict information \n", font=("Arial", 14))
    index = 0
    for name in class_names:
        text = canvasInfor.create_text(10, 90 + index * 25, anchor=tk.W, text = "{0}: {1}% \n ".format( name, round( predictions[0][index]*100, 2)), font=("Arial", 14))
        index+=1
    # text = canvasInfor.create_text(90, 100, anchor=tk.CENTER, text = "\n Black Rot: {0}% \n ESCA: {1}% \n Healthy: {2}% \n Leaf Blight: {3}%".format(round( predictions[0][0]*100, 2),round( predictions[0][1]*100, 2),round( predictions[0][2]*100, 2),round( predictions[0][3]*100, 2)), font=("Arial", 14))
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

#Create a function to load model with selection
def reloadModel():
    global model
    # get the new selected option
    new_option = selected_option.get()
    # do something with the new selected option
    print("Selected option:", new_option)
    if new_option == options[0]:
        model = None
        model = tf.keras.models.load_model('./pretrain_models/my_trained_model5.h5')
    elif new_option == options[1]:
        model = None
        model = tf.keras.models.load_model('./pretrain_models/my_trained_model3.h5')
    elif new_option == options[2]:
        model = None
        model = tf.keras.models.load_model('./pretrain_models/my_trained_model.h5')
    elif new_option == options[4]:
        model = None
        model = tf.keras.models.load_model('./pretrain_models/grape_disease_model.h5')
    else:
        model = None
        model = tf.keras.models.load_model('./pretrain_models/my_trained_model2.h5')
    messagebox.showinfo("Changed model", "Selected model is {0}.".format(new_option))
        
def open_train_window(window):
    global screen_height, screen_width
    window_size = 500
    X = (screen_width - window_size) // 2
    Y = (screen_height - window_size) // 2
    selection_window = tk.Toplevel(window)
    selection_window.transient(window)
    selection_window.lift()
    selection_window.grab_set()
    selection_window.title('Train model')
    selection_window.geometry(f"{window_size}x{window_size}+{X}+{Y}")

    train_folder_path = None
    test_folder_path = None
    training_in_progress = False
    
    def select_train_folder():
        folder_path = filedialog.askdirectory()
        canvasTrainInfo.delete('all')
        if folder_path:
            nonlocal train_folder_path
            train_folder_path = folder_path
            textTrainInfo = canvasTrainInfo.create_text(10, 10, anchor=tk.NW, width=400, text = f"Train folder: {folder_path}", font=("Arial", 14))
        else:
            textTrainInfo = canvasTrainInfo.create_text(10, 10, anchor=tk.NW, text = "Train folder: No folder selected", font=("Arial", 14))
    def select_test_folder():
        folder_path = filedialog.askdirectory()
        canvasTestInfo.delete('all')
        if folder_path:
            nonlocal test_folder_path
            test_folder_path = folder_path
            textTestInfo = canvasTestInfo.create_text(10, 10, anchor=tk.NW, width=400, text = f"Test folder: {folder_path}", font=("Arial", 14))
        else:
            textTestInfo = canvasTestInfo.create_text(10, 10, anchor=tk.NW, text = "Test folder: No folder selected", font=("Arial", 14))

    def train():
        try:
            nonlocal training_in_progress
            if not training_in_progress:
                nonlocal train_folder_path, test_folder_path
                if train_folder_path and test_folder_path:
                    window.grab_set()  # Lock all windows
                    labelProgress = tk.Label(selection_window, wraplength=180, text="Training in progress...", font=("Arial", 14))
                    labelProgress.place(x=270, y=400)
                    selection_window.update()  # Update the selection_window to show the label
                    result = None
                    option = selected_optionTrain.get()
                    epoch = int(combobox.get())
                    print(epoch)
                    if option == options[0]:
                        print(options[0])
                        result = ResNet50WithDropout.startTraining(train_folder_path, test_folder_path, epoch, "./pretrain_models/my_trained_model5.h5")
                    elif option == options[1]:
                        print(options[1])
                        result = ResNet50.startTraining(train_folder_path, test_folder_path, epoch, "./pretrain_models/my_trained_model3.h5")
                    elif option == options[2]:
                        print(options[2])
                        result = BasicCNN.train(train_folder_path, test_folder_path, epoch, "./pretrain_models/my_trained_model.h5")
                    elif option == options[3]:
                        print(options[3])
                        result = BasicCNNWithDropout.train(train_folder_path, test_folder_path, epoch, "./pretrain_models/my_trained_model2.h5")
                    else:
                        print(options[4])
                        result = ResNet50_imagenet.startTraining(train_folder_path, test_folder_path, epoch, "./pretrain_models/grape_disease_model.h5")
                    result[0] = round(result[0], 2)
                    result[1] = round(result[1], 2)
                    labelProgress.config(text=f"Training Complete: {result}")
                    labelProgress.place(x=250, y=430)
                    selection_window.update()
                    selection_window.deiconify()  # Display the root window
                    selection_window.focus_set()  # Set focus to the root window
                    selection_window.grab_release()  # Release the lock on windows
                else:
                    messagebox.showerror('Error', 'Please select both train and test folders.')
        except:
            messagebox.showerror('Error', 'Training failure!!!')

        
    canvasTrainInfo = tk.Canvas(selection_window, width=500, height=100, bd=2, relief='solid')    
    textTrainInfo = canvasTrainInfo.create_text(10, 10, anchor=tk.NW, text = "Train folder: No folder selected", font=("Arial", 14))
    canvasTrainInfo.place(x=0, y=0)
    
    select_train_folder_button = tk.Button(selection_window, text="Select train folder", command=select_train_folder)
    select_train_folder_button.place(x=50, y = 250)
    
    canvasTestInfo = tk.Canvas(selection_window, width=500, height=100, bd=2, relief='solid')    
    textTestInfo = canvasTestInfo.create_text(10, 10, anchor=tk.NW, text = "Test folder: No folder selected", font=("Arial", 14))
    canvasTestInfo.place(x=0, y=110)
    
    select_test_folder_button = tk.Button(selection_window, text="Select test folder", command=select_test_folder)
    select_test_folder_button.place(x=300, y = 250)
    
    # create a label
    label = tk.Label(selection_window, text="Select an model:")
    label.place(x=50, y = 350)
    # create a variable to hold the selected option
    selected_optionTrain = tk.StringVar(selection_window)
    # create a selection box (option menu)
    options = []
    options = modelNames.copy()
    selected_optionTrain.set(options[0])
    option_menu = tk.OptionMenu(selection_window, selected_optionTrain, *options)
    option_menu.config(width=23)
    option_menu.place(x=50, y = 350 + 30)
    
    def on_combobox_select():
        print(f"Selected value: {combobox.get()}")
        
    labelEpochs = tk.Label(selection_window, text="Number of epochs:")
    labelEpochs.place(x=50, y = 350 + 70)
    combobox = ttk.Combobox(selection_window, values=list(range(1, 51)))
    combobox.current(0)  # Set the default selection to the first value
    combobox.bind("<<ComboboxSelected>>", lambda event: on_combobox_select())
    combobox.place(x=50, y = 350 + 100)
    
    train_button = tk.Button(selection_window, text="Start train", command=train)
    train_button.place(x=300, y = 350)

# Create the Tkinter window
window = tk.Tk()
window.title('Grape Disease Classifier')
window_width = 900
window_height = 650
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Create a canvas to display banner
canvasBanner = tk.Canvas(window, width=900, height=180)
canvasBanner.place(x=0, y = 0)
canvas_imageBanner = canvasBanner.create_image(0, 0, anchor=tk.NW)
imageBanner = Image.open("./resource/banner.png")
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
confirm_button = tk.Button(window, text='Confirm', command=reloadModel, width=12)
confirm_button.place(x=760, y = 350 + 150)

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
    # do something with the new selected option
    print("New selected option:", selected_option.get())

# attach a trace to the selected_option variable
selected_option.trace("w", on_option_changed)

trainmodel_button = tk.Button(window, text='Train', command=lambda: open_train_window(window), width=12)
trainmodel_button.place(x=760, y = 350 + 200)
# Start the GUI main loop
window.mainloop()
