from img_class import ImageClass
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# initialize the application
ctk.set_default_color_theme(r"C:\Users\omara\OneDrive\Desktop\Assignments Course\imgprj\breeze.json")
app = ctk.CTk()
app.geometry("1920x1080")
app.title("ImgAPP")

# Global variables
global_image = None
tk_image = None
image_instance = None  # Instance of ImageClass

# Main frame for the app
main_frame = ctk.CTkFrame(master=app)
main_frame.pack(fill="both", expand=True)

# Layout setup
main_frame.grid_columnconfigure(0, weight=1)  # Image column
main_frame.grid_columnconfigure(1, weight=1)  # Function result column
main_frame.grid_rowconfigure(0, weight=1)  # Row

# Image display label
image_label = ctk.CTkLabel(master=main_frame, text="")
image_label.grid(row=0, column=0, padx=20, pady=20, sticky="n")

# Function result display label or canvas
result_canvas = None
result_label = ctk.CTkLabel(master=main_frame, text="", width=400, height=400)
result_label.grid(row=0, column=1, padx=20, pady=20, sticky="n")
result_label.grid_remove()  # Hide until a function is applied

def display_image(image):
    """Helper function to display the image on the image_label."""
    global tk_image
    tk_image = ImageTk.PhotoImage(image)
    image_label.configure(image=tk_image)
    image_label.image = tk_image  # Maintain reference to avoid garbage collection

def display_result(image):
    """Helper function to display the result on the result_label."""
    global result_label_image, result_canvas
    if result_canvas:
        result_canvas.get_tk_widget().destroy()
        result_canvas = None

    if isinstance(image, Image.Image):  # If the result is a PIL image
        result_label_image = ImageTk.PhotoImage(image)
        result_label.configure(image=result_label_image)
        result_label.image = result_label_image  # Maintain reference to avoid garbage collection
        result_label.grid()  # Show the result label
    elif isinstance(image, plt.Figure):  # If the result is a Matplotlib figure
        result_label.grid_remove()  # Hide the image label
        result_canvas = FigureCanvasTkAgg(image, master=main_frame)
        result_canvas.get_tk_widget().grid(row=0, column=1, padx=20, pady=20, sticky="n")
        result_canvas.draw()

def browse_file():
    global global_image, image_instance
    file_path = filedialog.askopenfilename(
        title="Select a File",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")],
    )
    if file_path:
        global_image = Image.open(file_path)
        image_instance = ImageClass(global_image)  # Initialize ImageClass with the loaded image
        display_image(global_image)
        result_label.grid_remove()  

def apply_function(function_name):
    if image_instance:
        function_map = {
            "Get Histogram": image_instance.get_histogram,
            "Halftone": image_instance.halftone,
            "Sobel": lambda: image_instance.simple(operation="sobel"),
            "Prewitt": lambda: image_instance.simple(operation="prewitt"),
            "Kirsch": lambda: image_instance.kirsch(),
            "Low-Filter": lambda: image_instance.filtering(level="low"),
            "High-Filter": lambda: image_instance.filtering(level="high"),
            "Median Filter": lambda: image_instance.filtering("median"),
            "Range": lambda: image_instance.advanced(operation="range"),
            "Variance": lambda: image_instance.advanced(operation="var"),
            "Difference of Gaussians 7X7": lambda: image_instance.DoG(level=7),
            "Difference of Gaussians 9X9": lambda: image_instance.DoG(level=9),
            "Difference": lambda: image_instance.advanced(operation="diffrence"),
            "Homogeneity": lambda: image_instance.advanced("homo"),
            "Threshold": lambda: image_instance.thresh_hold(),
            "To GrayScale": lambda: image_instance.grayscale_image(),
            "Contrast Based": lambda: image_instance.contrast_based(),
            "Invert": lambda: image_instance.operations(operation="invert"),
            "Add": lambda: image_instance.operations(operation="add"),
            "Subtract Image": lambda: image_instance.operations(operation="subtract"),
            "Manual": lambda: image_instance.histogram_segmentation(operation="manual"),
            "Valley": lambda: image_instance.histogram_segmentation(operation="valley"),
            "Peak": lambda: image_instance.histogram_segmentation(operation="peak"),
            "Adaptive": lambda: image_instance.histogram_segmentation("adapt"),
            "To GrayScale":lambda:image_instance.grayscale_image
        }
        if function_name in function_map:
            result = function_map[function_name]()
            display_result(result)

# Browse File Button
browse_button = ctk.CTkButton(
    master=main_frame,
    text="Browse File",
    command=browse_file
)
browse_button.grid(row=1, column=0, columnspan=2, pady=20)

# Buttons for image processing functions
button_frame = ctk.CTkFrame(master=main_frame)
button_frame.grid(row=2, column=0, columnspan=2, pady=20)

button_names = [
    "Get Histogram", "Halftone", "Sobel", "Prewitt",
    "Kirsch", "Low-Filter", "High-Filter", "Median Filter", "Range", "Variance", 
    "Difference of Gaussians 7X7", "Difference", "Homogeneity", "Threshold", 
    "To GrayScale", "Contrast Based", "Invert", "Add", "Subtract Image",
    "Manual", "Valley", "Peak", "Adaptive","Difference of Gaussians 9X9"
]

# Define the number of columns in the grid
columns = 5

# Place buttons in a grid
for i, name in enumerate(button_names):
    row = i // columns  # Determine the row number
    col = i % columns   # Determine the column number
    button = ctk.CTkButton(
        master=button_frame,
        text=name,
        command=lambda n=name: apply_function(n)
    )
    button.grid(row=row, column=col, padx=5, pady=5)

# Run the application
app.mainloop()
