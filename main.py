import sys
import os
import numpy as np
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ScoliosisDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('Scoliosis Detector')
        self.geometry('1000x700')
        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkLabel(main_frame, text="Scoliosis Detector", font=ctk.CTkFont(size=24, weight="bold"))
        header.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # Image frame
        self.image_frame = ctk.CTkFrame(main_frame)
        self.image_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.image_frame.grid_columnconfigure((0, 1), weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)

        # Original image
        self.original_image_label = ctk.CTkLabel(self.image_frame, text="Original Image")
        self.original_image_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Processed image
        self.processed_image_label = ctk.CTkLabel(self.image_frame, text="Processed Image")
        self.processed_image_label.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Results
        self.result_text = ctk.CTkTextbox(main_frame, height=150, font=ctk.CTkFont(size=12))
        self.result_text.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.result_text.insert("0.0", "Upload an image to get started")

        # Upload button
        self.upload_btn = ctk.CTkButton(main_frame, text='Upload Image', command=self.upload_image)
        self.upload_btn.grid(row=3, column=0, padx=20, pady=(10, 20))

    def load_model(self):
        try:
            self.model = load_model('improved_spine_rec_model.h5')
        except FileNotFoundError:
            messagebox.showerror("Error", "Model file 'improved_spine_rec_model.h5' not found. Make sure it's in the same directory.")
            self.quit()

    def upload_image(self):
        file_name = filedialog.askopenfilename(
            initialdir=os.path.expanduser("~"),
            title="Open Image File",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"),)
        )
        if file_name:
            if self.is_valid_image(file_name):
                self.original_image = cv2.imread(file_name)
                self.display_image(self.original_image, self.original_image_label, "Original Image")
                self.process_image(file_name)
            else:
                messagebox.showerror("Error", "Invalid image file. Please select a valid image.")

    def is_valid_image(self, file_path):
        try:
            Image.open(file_path)
            return True
        except:
            return False

    def process_image(self, file_path):
        def processing_thread():
            try:
                # Load and preprocess the image
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, (224, 224))
                img_array = img_to_array(img_resized)
                img_array = img_array.astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Make predictions
                predictions = self.model.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class] * 100

                # Image segmentation
                _, binary = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours on a color image
                segmented = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(segmented, contours, -1, (0, 255, 0), 2)

                # Prepare result text
                classes = ['Normal', 'Scoliosis', 'Spondylolisthesis']
                result_text = f"Image Analysis:\nPrediction: {classes[predicted_class]}\nConfidence: {confidence:.2f}%"
                for i, class_name in enumerate(classes):
                    if i != predicted_class:
                        result_text += f"\nProbability of {class_name}: {predictions[0][i] * 100:.2f}%"
                
                result_text += f"\n\nImage Features:\nNumber of contours: {len(contours)}"

                self.after(0, lambda: self.update_result_label(result_text, segmented))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"An error occurred during image processing: {str(e)}"))

        threading.Thread(target=processing_thread).start()

    def update_result_label(self, image_result, processed_image):
        result_text = f"{image_result}\n"
        
        # Simple risk assessment based on image analysis

        result_text += "\nPlease consult with a medical professional for further accurate diagnosis."
        
        self.result_text.delete("0.0", "end")
        self.result_text.insert("0.0", result_text)
        
        self.display_image(processed_image, self.processed_image_label, "Processed Image")

    def display_image(self, img, label, text):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil.resize((400, 400), Image.LANCZOS))
        label.configure(image=img_tk, text=text)
        label.image = img_tk  # Keep a reference

if __name__ == '__main__':
    app = ScoliosisDetectorApp()
    app.mainloop()
