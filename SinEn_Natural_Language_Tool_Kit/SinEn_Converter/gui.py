import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import tkinter.messagebox as mb
import sinen_converter
import threading

class GUI():
    def __init__(self):
        # Initialize the root window for the GUI
        self.root = tk.Tk()
        
        # Create a SinEn object to handle the text conversion
        self.sinen = sinen_converter.SinEn()

    def main(self):
        """
        Main method to create the GUI and add the widgets.
        """
        # Set the title of the window
        self.root.title("SinEn Converter")
        
        # Set the size of the window
        self.root.geometry("720x480")

        # Create a label for the input text
        input_label = tk.Label(self.root, text="Enter the text to be transformed:")
        input_label.pack()

        # Create a text widget for the input text
        input_text = tk.Text(self.root, height=10, width=480)
        input_text.pack()

        # Create a label for the output text
        output_label = tk.Label(self.root, text="Transformed Text:")
        output_label.pack()

        # Create a text widget for the output text
        output_text = tk.Text(self.root, height=10, width=480)
        output_text.pack()

        # Create a frame to hold the conversion options
        convert_frame = tk.Frame(self.root)
        convert_frame.pack()

        # Create a StringVar to hold the selected conversion type
        selected = tk.StringVar(self.root,"phoneme")
        
        # Create a radiobutton to select phoneme conversion
        rb_phoneme = tk.Radiobutton(convert_frame, text="Phoneme", value="phoneme", variable=selected)
        rb_phoneme.grid(row=0, column=0, padx=5, pady=10, sticky=tk.W)
        
        # Create a radiobutton to select Singlish conversion
        rb_singlish = tk.Radiobutton(convert_frame, text="Singlish", value="singlish", variable=selected)
        rb_singlish.grid(row=0, column=1, padx=5, pady=10, sticky=tk.W)

        # Create a button to initiate the conversion
        convert_button = tk.Button(convert_frame, text="Convert", command=lambda:self.convert(input_text,selected,output_text,convert_button))
        convert_button.grid(row=1, column=0, columnspan=2, padx=5, sticky=tk.E + tk.W)

        # Create a frame to hold the file and clipboard buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X,padx=5,pady=10)

        # Create the "Load from File" button
        load_button = tk.Button(button_frame, text="Load from File", command=lambda:self.load_file(input_text,convert_button))
        load_button.pack(side=tk.LEFT, padx=5)

        # Create the "Save as File" button
        save_button = tk.Button(button_frame, text="Save as File", command=lambda:self.save_as_file(output_text))
        save_button.pack(side=tk.LEFT, padx=5)

        # Create the "Paste" button
        paste_button = tk.Button(button_frame, text="Paste", command=lambda:self.paste_text(input_text))
        paste_button.pack(side=tk.LEFT, padx=5)

        # Create the "Copy" button
        copy_button = tk.Button(button_frame, text="Copy", command=lambda:self.copy_text(output_text))
        copy_button.pack(side=tk.LEFT, padx=5)

        # Start the GUI event loop
        self.root.mainloop()

    # This method converts text from the input widget based on the selected option. 
    # It uses the sinen object's convertToPhoneme or convertToSinglish method to perform the conversion. 
    # The output is then set in the output widget.
    def convert(self, input_text, selected, output_text, convert_button):
        text = self.get_text(input_text)
        # Change the text of the button to "Converting..."
        convert_button.config(text="Converting...")
        # Disable the button
        convert_button.config(state="disabled")

        progress = tk.Toplevel(self.root)
        progress.title("Converting...")

        progress_bar = ttk.Progressbar(progress, orient="horizontal", length=200, mode="indeterminate")
        progress_bar.pack()
        progress_bar.start()

        self.root.update()

        def run_conversion():
            if selected.get() == "phoneme":
                # code to convert to Phoneme
                output = self.sinen.convert_to_phoneme(text)
            elif selected.get() == "singlish":
                # code to convert to Singlish
                output = self.sinen.convert_to_singlish(text)

            self.set_text(output_text, output)
            progress_bar.stop()
            progress.destroy()
            # Change the text of the button back to "Convert"
            convert_button.config(text="Convert")
            # Enable the button
            convert_button.config(state="normal")

        # Start a new thread to run the conversion process
        thread = threading.Thread(target=run_conversion)
        thread.start()

    # This method loads text from a file and sets it in the input widget.
    def load_file(self,input_text,convert_button):
        # Open a file dialog to select a file
        file_path = fd.askopenfilename()
    
        # Check if the file path is not None, meaning a file was selected
        if file_path:
            # Change the text of the convert button to "Loading..."
            convert_button.config(text="Loading...")
            # Disable the button
            convert_button.config(state="disabled")
    
            # Create a new window for the progress bar
            progress = tk.Toplevel(self.root)
            progress.title("Loading...")
            # Create a horizontal progress bar
            progress_bar = ttk.Progressbar(progress, orient="horizontal", length=200, mode="determinate")
            progress_bar.pack()
            progress_bar.start()
    
            # Update the main window to show the progress bar
            self.root.update()
    
            # Start a new thread to read the contents of the file
            thread = threading.Thread(target=self.read_file, args=(file_path, input_text, progress_bar, progress,convert_button))
            thread.start()
    
    # Read the contents of the file
    def read_file(self,file_path, input_text, progress_bar, progress,convert_button):
        # Call the open_file method of sinen and get the title and body of the file
        title, body = self.sinen.open_file(file_path)
        # Stop the progress bar
        progress_bar.stop()
        # Destroy the progress bar window
        progress.destroy()
    
        # Change the text of the convert button back to "Convert"
        convert_button.config(text="Convert")
        # Enable the button
        convert_button.config(state="normal")
    
        # Check if the body of the file is empty, meaning there were no errors opening the file
        if not body:
            # Set the contents of the file in the input widget
            self.set_text(input_text, title)
        else:
            # If the body is not empty, there were errors opening the file
            mb.showerror(title, body)
    
    # This method saves the text from the output widget to a file.
    def save_as_file(self, output_text):
        # File type options for the save file dialog
        file_types = [("Text files", "*.txt"),
                      ("All files", "*.*")]
        # Open the save file dialog
        file_path = fd.asksaveasfilename(defaultextension=".txt", filetypes=file_types)
        if file_path:
            # Save the contents of the output widget to the selected file
            title, body, error = self.sinen.save_file(file_path, self.get_text(output_text))
            # Show the dialog
            # If there was an error
            if error:
                mb.showerror(title, body)
            # If there was no error
            else:
                mb.showinfo(title, body)
    
    # This method pastes text from the clipboard to the input widget.
    def paste_text(self,input_text):
        # Get text from the clipboard
        text = self.root.clipboard_get()
        # Set the text in the input widget
        self.set_text(input_text,text)
    
    # This method copies the text from the output widget to the clipboard.
    def copy_text(self,output_text):
        try:
            # Clear the clipboard
            self.root.clipboard_clear()
            # Append the text from the output widget to the clipboard
            self.root.clipboard_append(self.get_text(output_text))
            # Show success message
            mb.showinfo("Copied!", "Text copied to clipboard successfully.")
        except Exception as e:
            # Show error message
            mb.showerror("Error", "An error occurred while copying the text.\n" + str(e))

    
    # This method sets text in the specified text widget.
    def set_text(self,text_widget,text):
        # Delete existing text in the widget
        text_widget.delete("1.0",tk.END)
        # Insert the new text in the widget
        text_widget.insert("1.0",text)
    
    # This method returns the text from the specified text widget.
    def get_text(self,text_widget):
        # Get the text from the widget
        text = text_widget.get("1.0", tk.END)
        return text
    