import cv2
import numpy as np
import os

class ImageInterpreter:
    def __init__(self, image, choice):
        # Set image and kernels_list to the type of kernel user chose
        self.image = image
        self.kernels_list = self.set_kernel(choice)

    # If the user's choice was Gaussian, set the kernel to a Gaussian matrix
    def Gaussian_distro_matrix(self):
        kernels = []
        gaussian_kernel = np.array([[1/16, 1/8, 1/16],
                                    [1/8,  1/4, 1/8],
                                    [1/16, 1/8, 1/16]])
        kernels.append(gaussian_kernel)
        
        return kernels
    
    # If the user's choice was layered, set the kernel to a layered matrix
    def shape_kernel(self):
        kernels = []
        
        kernel = np.array(  [[-1,-1,-1],
                             [1,1,1],
                             [0,0,0]])
        kernels.append(kernel)

        kernel = np.array(  [[-1,1,0],
                             [-1,1,0],
                             [-1,1,0]])
        kernels.append(kernel)

        kernel = np.array(  [[0,0,0],
                             [1,1,1],
                             [-1,-1,-1]])
        kernels.append(kernel)

        kernel = np.array(  [[0,1,-1],
                             [0,1,-1],
                             [0,1,-1]])
        kernels.append(kernel)

        # Add more kernels if needed

        return kernels

    # Set the kernel based on the user's choice here. If it's an invalid input, throw an error and exit
    def set_kernel(self, choice):
        if choice == "1":
            return self.Gaussian_distro_matrix()
        elif choice == "2":
            return self.shape_kernel()
        else:
            print("\nError in operation type. Incorrect argument passed. Please make sure you choose a valid number.")
            exit()

    def calculate_new_pixel(self, old_matrix, kernel):
        new_pixel = [0, 0, 0]  # Initialize a new pixel with zeros for R, G, and B values

        for y in range(3):
            for x in range(3):
                kernel_value = kernel[x][y]
                old_img_pixel = old_matrix[x][y]

                # Multiply each color channel by the kernel value
                new_pixel[0] += kernel_value * old_img_pixel[0]  # Red channel
                new_pixel[1] += kernel_value * old_img_pixel[1]  # Green channel
                new_pixel[2] += kernel_value * old_img_pixel[2]  # Blue channel

        return new_pixel

    def process_image(self):
        height, width, channels = self.image.shape

        new_images = []

        for kernel in self.kernels_list:
            new_image = np.copy(self.image)

            # Translate from top down, left to right
            for y in range(1, height - 2):
                for x in range(1, width - 2):

                    top_left =      self.image[x-1, y-1]
                    top_mid =       self.image[x, y-1]
                    top_right =     self.image[x+1, y-1]

                    center_left =   self.image[x-1, y]
                    center_mid =    self.image[x, y]
                    center_right =  self.image[x+1, y]

                    bottom_left =   self.image[x-1, y+1]
                    bottom_mid =    self.image[x, y+1]
                    bottom_right =  self.image[x+1, y+1]

                    # Matrix of 3x3 pixel window
                    matrix_old = [[top_left, top_mid, top_right],
                                    [center_left, center_mid, center_right],
                                    [bottom_left, bottom_mid, bottom_right]]
                    
                    new_val = self.calculate_new_pixel(matrix_old, kernel)
                    new_image[x, y] = [new_val[0], new_val[1], new_val[2]]


            new_images.append(new_image)

        return new_images

def main():
    # Get the current working directory
    current_directory = os.getcwd()
    # List of image file extensions to filter
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    # Use list comprehension to filter only image files
    image_files = [file for file in os.listdir(current_directory) if os.path.isfile(file) and
                   os.path.splitext(file)[1].lower() in image_extensions and file != 'output']
    # Loop through the list of image files and print their names
    for file in image_files:
        print(file)
        
    # Input filename. If it's an invalid file name, throw an error and quit
    filename = input("\nChoose the image you want to upload: ")
    if filename not in image_files:
        print("\nError: file does not exist, exiting program.")
        exit()

    # Input convolution type, error handling will be dealt with in the constructor of the Image interpreter
    Convolution_type = input("\n1. Gaussian Blur\n2. Layered\n\nChoose the number for the type of Convolution to operate: ")

    image = cv2.imread(filename) 
    image_interpreter = ImageInterpreter(image, Convolution_type)
    new_images = image_interpreter.process_image()

    # Save processed images
    if len(new_images) == 1:
        output_filename = f'output_{os.path.splitext(filename)[0]}.jpeg'  # Output filename based on input filename
        cv2.imwrite(output_filename, new_images[0])
        print(f"Processed image saved as: {output_filename}")
    elif len(new_images) == 4:
        for i, new_image in enumerate(new_images):
            output_filename = f'output_{os.path.splitext(filename)[0]}_{i + 1}.jpeg'  # Output filename based on input filename
            cv2.imwrite(output_filename, new_image)
            print(f"Processed image {i + 1} saved as: {output_filename}")

if __name__ == "__main__":
    main()
