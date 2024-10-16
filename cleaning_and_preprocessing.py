#cleaning of the images for data augmentation
# 1. First we start normalizing the images
import cv2

# first write a definition for the normalizing of the images, which can later be used when running over all the images
def image_normalization (image):
    """
    This definition will first grayscale images to keep the contrast in the image, whenever it is normalized.
    It will normalize the image to a [0,1] scale.
    :param image: the image that will be normalized
    :return: a normalized image
    """

    #greyscale the image
    #first greyscale so that the contrast will be enhanced
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #normalize the image
    normalized_image = cv2.normalize(greyscale_image, None , alpha=0, beta=255, norm_type= cv2.NORM_MINMAX)

    #convert back to color to be used in the SAM models
    normalized_image_color = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

    #return the final output
    return normalized_image_color

#normalize all images
import os

#change to your own folder if you run the code
path_to_folder = r'C:\Users\20224435\PycharmProjects\DC3_Group8\all images'
path_to_folder_output = r'C:\Users\20224435\PycharmProjects\DC3_Group8\normalized_images'

#define a definition for the background noise removal
def background_noise_removal (image):
    """
    removes the background noise of the images. Returns the image without the background noise.
    :param image: image, the path to the image
    :return: the image without background noise.
    """

    #denoising a coloured image
    cleaned_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cleaned_image

#run over all images in the folder
#i for visualizing how far it is
i = 1
for image in os.listdir(path_to_folder):

    #get the path to the image
    image_path = os.path.join(path_to_folder, image)
    image_input = cv2.imread(image_path)

    #normalize the image
    normalized_image = image_normalization(image_input)

    #remove the background noise of the normalized image
    cleaned_image = background_noise_removal(normalized_image)

    #make path to save the new image
    output_image_path = os.path.join(path_to_folder_output, image)
    #save new image in new folder
    cv2.imwrite(output_image_path, cleaned_image)
    #check which file youre at.
    print(i)
    i += 1




