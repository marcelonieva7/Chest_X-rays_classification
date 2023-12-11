import imageio.v2 as imageio
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.io.gfile import glob 

def get_img_paths(img_dir):
  """
    Get a list of image file paths from the specified directory.

    Parameters:
      img_dir (str): The directory containing the images.

    Returns:
      list: A list of image file paths.
  """
  impaths_jpeg = glob(img_dir + '/*/*/*.jpeg')
  impaths_jpg = glob(img_dir + '/*/*/*.jpg')
  impaths_png = glob(img_dir + '/*/*/*.png')

  impaths = impaths_jpeg + impaths_jpg + impaths_png
  
  #throw an error if no images are found
  if len(impaths) == 0:
    raise ValueError('No images found in the specified directory.')
  return impaths

def extract_image_props(impath):
  """
  Extracts various properties of an image given its path.

  Parameters:
    impath (str): The path of the image file.

  Returns:
    tuple: A tuple containing the following elements:
      - datasplit (str): The name of the data split the image belongs to.
      - impath (str): The path of the image file.
      - label (str): The label of the image.
      - xsize (int): The width of the image in pixels.
      - ysize (int): The height of the image in pixels.
      - br_med (float): The median brightness value of the image.
      - br_std (float): The standard deviation of brightness values in the image.
  """
  label = impath.split('/')[-2]
  label = label.lower().replace(' ', '_')
  
  im = imageio.imread(impath)
  br_med = np.median(im)
  br_std = np.std(im)
  xsize = im.shape[1]
  ysize = im.shape[0]
  datasplit = impath.split('/')[-3]
  
  return datasplit, impath, label, xsize, ysize, br_med, br_std

def extract_image_props_all(DIR):
  """
  Extracts image properties for all images in a given directory.
  
  Parameters:
  - DIR: A string representing the directory path containing the images.
  
  Returns:
  - df: A pandas DataFrame containing the extracted image properties. The DataFrame has the following columns:
        - datasplit: A string representing the data split of the image.
        - path: A string representing the path of the image.
        - label: A string representing the label of the image.
        - xsize: An integer representing the width of the image.
        - ysize: An integer representing the height of the image.
        - br_med: A float representing the median brightness of the image.
        - br_std: A float representing the standard deviation of the brightness of the image.
  """
  impaths = get_img_paths(DIR)
  
  df = pd.DataFrame(columns=['datasplit', 'path', 'label', 'xsize', 'ysize',
                              'br_med', 'br_std'])

  for i in impaths:
    datasplit, impath, label, xsize, ysize, br_med, br_std = extract_image_props(i)
    new_row = {'datasplit':datasplit, 'path':impath, 'label':label, 'xsize':xsize, 'ysize':ysize, 'br_med':br_med, 'br_std':br_std}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
  
  df['aspectratio_yx'] = df.ysize / df.xsize
  
  return df

def get_random_images(df, datasplit, nsize=5):
  """
  Generates a list of random images from a dataframe based on the specified data split and label.

  Args:
    df (DataFrame): The dataframe containing the image data.
    datasplit (str): The data split to filter the dataframe by.
    nsize (int, optional): The number of random images to generate for each label. Defaults to 5.

  Returns:
    List: A list of randomly selected image paths.
  """
  labels = df['label'].unique()    
  images = []

  
  split_data = df[df['datasplit'] == datasplit]
  for label in labels:
      label_data = split_data[split_data['label'] == label]
      images.append(label_data['path'].sample(n=nsize).values)
      
  
  return images

def show_images(img_paths):
  """
    Display multiple images given their file paths.
  
    Args:
      img_paths (list): A list of file paths of the images to display.
  
    Returns:
      None
  """
  fig, axes = plt.subplots(1, len(img_paths), figsize=(15, 5))

  # Print the name of the folder containing the images
  print('\t' + os.path.basename(os.path.dirname(img_paths[0])))

  for i, path in enumerate(img_paths):
    # Load the image
    img = Image.open(path)

    # Convert to RGB if not in RGB mode
    if img.mode != 'RGB':
      img = img.convert('RGB')
    
    title = os.path.basename(path)
    # Truncate the title if it is too long
    title = title[:20] + "..." if len(title) > 20 else title

    # Display the image
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(title)

  plt.show()

COLORS = ['#009688', '#00BCD4', '#03A9F4', '#3F51B5', '#673AB7', '#9C27B0', '#E91E63']

def multivariate_grid(df, x, y, lim={'x': None, 'y': None}):
  """
  Generate a multivariate grid plot using seaborn.

  Parameters:
  - df: the DataFrame containing the data.
  - x: the name of the column to be plotted on the x-axis.
  - y: the name of the column to be plotted on the y-axis.
  - lim: a dictionary specifying the limits of the x and y axes. 
    Defaults to {'x': None, 'y': None}.

  Returns:
  None
  """
  g = sns.jointplot(x=x,
                    y=y,               
                    data=df,
                    hue='datasplit',
                    marginal_kws={"alpha": 0.45, 'linewidths': 2},
                    ratio=5,
                    xlim=lim['x'],
                    ylim=lim['y'],
                    joint_kws={'alpha': 0.65},
                    palette={'train': COLORS[1], 'test': COLORS[6]},
)
  plt.legend(loc='upper left')
  g.ax_joint.grid(False)
  g.ax_marg_x.grid(False)
  g.ax_marg_y.grid(False)
  plt.show()

def plot_average_intensity_histograms(image_paths_list, target_shape=(256, 256)):
    """
    Plot average intensity histograms for a list of images.

    Parameters:
    - image_paths_list (List[List[str]]): A list of lists containing paths to images.
    - target_shape (Tuple[int, int]): The target shape to resize the images to. Default is (256, 256).

    Returns:
    - None

    This function takes a list of lists of image paths and plots the average intensity histograms
    for each group of images. It first resizes each image to the target shape, converts them to
    grayscale, and then calculates the average pixel intensity of each image. The average intensity
    histograms are then plotted using matplotlib.

    Note:
    - The target_shape parameter is optional. If not provided, the default target shape is (256, 256).
    """
    intensity_arrays_list = []
    classes = []

    for image_paths in image_paths_list:
        intensity_arrays = []

        for path in image_paths:
            image_pil = Image.open(path)
            original_shape = image_pil.size[::-1]
            resized_image = image_pil.resize(target_shape, Image.BILINEAR)
            resized_image = resized_image.convert('L')
            image_array = np.array(resized_image)

            intensity_arrays.append(image_array.ravel())

        intensity_arrays_list.append(np.mean(intensity_arrays, axis=0))
        classes.append(os.path.basename(os.path.dirname(image_paths[0])))

    plt.figure(figsize=(15, 5))

    for i, intensity_array in enumerate(intensity_arrays_list):
        plt.subplot(1, 3, i+1)
        plt.hist(intensity_array, bins=256, color='gray', alpha=0.7)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title(f'Average Pixel Intensity \n \n {classes[i]}')

    plt.tight_layout()
    plt.show()