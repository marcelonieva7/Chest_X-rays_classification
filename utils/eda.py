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
  Extracts image properties from the given image path.
  
  Args:
      impath (str): The path of the image file.
  
  Returns:
      tuple: A tuple containing the following image properties:
          - datasplit (str): The split of the image data.
          - impath (str): The path of the image file.
          - label (str): The label of the image.
          - xsize (int): The width of the image.
          - ysize (int): The height of the image.
          - br_med (float): The median brightness of the image.
          - br_std (float): The standard deviation of the brightness of the image.
          - intens (ndarray): The intensity values of the image.
  """
  label = impath.split('/')[-2]
  label = label.lower().replace(' ', '_')

  image_pil = Image.open(impath)
  resized_image = image_pil.resize((256,256), Image.BILINEAR)
  resized_image = resized_image.convert('L')
  image_array = np.array(resized_image)

  
  intens = image_array.ravel()
  im = imageio.imread(impath)
  br_med = np.median(im)
  br_std = np.std(im)
  xsize = im.shape[1]
  ysize = im.shape[0]
  datasplit = impath.split('/')[-3]
  
  return datasplit, impath, label, xsize, ysize, br_med, br_std, intens

def extract_image_props_all(DIR):
  """
  Extracts image properties for all images in a given directory.

  Args:
      DIR (str): The directory path where the images are located.

  Returns:
      pandas.DataFrame: A DataFrame containing the extracted image properties. The columns of the DataFrame include:
        - datasplit: The data split of the image.
        - path: The path of the image.
        - label: The label of the image.
        - xsize: The size of the image along the x-axis.
        - ysize: The size of the image along the y-axis.
        - br_med: The median brightness of the image.
        - br_std: The standard deviation of brightness of the image.
        - intens: The intensity of the image.
        - aspectratio_yx: The aspect ratio of the image, calculated as ysize / xsize.
  """
  impaths = get_img_paths(DIR)
  
  df = pd.DataFrame(columns=['datasplit', 'path', 'label', 'xsize', 'ysize',
                              'br_med', 'br_std'])

  for i in impaths:
    datasplit, impath, label, xsize, ysize, br_med, br_std, intens = extract_image_props(i)
    new_row = {'datasplit':datasplit, 'path':impath, 'label':label, 'xsize':xsize, 'ysize':ysize, 'br_med':br_med, 'br_std':br_std, 'intens':intens}
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

def plot_average_intensity_histograms(df, labels):
    """
    Generate a histogram of the average pixel intensity for each label in the given 
    DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - labels (list): A list of labels to plot the histograms for.

    Returns:
    - None
    """
    colors = [COLORS[3], COLORS[5], COLORS[1]]
    intensity_list = []

    for l in labels:
      intens = df[df['label'] == l].intens
      intens_val = intens.values.tolist()
      intens_mean = np.mean(intens_val, axis=0)
      intensity_list.append(intens_mean)

    plt.figure(figsize=(15, 5))

    for i, intensity_array in enumerate(intensity_list):
        plt.subplot(1, 3, i+1)
        plt.hist(intensity_array, bins=256, color=colors[i], )
      
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title(f'Average Pixel Intensity \n \n {labels[i]}')

    plt.tight_layout()
    plt.show()