import cv2
import pathlib
import random

class GrayGenerator():
  def __init__(self, directory):
    self.directory = directory
    self.data_dir = pathlib.Path(directory)

  def random_crop(self, image, output_size):
    random_x = random.randrange(0, image.shape[0] - output_size[0] + 1)
    random_y = random.randrange(0, image.shape[1] - output_size[1] + 1)
    crop_image = image[random_x: random_x + output_size[0], random_y: random_y + output_size[1]]
    return crop_image

  def crop_resize(self, image, target_size):
    cropped_img = self.random_crop(image, target_size)
    return cv2.resize(cropped_img, (image.shape[0], image.shape[1]), cv2.INTER_CUBIC)

  def load_resized_image(self, input_directory, output_directory, output_size, extension='jpg'):
    dir = pathlib.Path(input_directory)
    path_list = list(dir.glob('*'))
    for path in path_list:
      img = cv2.imread(str(path))

      resized_original_img = cv2.resize(img, output_size, cv2.INTER_AREA)
      cv2.imwrite(output_directory + '/' + str(str(path).split('/')[-1].split('.')[0]) + '.' + extension,
                  resized_original_img)-4

  def load_RGB2GRAY(self, input_directory, output_directory, extension='jpg'):
    dir = pathlib.Path(input_directory)
    path_list = list(dir.glob('*'))
    for path in path_list:
      img = cv2.imread(str(path))

      gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      cv2.imwrite(output_directory + '/' + str(str(path).split('/')[-1].split('.')[0]) + '.' + extension, gray_img)

  def load_crop_resize(self, input_directory, output_directory, output_size, extension='jpg'):
    dir = pathlib.Path(input_directory)
    path_list = list(dir.glob('*'))
    for path in path_list:
      img = cv2.imread(str(path))

      crop_resize_img = self.crop_resize(img, output_size)
      cv2.imwrite(output_directory + '/' + str(str(path).split('/')[-1].split('.')[0]) + '.' + extension,
                  crop_resize_img)

  def load(self, resize_dir, img_dir, grayimg_dir, resize_size=[256, 256], crop_size=[256, 256], resize=True,
           extension='jpg'):
    if (resize):
      self.load_resized_image(self.data_dir, resize_dir, output_size=resize_size, extension=extension)
      self.load_crop_resize(resize_dir, img_dir, output_size=crop_size, extension=extension)
      self.load_RGB2GRAY(img_dir, grayimg_dir, extension=extension)
    else:
      self.load_crop_resize(self.data_dir, img_dir, output_size=crop_size, extension=extension)
      self.load_RGB2GRAY(img_dir, grayimg_dir, extension=extension)

class Blur_Generator():
  def __init__(self, directory):
    self.directory = directory
    self.data_dir = pathlib.Path(directory)

  def load_resized_image(self, input_directory, output_directory, output_size, extension='jpg'):
    dir = pathlib.Path(input_directory)
    path_list = list(dir.glob('*'))
    for path in path_list:
      img = cv2.imread(str(path))

      resized_original_img = cv2.resize(img, output_size, cv2.INTER_AREA)
      cv2.imwrite(output_directory + '/' + str(str(path).split('/')[-1].split('.')[0]) + '.' + extension,
                  resized_original_img)

  def load_blur_image(self, input_directory, output_directory, degree=(3, 31), extension='jpg'):
    dir = pathlib.Path(input_directory)
    path_list = list(dir.glob('*'))
    for path in path_list:
      img = cv2.imread(str(path))

      blurred_img = blur = cv2.blur(img, (random.randrange(degree[0], degree[1]), random.randrange(degree[0], degree[1])),
                                    anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
      cv2.imwrite(output_directory + '/' + str(str(path).split('/')[-1].split('.')[0]) + '.' + extension,
                  blurred_img)

  def load(self, resize_dir=None, blur_dir=None, resize=False, degree=(3, 31), resize_size=64, extension='jpg'):
    if resize:
      self.load_resized_image(self.data_dir, resize_dir, output_size=resize_size, extension=extension)
      self.load_blur_image(resize_dir, blur_dir, degree, extension)
    else:
      self.load_blur_image(self.data_dir, blur_dir, degree, extension)


