from multiprocessing import Pool
from itertools import repeat
from PIL import Image
import os
from pudb import set_trace

SIZE = (600,600)
save_directory = 'size_' + str(SIZE[0]) + '_' + str(SIZE[1])


def get_image_paths(folder):
  return [os.path.join(folder, f)
      for f in os.listdir(folder)
      if '.JPG' in f]

def create_thumbnail(filename, save_dir_path):
  im = Image.open(filename)
  im.thumbnail(SIZE, Image.ANTIALIAS)
  save_path = os.path.join(save_dir_path, os.path.splitext(os.path.basename(filename))[0] +
                           '-' + save_directory +
                           os.path.splitext(os.path.basename(filename))[1])
  im.save(save_path)

if __name__ == '__main__':
  folder = '/media/auro/RAID 5/SushiPics/orig'
  save_dir_path = os.path.join(os.path.dirname(folder), save_directory)
  os.makedirs(save_dir_path, exist_ok=True)

  
images = get_image_paths(folder)
    
pool = Pool()
pool.starmap(create_thumbnail, zip(images, repeat(save_dir_path)))

# for image in images:
#  create_thumbnail(image, save_dir_path)
