import imageio
import os

path = '/Users/varunmanmohangudibanda/Desktop/grad school/Research/Convection/RBC_Time_Marching/RB_Convection/animations/' 

image_folder = os.fsencode(path)

filenames = []

for file in os.listdir(image_folder):
    filename = os.fsdecode(file)
    if filename.endswith( ('.jpg', '.png', '.gif') ):
        filenames.append(os.path.join(path, filename))

filenames.sort()

images = [imageio.imread(f) for f in filenames]

imageio.mimsave(os.path.join('movie.gif'), images, duration = 0.04) # modify duration as needed