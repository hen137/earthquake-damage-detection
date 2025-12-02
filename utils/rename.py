import os

# os.makedirs('./data/flood/rename/Image')
# os.makedirs('./data/flood/rename/Mask')
for i, (image, mask) in enumerate(zip(os.listdir('./data/flood/test/Image'), os.listdir('./data/flood/test/Mask'))):
    # print(int(image.split('.')[0]))
    # if int(image.split('.')[0]) != int(mask.split('.')[0]):
    #     print(image, mask)
    os.rename(os.path.join('./data/flood/test/Image', image), './data/flood/test/Image' + f'/{i}.jpg')
    os.rename(os.path.join('./data/flood/test/Mask', mask), './data/flood/test/Mask' + f'/{i}.png')