import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio


def imread(path, is_grayscale=True):
    """
    Read image from the giving path.
    Default value is gray-scale, and image is read by YCbCr format as the paper.
    """
    if is_grayscale:
        return iio.imread(path, as_gray=True, pilmode='YCbCr').astype(np.float32)
    else:
        return iio.imread(path, pilmode='YCbCr').astype(np.float32)


def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def preprocess(path, scale=3):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with interpolation
    Args:
      path: file path of desired file
      input_: image applied interpolation (low-resolution)
      label_: image with original resolution (high-resolution), groundtruth
    """
    image = imread(path, is_grayscale=True)
    label_ = modcrop(image, scale)

    # Must be normalized
    label_ = label_ / 255.

    input_ = ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
    input_ = ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)

    return input_, label_


"""Define the model weights and biases 
"""
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=9//2) # we want to preserve the image size, so we could either make the padding equals to kernal_size//2 or use padding = "same" ( pytotch will automatically compute the padding that preserves the image for us)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=1//2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=5//2)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out


"""Load the pre-trained model file
"""
model = SRCNN()
model.load_state_dict(torch.load('./model/model.pth'))
model.eval()

"""Read the test image
"""
LR_image, HR_image = preprocess('./image/butterfly_GT.bmp')
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(LR_image, axis=0), axis=0)
input_ = torch.from_numpy(input_)

"""Run the model and get the SR image
"""
with torch.no_grad():
    output_ = model(input_)

# convert the output to np image format
output_ = np.array(torch.squeeze(output_))

# PSNR calculation

LR_HR_PSNR = peak_signal_noise_ratio(LR_image,HR_image)
HRSRCNN_HR_PSNR = peak_signal_noise_ratio(HR_image,output_)

# Save results
iio.imsave("./Ground_Truth_HR.png",HR_image)  # input to the model (low resloution image)
iio.imsave("./SRCNN_HR.png",output_) # output of the model
iio.imsave("./Base_HR.png",LR_image)  # input to the model (low resloution image)

print("PSNR between SRCNN output and ground truth = {:0.4f}".format(HRSRCNN_HR_PSNR))
print("PSNR between low resolution (input) image and ground truth image = {:0.4f}".format(LR_HR_PSNR))
print()
print("Visualization of results : ")

#Visualize results
fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True)

axs[0].imshow(LR_image,cmap='gray')
axs[0].set_title("Low resolution image")
axs[0].set_xlabel('PSNR = {:0.2f}'.format(LR_HR_PSNR), fontsize=12)
axs[0].set_xticks([])
axs[0].set_yticks([])

axs[1].imshow(output_,cmap='gray')
axs[1].set_title("SRCNN output")
axs[1].set_xlabel('PSNR = {:0.2f}'.format(HRSRCNN_HR_PSNR), fontsize=12)
axs[1].set_xticks([])
axs[1].set_yticks([])

axs[2].imshow(HR_image,cmap='gray')
axs[2].set_title("High resolution image")
axs[2].set_xlabel('Ground truth Image', fontsize=12)
axs[2].set_xticks([])
axs[2].set_yticks([])