# import argparse
# import os
# from math import log10

# import numpy as np
# import pandas as pd
# import torch
# import torchvision.utils as utils
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# import pytorch_ssim
# from data_utils import TestDatasetFromFolder, display_transform
# from model import Generator

# parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
# parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
# parser.add_argument('--model_name', default='model2_80train_20val_50epoch.pth', type=str, help='generator model epoch name')
# opt = parser.parse_args()

# UPSCALE_FACTOR = opt.upscale_factor
# MODEL_NAME = opt.model_name

# results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
#            'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

# model = Generator(UPSCALE_FACTOR).eval()
# if torch.cuda.is_available():
#     model = model.cuda()
# model.load_state_dict(torch.load('epoch_saved/' + MODEL_NAME))
# print(os.cpu_count())

# test_set = TestDatasetFromFolder('temp_not_in_use', upscale_factor=UPSCALE_FACTOR)
# print(len(test_set))
# test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
# test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

# out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
# if not os.path.exists(out_path):
#     os.makedirs(out_path)

# with torch.no_grad():
#     for image_name, lr_image, hr_restore_img, hr_image in test_bar:
#         image_name = image_name[0]
#         lr_image = Variable(lr_image)
#         hr_image = Variable(hr_image)
#         if torch.cuda.is_available():
#             lr_image = lr_image.cuda()
#             hr_image = hr_image.cuda()

#         sr_image = model(lr_image)
#         mse = ((hr_image - sr_image) ** 2).data.mean()
#         psnr = 10 * log10(1 / mse)
#         ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

#         test_images = torch.stack(
#             [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
#             display_transform()(sr_image.data.cpu().squeeze(0))])
#         image = utils.make_grid(test_images, nrow=3, padding=5)
#         utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
#                         image_name.split('.')[-1], padding=5)

#         # save psnr\ssim
#         results[image_name.split('_')[0]]['psnr'].append(psnr)
#         results[image_name.split('_')[0]]['ssim'].append(ssim)

# out_path = 'statistics/'
# saved_results = {'psnr': [], 'ssim': []}
# for item in results.values():
#     psnr = np.array(item['psnr'])
#     ssim = np.array(item['ssim'])
#     if (len(psnr) == 0) or (len(ssim) == 0):
#         psnr = 'No data'
#         ssim = 'No data'
#     else:
#         psnr = psnr.mean()
#         ssim = ssim.mean()
#     saved_results['psnr'].append(psnr)
#     saved_results['ssim'].append(ssim)

# data_frame = pd.DataFrame(saved_results, results.keys())
# data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')


import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='model6_dropout_and_filters.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

print(f"Upscale Factor: {UPSCALE_FACTOR}")
print(f"Model Name: {MODEL_NAME}")

results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}, 'SRGAN': {'psnr': [], 'ssim': []}}

model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('epoch_saved/' + MODEL_NAME))

print("Model loaded successfully.")
print(os.cpu_count())

test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

print(f"Number of test images: {len(test_set)}")

test_bar = tqdm(test_loader, desc='[Testing benchmark datasets]')

csv_file_name = MODEL_NAME.split(".")[0]

out_path = 'benchmark_results/' + csv_file_name + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

def resize_image(image, size):
    transform = transforms.Resize(size)
    return transform(image)

with torch.no_grad():
    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        # print(f"Processing image: {image_name}")

        lr_image = Variable(lr_image)
        hr_image = Variable(hr_image)
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        sr_image = model(lr_image)
        
        # Resize sr_image to match hr_image dimensions if necessary
        if sr_image.size() != hr_image.size():
            sr_image = resize_image(sr_image, hr_image.size()[2:])

        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()  # Use .item() to get the scalar value

        # print(f"PSNR: {psnr}, SSIM: {ssim}")

        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
             display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)

        # print(f"RESULTS: {results}  PSNR: {psnr}   SSIM: {ssim}")
        # pdb.set_trace()   
        results['SRGAN']['psnr'].append(psnr)
        results['SRGAN']['ssim'].append(ssim)
        # results[image_name.split('_')[0]]['psnr'].append(psnr)
        # results[image_name.split('_')[0]]['ssim'].append(ssim)

print("Saving results...")

out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': []}

# Add progress bar for saving results
results_bar = tqdm(results.items(), desc='[Saving results]')
for dataset, item in results_bar:
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    if len(psnr) == 0 or len(ssim) == 0:
        psnr = 'No data'
        ssim = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + csv_file_name + '_test_results.csv', index_label='DataSet')

print("Results saved successfully.")
