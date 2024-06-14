# Video-SRGAN

This is a video super-resolution generative adversarial network model created using a single image SRGAN model as the base architecture. This model is used to research potential techniques that could improve such models for super-resolution tasks. To maintain temporal consistency across the sequence of frames, we integrate a long short-term memory (LSTM) layer in the generator network.  Additionally, we incorporate
temporal smoothing techniques1 by maintaining motion continuity between frames, employing Gaussian smoothing to the motion vectors to average them with a Gaussian-weighted sum. Finally, a temporal loss function is added to ensure coherence between consecutive frames. Further details can be found in our research paper: [temp no link until approved]

benchmark_results, epochs, epoch_saved, and training_results will be filled as you run your models on datasets. Do not worry if they are initially empty.

# Datasets

Any training dataset can be used as long as you have separate the videos into high and low quality videos. This is important for the GAN architecture. 

If you want to follow our example, we use this dataset: https://github.com/IanYeung/RealVSR?tab=readme-ov-file

We train on videos.zip and test on LQ_test.zip and GQ_test.zip. If needed, there are utility scripts in utils/ provided that may help in managing these videos.

# Training

In train.py, change the paths to the training and validation set to yours respectively. Alter the available arguments as desired.
Do the same for the test set path in test_benchmark.py.

`python train.py`

After running this script, in epochs/ you will find 50 iterations of netG and netD .pth. Move the last netG.pth file to epoch_saved. You will use this for testing.

Now run the testing script on the .pth model that you trained:

`python test_benchmark.py`

You will find both the training and testing results in statistics/.
