set -ex
python test.py \
    --dataroot ./datasets/img_data  \
    --model test \
    --name first_pix2pix \
    --epoch latest \
    --netG unet_256 \
    --direction AtoB \
    --input_nc 1 \
    --output_nc 1 \
    --dataset_mode single \
    --num_test 1 \
    --ngf 1 \
    --ndf 1 \
    --preprocess resize \
    --no_flip \
    --batch_size 1 \
    --norm batch

# Warning: wandb package cannot be found. The option "--use_wandb" will result in error.
# Warning: wandb package cannot be found. The option "--use_wandb" will result in error.
# usage: test.py [-h] --dataroot DATAROOT [--name NAME] [--use_wandb] [--gpu_ids GPU_IDS] [--checkpoints_dir CHECKPOINTS_DIR]
#                [--model MODEL] [--input_nc INPUT_NC] [--output_nc OUTPUT_NC] [--ngf NGF] [--ndf NDF] [--netD NETD] [--netG NETG]
#                [--n_layers_D N_LAYERS_D] [--norm NORM] [--init_type INIT_TYPE] [--init_gain INIT_GAIN] [--no_dropout]
#                [--dataset_mode DATASET_MODE] [--direction DIRECTION] [--serial_batches] [--num_threads NUM_THREADS]
#                [--batch_size BATCH_SIZE] [--load_size LOAD_SIZE] [--crop_size CROP_SIZE] [--max_dataset_size MAX_DATASET_SIZE]
#                [--preprocess PREPROCESS] [--no_flip] [--display_winsize DISPLAY_WINSIZE] [--epoch EPOCH] [--load_iter LOAD_ITER]
#                [--verbose] [--suffix SUFFIX] [--results_dir RESULTS_DIR] [--aspect_ratio ASPECT_RATIO] [--phase PHASE] [--eval]
#                [--num_test NUM_TEST]

# optional arguments:
#   -h, --help            show this help message and exit
#   --dataroot DATAROOT   path to images (should have subfolders trainA, trainB, valA, valB, etc) (default: None)
#   --name NAME           name of the experiment. It decides where to store samples and models (default: experiment_name)
#   --use_wandb           use wandb (default: False)
#   --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU (default: 0)
#   --checkpoints_dir CHECKPOINTS_DIR
#                         models are saved here (default: ./checkpoints)
#   --model MODEL         chooses which model to use. [cycle_gan | pix2pix | test | colorization] (default: test)
#   --input_nc INPUT_NC   # of input image channels: 3 for RGB and 1 for grayscale (default: 3)
#   --output_nc OUTPUT_NC
#                         # of output image channels: 3 for RGB and 1 for grayscale (default: 3)
#   --ngf NGF             # of gen filters in the last conv layer (default: 64)
#   --ndf NDF             # of discrim filters in the first conv layer (default: 64)
#   --netD NETD           specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers
#                         allows you to specify the layers in the discriminator (default: basic)
#   --netG NETG           specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128] (default: resnet_9blocks)
#   --n_layers_D N_LAYERS_D
#                         only used if netD==n_layers (default: 3)
#   --norm NORM           instance normalization or batch normalization [instance | batch | none] (default: instance)
#   --init_type INIT_TYPE
#                         network initialization [normal | xavier | kaiming | orthogonal] (default: normal)
#   --init_gain INIT_GAIN
#                         scaling factor for normal, xavier and orthogonal. (default: 0.02)
#   --no_dropout          no dropout for the generator (default: False)
#   --dataset_mode DATASET_MODE
#                         chooses how datasets are loaded. [unaligned | aligned | single | colorization] (default: unaligned)
#   --direction DIRECTION
#                         AtoB or BtoA (default: AtoB)
#   --serial_batches      if true, takes images in order to make batches, otherwise takes them randomly (default: False)
#   --num_threads NUM_THREADS
#                         # threads for loading data (default: 4)
#   --batch_size BATCH_SIZE
#                         input batch size (default: 1)
#   --load_size LOAD_SIZE
#                         scale images to this size (default: 256)
#   --crop_size CROP_SIZE
#                         then crop to this size (default: 256)
#   --max_dataset_size MAX_DATASET_SIZE
#                         Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only
#                         a subset is loaded. (default: inf)
#   --preprocess PREPROCESS
#                         scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
#                         (default: resize_and_crop)
#   --no_flip             if specified, do not flip the images for data augmentation (default: False)
#   --display_winsize DISPLAY_WINSIZE
#                         display window size for both visdom and HTML (default: 256)
#   --epoch EPOCH         which epoch to load? set to latest to use latest cached model (default: latest)
#   --load_iter LOAD_ITER
#                         which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will
#                         load models by [epoch] (default: 0)
#   --verbose             if specified, print more debugging information (default: False)
#   --suffix SUFFIX       customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size} (default: )
#   --results_dir RESULTS_DIR
#                         saves results here. (default: ./results/)
#   --aspect_ratio ASPECT_RATIO
#                         aspect ratio of result images (default: 1.0)
#   --phase PHASE         train, val, test, etc (default: test)
#   --eval                use eval mode during test time. (default: False)
#   --num_test NUM_TEST   how many test images to run (default: 50)
