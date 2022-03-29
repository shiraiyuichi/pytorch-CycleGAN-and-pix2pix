set -ex
python train.py \
    --dataroot ./datasets/img_data \
    --name first_pix2pix \
    --model pix2pix \
    --direction AtoB \
    --input_nc 1 \
    --output_nc 1 \
    --dataset_mode unaligned \
    --serial_batches \
    --preprocess resize_and_crop \
    --no_flip \
    --netG unet_256 \
    --netD pixel \
    --lambda_L1 1 \
    --lr 1 \
    --norm batch \
    --pool_size 0 \
    --n_epochs 1 \
    --n_epochs_decay 1 \
    --ngf 1 \
    --ndf 1 \
    --batch_size 1 \

# Warning: wandb package cannot be found. The option "--use_wandb" will result in error.
# usage: train.py [-h] --dataroot DATAROOT [--name NAME] [--use_wandb] [--gpu_ids GPU_IDS]
#                 [--checkpoints_dir CHECKPOINTS_DIR] [--model MODEL] [--input_nc INPUT_NC]
#                 [--output_nc OUTPUT_NC] [--ngf NGF] [--ndf NDF] [--netD NETD] [--netG NETG]
#                 [--n_layers_D N_LAYERS_D] [--norm NORM] [--init_type INIT_TYPE]
#                 [--init_gain INIT_GAIN] [--no_dropout] [--dataset_mode DATASET_MODE]
#                 [--direction DIRECTION] [--serial_batches] [--num_threads NUM_THREADS]
#                 [--batch_size BATCH_SIZE] [--load_size LOAD_SIZE] [--crop_size CROP_SIZE]
#                 [--max_dataset_size MAX_DATASET_SIZE] [--preprocess PREPROCESS] [--no_flip]
#                 [--display_winsize DISPLAY_WINSIZE] [--epoch EPOCH] [--load_iter LOAD_ITER]
#                 [--verbose] [--suffix SUFFIX] [--display_freq DISPLAY_FREQ]
#                 [--display_ncols DISPLAY_NCOLS] [--display_id DISPLAY_ID]
#                 [--display_server DISPLAY_SERVER] [--display_env DISPLAY_ENV]
#                 [--display_port DISPLAY_PORT] [--update_html_freq UPDATE_HTML_FREQ]
#                 [--print_freq PRINT_FREQ] [--no_html] [--save_latest_freq SAVE_LATEST_FREQ]
#                 [--save_epoch_freq SAVE_EPOCH_FREQ] [--save_by_iter] [--continue_train]
#                 [--epoch_count EPOCH_COUNT] [--phase PHASE] [--n_epochs N_EPOCHS]
#                 [--n_epochs_decay N_EPOCHS_DECAY] [--beta1 BETA1] [--lr LR] [--gan_mode GAN_MODE]
#                 [--pool_size POOL_SIZE] [--lr_policy LR_POLICY] [--lr_decay_iters LR_DECAY_ITERS]

# optional arguments:
#   -h, --help            show this help message and exit
#   --dataroot DATAROOT   path to images (should have subfolders trainA, trainB, valA, valB, etc)
#                         (default: None)
#   --name NAME           name of the experiment. It decides where to store samples and models
#                         (default: experiment_name)
#   --use_wandb           use wandb (default: False)
#   --gpu_ids GPU_IDS     gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU (default: 0)
#   --checkpoints_dir CHECKPOINTS_DIR
#                         models are saved here (default: ./checkpoints)
#   --model MODEL         chooses which model to use. [cycle_gan | pix2pix | test | colorization]
#                         (default: cycle_gan)
#   --input_nc INPUT_NC   # of input image channels: 3 for RGB and 1 for grayscale (default: 3)
#   --output_nc OUTPUT_NC
#                         # of output image channels: 3 for RGB and 1 for grayscale (default: 3)
#   --ngf NGF             # of gen filters in the last conv layer (default: 64)
#   --ndf NDF             # of discrim filters in the first conv layer (default: 64)
#   --netD NETD           specify discriminator architecture [basic | n_layers | pixel]. The basic
#                         model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the
#                         discriminator (default: basic)
#   --netG NETG           specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256
#                         | unet_128] (default: resnet_9blocks)
#   --n_layers_D N_LAYERS_D
#                         only used if netD==n_layers (default: 3)
#   --norm NORM           instance normalization or batch normalization [instance | batch | none]
#                         (default: instance)
#   --init_type INIT_TYPE
#                         network initialization [normal | xavier | kaiming | orthogonal] (default:
#                         normal)
#   --init_gain INIT_GAIN
#                         scaling factor for normal, xavier and orthogonal. (default: 0.02)
#   --no_dropout          no dropout for the generator (default: False)
#   --dataset_mode DATASET_MODE
#                         chooses how datasets are loaded. [unaligned | aligned | single |
#                         colorization] (default: unaligned)
#   --direction DIRECTION
#                         AtoB or BtoA (default: AtoB)
#   --serial_batches      if true, takes images in order to make batches, otherwise takes them
#                         randomly (default: False)
#   --num_threads NUM_THREADS
#                         # threads for loading data (default: 4)
#   --batch_size BATCH_SIZE
#                         input batch size (default: 1)
#   --load_size LOAD_SIZE
#                         scale images to this size (default: 286)
#   --crop_size CROP_SIZE
#                         then crop to this size (default: 256)
#   --max_dataset_size MAX_DATASET_SIZE
#                         Maximum number of samples allowed per dataset. If the dataset directory
#                         contains more than max_dataset_size, only a subset is loaded. (default:
#                         inf)
#   --preprocess PREPROCESS
#                         scaling and cropping of images at load time [resize_and_crop | crop |
#                         scale_width | scale_width_and_crop | none] (default: resize_and_crop)
#   --no_flip             if specified, do not flip the images for data augmentation (default: False)
#   --display_winsize DISPLAY_WINSIZE
#                         display window size for both visdom and HTML (default: 256)
#   --epoch EPOCH         which epoch to load? set to latest to use latest cached model (default:
#                         latest)
#   --load_iter LOAD_ITER
#                         which iteration to load? if load_iter > 0, the code will load models by
#                         iter_[load_iter]; otherwise, the code will load models by [epoch] (default:
#                         0)
#   --verbose             if specified, print more debugging information (default: False)
#   --suffix SUFFIX       customized suffix: opt.name = opt.name + suffix: e.g.,
#                         {model}_{netG}_size{load_size} (default: )
#   --display_freq DISPLAY_FREQ
#                         frequency of showing training results on screen (default: 400)
#   --display_ncols DISPLAY_NCOLS
#                         if positive, display all images in a single visdom web panel with certain
#                         number of images per row. (default: 4)
#   --display_id DISPLAY_ID
#                         window id of the web display (default: 1)
#   --display_server DISPLAY_SERVER
#                         visdom server of the web display (default: http://localhost)
#   --display_env DISPLAY_ENV
#                         visdom display environment name (default is "main") (default: main)
#   --display_port DISPLAY_PORT
#                         visdom port of the web display (default: 8097)
#   --update_html_freq UPDATE_HTML_FREQ
#                         frequency of saving training results to html (default: 1000)
#   --print_freq PRINT_FREQ
#                         frequency of showing training results on console (default: 100)
#   --no_html             do not save intermediate training results to
#                         [opt.checkpoints_dir]/[opt.name]/web/ (default: False)
#   --save_latest_freq SAVE_LATEST_FREQ
#                         frequency of saving the latest results (default: 5000)
#   --save_epoch_freq SAVE_EPOCH_FREQ
#                         frequency of saving checkpoints at the end of epochs (default: 5)
#   --save_by_iter        whether saves model by iteration (default: False)
#   --continue_train      continue training: load the latest model (default: False)
#   --epoch_count EPOCH_COUNT
#                         the starting epoch count, we save the model by <epoch_count>,
#                         <epoch_count>+<save_latest_freq>, ... (default: 1)
#   --phase PHASE         train, val, test, etc (default: train)
#   --n_epochs N_EPOCHS   number of epochs with the initial learning rate (default: 100)
#   --n_epochs_decay N_EPOCHS_DECAY
#                         number of epochs to linearly decay learning rate to zero (default: 100)
#   --beta1 BETA1         momentum term of adam (default: 0.5)
#   --lr LR               initial learning rate for adam (default: 0.0002)
#   --gan_mode GAN_MODE   the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is
#                         the cross-entropy objective used in the original GAN paper. (default:
#                         lsgan)
#   --pool_size POOL_SIZE
#                         the size of image buffer that stores previously generated images (default:
#                         50)
#   --lr_policy LR_POLICY
#                         learning rate policy. [linear | step | plateau | cosine] (default: linear)
#   --lr_decay_iters LR_DECAY_ITERS
#                         multiply by a gamma every lr_decay_iters iterations (default: 50)

