set -ex
python exr_to_depth.py \
    --input_img_directory datasets/img_data/exr \
    --output_img_directory datasets/img_data/trainA \


# usage: exr_to_depth.py [-h] [--input_img_directory INPUT_IMG_DIRECTORY]
#                        [--output_img_directory OUTPUT_IMG_DIRECTORY]

# exr to depth program

# optional arguments:
#   -h, --help            show this help message and exit
#   --input_img_directory INPUT_IMG_DIRECTORY
#                         input directory
#   --output_img_directory OUTPUT_IMG_DIRECTORY
#                         output directory

