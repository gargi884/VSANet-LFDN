# code

## Dependencies
- Python 3.10
- PyTorch 2.5.1
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## Create environment and install packages
- `conda create -n code python=3.10`
- `conda activate code`
- `pip install -r requirements.txt`

## Testing

- Download STFLytro train dataset from [GoogleDrive](https://drive.google.com/file/d/13YJlRqhNijItA61DPh2XzOFQf7v6f35O/view?usp=drive_link) 

- Download EPFL test dataset from [GoogleDrive](https://drive.google.com/file/d/1u0j1U6VFvxFvWWP2fNSmaJTMT5ILRi-X/view?usp=sharing) 

- Keep test datasets in location `datasets`

- Generate the denoised images on STFLytro dataset
`python test_STFLytro.py`


## Testing

- Download STFLytro test dataset from [GoogleDrive](https://drive.google.com/file/d/1KEb-Q-H-VCyk83fhwuR3zPicm7GXXV6I/view?usp=drive_link) 

- Download EPFL test dataset from [GoogleDrive](https://drive.google.com/file/d/1u0j1U6VFvxFvWWP2fNSmaJTMT5ILRi-X/view?usp=sharing) 

- Keep test datasets in location `datasets`

- Generate the denoised images on STFLytro dataset
`python test_STFLytro.py`

- Generate the denoised images on EPFL dataset
`python test_EPFL.py`

- The denoised images are in `results/`.
 
