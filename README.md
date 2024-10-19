instructions to run:

go to eTraM/rvt_eTram and run the following:

	conda create -y -n rvt python=3.9 pip
	conda activate rvt
	conda config --set channel_priority flexible
	

if you have an nvidia gpu, run the following:
	CUDA_VERSION=11.8
    conda install -y h5py=3.8.0 blosc-hdf5-plugin=1.0.0 \
hydra-core=1.3.2 einops=0.6.0 torchdata=0.6.0 tqdm numba \
pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=$CUDA_VERSION \
-c pytorch -c nvidia -c conda-forge

otherwise:
    conda install -y h5py=3.8.0 blosc-hdf5-plugin=1.0.0 \
hydra-core=1.3.2 einops=0.6.0 torchdata=0.6.0 tqdm numba \
pytorch=2.0.0 torchvision=0.15.0 \
-c pytorch -c conda-forge


and then run the following to wrap up installations:

    python -m pip install pytorch-lightning==1.8.6 wandb==0.14.0 \
pandas==1.5.3 plotly==5.13.1 opencv-python==4.6.0.66 tabulate==0.9.0 \
pycocotools==2.0.6 bbox-visualizer==0.1.0 StrEnum==0.4.10


in that directory, open up inference.py and change the following line:

    initialize(config_path="eTraM/rvt_eTram/config/model", version_base=None)

to:
    initialize(config_path="config/model", version_base=None)
	
you will need to change this path back later, but this will help make sure you
can get the model running on its own.

Once you've changed that line, you need to download the pretrained model weights from:

https://arizonastateu-my.sharepoint.com/:u:/g/personal/averma90_sundevils_asu_edu/EWSl7Y3riQZJjQ7gEfpFt2EBfXktubreHoudWmTzcRXRWA?e=A6MsVn

and update the 'checkpoint' config/model/rnndet.yaml to be the path of your weights.

now try running the inference script:

python inference.py

once you get predictions, go back to inference.py and change this line:

    initialize(config_path="config/model", version_base=None)

back to 

    initialize(config_path="eTraM/rvt_eTram/config/model", version_base=None)


go back to the project root:

    cd ../..

and to link rust/python, run:
    pip install maturin
    maturin develop

you may need to resolve linking issues here, specifically ensuring that the linker knows
to prioritize your conda environment.

then you can run the project:
    cargo run
