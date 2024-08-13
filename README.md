# Stable Diffusion web UI - Wiseman-Timelord
Wiseman-Timelords Hacks for CPU ONLY Stable Diffusion v1.10 Setups.

### Description
- I found Stable Diffusion Webui was created almost exclusively for nVidia users, and I say that because the thread usage was always balls low, making the issue worse was that multiple warnings pop up when using cpu. I think it was probably using 1 or something like 4, or maybe just half before, and there is no way to specify how many threads. People whom choose AMD hardware, probably have an AMD CPU with a high number of threads, and these were just not being put to use, and so, a pure quality of life mod, to save people time.

## Features
- Work So Far...
1. Stopped some Cuda warnings by streamlining "autocast_mode.py".
2. Enabled multi-core for torch, it will automatically use 80% of the threads.
- Work Intended...
1. Avx2 and/or Aocl, specific code.
2. User friendly installer/patcher, that, ensures torch/torchvision cpu are installed, then searches for files in possibly locations and patches.

## Installation and Running
This fork will be windows ONLY, as I cant test anything else. Instructions are currently...
1. install "StableDiffusion-WebUi", and its requirements.txt, and ensure the models are in the models folder appropriately.
2. run the batch to remove the non-cpu versions of torch and torchvision, and instead install torch cpu and torchvision cpu versions compatible with Stable diffusion 1.10
3. replace "YourDrive:\**ParentFolders**\Python310\Lib\site-packages\torch\cpu\amp\autocast_mode.py" with the "autocast_mode.py" supplied.
4. run as normal, and ignore any additional errors, if errors when loading model, try load other, then one you wanted again, its a bit iffy sometimes.

## Credits
- Stable Diffusion - https://github.com/Stability-AI/stablediffusion, https://github.com/CompVis/taming-transformers, https://github.com/mcmonkey4eva/sd3-ref
