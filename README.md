# Stable Diffusion web UI - Wiseman-Timelord
Status: Beta; The hacks work, but are experimental and quick-fixes.

### Description:
Wiseman-Timelords Hacks for CPU ONLY Stable Diffusion v1.10 Setups. I found Stable Diffusion Webui was created almost exclusively for nVidia users, and I say that because the thread usage was always balls low, going by the default of llama.cpp, this is probably 4 threads, even after specifying `-use-cpu all`, possibly that is ALL cpus, NOT all threads. People whom choose AMD hardware, probably have an AMD CPU with a high number of threads, and these were just not being put to use, and so, a pure quality of life mod, to save people time. So it fixes some things for sd-webui version ~1.10 (2024\08\13). And if I can figure out optimizations specific to cpu, and implementthen I may wrap the whole thing up as a optimizer/patcher for StableDiffusionWebUi/Torch, for CPU Only, kinda thing. After that, where do we go? Aocl, Avx2, Vulkan, OpenCL? Who knows, dont care, its progress so long as it goes faster, and dont have errors.

## Features:
- Work So Far...
1. Enabled multi-core for torch, it will use 85% of available threads, set in global at top of "devices.py".
2. batch to remove, torch, torchvision, torchaudio, then installs, torch+cpu, torchvision+cpu, torchaudio+cpu.
3. figured out the argument `COMMANDLINE_ARGS=--use-cpu all --no-half --skip-torch-cuda-test`.
- Work Intended...
1. It will next be building the version of pytorch for the machine, though, depending upon the complexity, it will likely be AVX2 only, but hopefully auto whatever the machine config is.
2. Potentially Aocl optimizations, we would definately be AMD only by then.
3. Potentially Vulkan build of pytorch, but would that then be re-implementing the devices, and hardcoding/replacing the cuda code with vulkan, and not having cpu. 

### Preview:
- No issues there, and look it shows how many threads its using...
```
venv "D:\ProgsOthers\StableDiffusion-WebUI\stable-diffusion-webui-bad\venv\Scripts\Python.exe"
Python 3.10.6 (tags/v3.10.6:9c7b4bd, Aug  1 2022, 21:53:49) [MSC v.1932 64 bit (AMD64)]
Version: v1.10.1
Commit hash: 82a973c04367123ae98bd9abdf80d9eda9b910e2
Launching Web UI with arguments: --use-cpu all --no-half --skip-torch-cuda-test --skip-load-model-at-start --api --port 7860
(85%) of total threads = 20 out of 24
no module 'xformers'. Processing without...
no module 'xformers'. Processing without...
No module 'xformers'. Proceeding without it.
Warning: caught exception 'Torch not compiled with CUDA enabled', memory monitor disabled
ControlNet preprocessor location: D:\ProgsOthers\StableDiffusion-WebUI\stable-diffusion-webui-bad\extensions\sd-webui-controlnet\annotator\downloads
2024-08-13 21:18:42,076 - ControlNet - INFO - ControlNet v1.1.455
2024-08-13 21:18:42,449 - ControlNet - INFO - ControlNet UI callback registered.
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
Startup time: 8.3s (prepare environment: 0.3s, import torch: 2.9s, import gradio: 0.8s, setup paths: 0.9s, initialize shared: 0.1s, other imports: 0.6s, load scripts: 1.4s, create ui: 0.6s, gradio launch: 0.4s, add APIs: 0.4s).
Loading weights [6ce0161689] from D:\ProgsOthers\StableDiffusion-WebUI\stable-diffusion-webui-bad\models\Stable-diffusion\v1-5-pruned-emaonly.safetensors
Creating model from config: D:\ProgsOthers\StableDiffusion-WebUI\stable-diffusion-webui-bad\configs\v1-inference.yaml
C:\Progra~1\Python310\lib\site-packages\huggingface_hub\file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Applying attention optimization: InvokeAI... done.
Model loaded in 2.8s (load weights from disk: 0.3s, create model: 0.8s, apply weights to model: 1.6s).
```

## Installation and Running
This fork will be windows ONLY, as I cant test anything else. Instructions are currently...
1. install "StableDiffusion-WebUi", and its requirements.txt, and ensure the models are in the models folder appropriately.
2. run the batch `install-torch-cpu.bat` to remove the non-cpu versions of torch and torchvision, and instead install compatible torch cpu and torchvision cpu versions; alternatively run `pip.exe uninstall torch torchvision torchaudio -y`, then `pip.exe install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu`. 
3. replace "...\StableDiffusion-Webui\modules\devices.py" with the "devices.py", supplied.
4. The arguments in "webui-user.bat" i suggest `COMMANDLINE_ARGS=--use-cpu all --no-half --skip-torch-cuda-test`.
5. run as normal, and ignore any additional errors, if errors when loading model, try load other, then one you wanted again, its a bit iffy sometimes, but you will notice, that the cpu usage is now blowing guages when you generate your images, this has to be progress worth sharing.

### Notes:
- If you are an AI programmer, backup the py files provided, and feed them into GPT, ask for 1 improvement/optimization at a time towards your specific processor, and then test, fall back to working versions, start simple.
- Torch could theoretically be made opencl/vulkan, and if I can use vulkan for torch, then it would be possible to use vulkan for amd and nvidia, and streamlines the scripts. Just have options Cpu and Vulkan, got to test it first.

## Credits
- Stable Diffusion - https://github.com/Stability-AI/stablediffusion, https://github.com/CompVis/taming-transformers, https://github.com/mcmonkey4eva/sd3-ref
