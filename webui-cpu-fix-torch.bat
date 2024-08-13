@echo off

set PYTHON=
set PIP=

pip.exe uninstall torch torchvision -y
pip.exe install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

pause