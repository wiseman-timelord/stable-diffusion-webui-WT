@echo off

:: Global Variables
set PIP_EXE_TO_USE="C:\Program Files\Python310\Scripts\pip.exe"
set ATTEMPTS=0

:check_pip
:: Check if pip exists at the specified location
if not exist %PIP_EXE_TO_USE% (
    echo Pip not found at %PIP_EXE_TO_USE%
    set /p PIP_EXE_TO_USE="Please enter the full path to pip.exe (including pip.exe): "
    
    :: Check if the user-provided path exists
    if not exist %PIP_EXE_TO_USE% (
        set /a ATTEMPTS+=1
        if %ATTEMPTS% lss 3 (
            echo The provided path does not exist. Attempt %ATTEMPTS% of 3. Please try again.
            goto check_pip
        ) else (
            echo Maximum attempts reached. Exiting script.
            pause
            exit /b 1
        )
    )
)

:: Reset attempts if a valid path is found
set ATTEMPTS=0

:: Use the specified pip executable
%PIP_EXE_TO_USE% uninstall torch torchvision torchaudio -y
%PIP_EXE_TO_USE% install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

:: End Of Script
pause
