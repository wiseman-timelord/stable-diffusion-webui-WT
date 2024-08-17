:: Script: Install-Requirements.bat
@echo off
setlocal enabledelayedexpansion

:: Check for administrative privileges
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Error: Admin Required!
    echo Right Click, then Run As Administrator.
    timeout /t 3 >nul
    goto :end_of_file
)

:: Initialization
set "working_directory_location=%~dp0"
set "working_directory_location=%working_directory_location:~0,-1%"
pushd "%working_directory_location%"
echo Working Dir: "%working_directory_location%"



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




%PIP_EXE_TO_USE% install -r requirements.txt
pause