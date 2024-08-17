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

pip install -r requirements.txt
pause