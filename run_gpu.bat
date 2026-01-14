@echo off
REM Run script for train_mnist_gpu.exe
REM Sets up DLL paths before running

cd /d "%~dp0"
cd examples

set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;E:\NVIDIA\CUDNN\v9.17\bin\13.1;%PATH%"

echo Starting MNIST GPU Training...
echo.
rem Default values (can be changed here)
set BATCH_SIZE=4096
set LEARN_RATE=0.005
set EPOCHS=500

rem Override with command line arguments if provided
if not "%1"=="" set BATCH_SIZE=%1
if not "%2"=="" set LEARN_RATE=%2
if not "%3"=="" set EPOCHS=%3

echo Running with Batch Size: %BATCH_SIZE%, Learning Rate: %LEARN_RATE%, Epochs: %EPOCHS%
train_mnist_gpu.exe %BATCH_SIZE% %LEARN_RATE% %EPOCHS%

pause
