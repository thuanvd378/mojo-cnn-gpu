@echo off
REM Run script for train_mnist_cpu.exe

cd /d "%~dp0"
cd examples

echo Starting MNIST CPU Training...
echo.

rem Default values
set BATCH_SIZE=24
set LEARN_RATE=0.04
set EPOCHS=10

rem Override with command line arguments if provided
if not "%1"=="" set BATCH_SIZE=%1
if not "%2"=="" set LEARN_RATE=%2
if not "%3"=="" set EPOCHS=%3

echo Running with Batch Size: %BATCH_SIZE%, Learning Rate: %LEARN_RATE%, Epochs: %EPOCHS%
train_mnist_cpu.exe %BATCH_SIZE% %LEARN_RATE% %EPOCHS%

pause
