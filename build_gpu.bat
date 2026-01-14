@echo off
REM Build script for Mojo-CNN GPU (cuDNN) version
echo ======================================
echo Building Mojo CNN - GPU (cuDNN) Version
echo ======================================

REM Check for CUDA
if not defined CUDA_PATH (
    echo ERROR: CUDA_PATH environment variable is not set!
    echo Please install CUDA Toolkit.
    pause
    exit /b 1
)

REM Set paths - EDIT THESE IF NEEDED
REM If you have CUDNN_PATH set in env vars, it will be used.
REM Otherwise, please set it here or in your environment.
if not defined CUDNN_PATH (
    echo WARNING: CUDNN_PATH not set. attempting to use CUDA_PATH...
    set "CUDNN_PATH=%CUDA_PATH%"
)

set "CUDNN_INCLUDE=%CUDNN_PATH%\include"
set "CUDNN_LIB=%CUDNN_PATH%\lib\x64"

REM Check if cuDNN exists
if not exist "%CUDNN_INCLUDE%\cudnn.h" (
    echo ERROR: cuDNN header not found at %CUDNN_INCLUDE%
    echo Please ensure CUDNN_PATH is set correctly to your cuDNN installation directory.
    echo Example: set CUDNN_PATH=E:\NVIDIA\CUDNN\v9.x
    pause
    exit /b 1
)

echo.
echo Setting up Visual Studio environment...
echo.

REM Verify cl.exe is accessible
where cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: cl.exe not found in PATH!
    echo Please run this script from the "x64 Native Tools Command Prompt for VS 2019/2022"
    pause
    exit /b 1
)

echo Found MSVC compiler: 
where cl
echo.
echo Found cuDNN at: %CUDNN_INCLUDE%
echo.
echo Compiling with nvcc + cuDNN...
echo.

REM Build the GPU training example
nvcc -x cu ^
  -I./mojo-gpu ^
  -I"%CUDNN_INCLUDE%" ^
  -I"%CUDA_PATH%\include" ^
  -L"%CUDNN_LIB%" ^
  -L"%CUDA_PATH%\lib\x64" ^
  -lcudnn -lcublas ^
  -O3 ^
  examples/train_mnist_gpu.cpp ^
  -o examples/train_mnist_gpu.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================
    echo Build SUCCESS!
    echo ======================================
    echo.
    echo Executable: examples\train_mnist_gpu.exe
    echo.
    echo To run:
    echo   examples\train_mnist_gpu.exe [batch_size] [learning_rate]
    echo   examples\train_mnist_gpu.exe 64 0.001
    echo.
) else (
    echo.
    echo ======================================
    echo Build FAILED!
    echo ======================================
    echo.
    echo Please check the errors above.
    pause
    exit /b 1
)
