@echo off
echo ======================================
echo Building Mojo CNN - CPU Version
echo ======================================

where cl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: cl.exe not found in PATH!
    echo Please run this script from the "x64 Native Tools Command Prompt for VS 2019/2022"
    pause
    exit /b 1
)

cd examples
cl /O2 /openmp /D MOJO_AVX /EHsc /I "../mojo" train_mnist.cpp /Fe:train_mnist_cpu.exe
if %errorlevel% neq 0 exit /b %errorlevel%

echo Build Success!
train_mnist_cpu.exe
