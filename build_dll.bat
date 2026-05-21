@echo off
setlocal
cd /d %~dp0

set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
    echo ERROR: Visual Studio BuildTools not found at %VCVARS%
    exit /b 1
)

call %VCVARS% > nul 2>&1

if not exist build mkdir build

echo Compiling aec_v2.dll (shared library for Python ctypes)...
rem 中间件 (.obj) 写到 build\，DLL 留在顶层；.lib/.exp 也走 build\
cl.exe /nologo /std:c11 /O2 /W3 /utf-8 /I. /Iooura_fft /MT /LD ^
    /Fobuild\ ^
    aec_v2.c ooura_fft\ooura_fft.c ooura_fft\ooura_fft_mips.c ^
    /Fe:aec_v2.dll ^
    /link /DEF:aec_v2.def /IMPLIB:build\aec_v2.lib
if %ERRORLEVEL% neq 0 ( echo Build failed & exit /b 1 )

echo.
echo Build successful: aec_v2.dll  (intermediates -^> build\)
dir /b aec_v2.dll
dir /b build\aec_v2.lib build\aec_v2.exp 2>nul

endlocal
