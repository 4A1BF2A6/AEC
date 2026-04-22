@echo off
setlocal

set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
    echo ERROR: Visual Studio BuildTools not found at %VCVARS%
    exit /b 1
)

call %VCVARS% > nul 2>&1

echo Compiling aec_test.exe (synthetic test)...
cl.exe /nologo /std:c11 /O2 /W3 /utf-8 /I. /Iooura_fft /MT ^
    aec_v2.c ooura_fft\ooura_fft.c ooura_fft\ooura_fft_mips.c test_main.c ^
    /Fe:aec_test.exe
if %ERRORLEVEL% neq 0 ( echo Build failed & exit /b 1 )

echo Compiling aec_wav.exe (WAV file processor)...
cl.exe /nologo /std:c11 /O2 /W3 /utf-8 /I. /Iooura_fft /MT ^
    aec_v2.c ooura_fft\ooura_fft.c ooura_fft\ooura_fft_mips.c test_real.c ^
    /Fe:aec_wav.exe
if %ERRORLEVEL% neq 0 ( echo Build failed & exit /b 1 )

echo Compiling aec_rt.exe (real-time PortAudio)...
cl.exe /nologo /std:c11 /O2 /W3 /utf-8 /I. /Iooura_fft /IC:\vcpkg\installed\x64-windows\include /MT ^
    aec_v2.c ooura_fft\ooura_fft.c ooura_fft\ooura_fft_mips.c main_realtime.c ^
    /Fe:aec_rt.exe ^
    /link C:\vcpkg\installed\x64-windows\lib\portaudio.lib ^
          winmm.lib ole32.lib uuid.lib setupapi.lib avrt.lib
if %ERRORLEVEL% neq 0 ( echo Build failed & exit /b 1 )

echo.
echo Build successful: aec_test.exe  aec_wav.exe  aec_rt.exe
echo.
echo Running synthetic test...
aec_test.exe

endlocal
