@echo off
set PYTHON=D:\arte\work\astro-kahn-it-com\000.cosy-talk\ComfyUI_windows_portable\python_embeded\python.exe
%PYTHON% -m pip install chatterbox-tts soundfile --prefer-binary
%PYTHON% generate.py
pause
