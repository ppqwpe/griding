@echo off
rem Griding without the graphics card - use this if Griding.bat freezes or crashes on this PC.
rem Keep this file ASCII-only: cmd.exe misreads non-ASCII text in .bat files.
set FACEGRID_CPU=1
call "%~dp0Griding.bat"
