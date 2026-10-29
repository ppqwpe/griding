@echo off
rem Griding - double-click to start. Everything it needs is inside the app folder.
rem Keep this file ASCII-only: cmd.exe misreads non-ASCII text in .bat files.
title Griding
if not exist "%~dp0app\python\python.exe" (
  echo.
  echo   Griding files are missing next to this file.
  echo   If you opened it from inside a zip, extract the whole zip first
  echo   ^(right-click - Extract All^), then run Griding.bat in the extracted folder.
  echo.
  pause
  exit /b 1
)
"%~dp0app\python\python.exe" -u -X utf8 "%~dp0app\server.py"
if errorlevel 1 pause
