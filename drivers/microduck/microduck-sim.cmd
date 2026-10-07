@echo off
title Ito - Virtual Microduck
cd /d "%~dp0"
wsl.exe -d ito-microduck -u root --cd /opt/ito -- /opt/ito/stack/microduck-sim --viewer
set "sim_result=%ERRORLEVEL%"
if not "%sim_result%"=="0" pause
exit /b %sim_result%
