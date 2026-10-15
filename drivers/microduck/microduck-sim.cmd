@echo off
title Ito - Virtual Microduck
cd /d "%~dp0"
rem The default WSL distro, set up with drivers/microduck/setup-sim.sh ~/.local/share/ito-microduck
wsl.exe --cd ~ --exec .local/share/ito-microduck/microduck-sim --viewer %*
set "sim_result=%ERRORLEVEL%"
if not "%sim_result%"=="0" pause
exit /b %sim_result%
