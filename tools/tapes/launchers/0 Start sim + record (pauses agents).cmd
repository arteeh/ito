@echo off
title Ito sprint 2 - pause agents, start the sim with recording on
wsl.exe -d Ubuntu-26.04 -- bash /home/maarten/.unii/work/tonight/pause.sh
if errorlevel 1 (pause & exit /b 1)
wsl.exe -d Ubuntu-26.04 --cd ~ -- bash /mnt/c/Users/me/Projects/ito/sprint2/sim-record.sh
set "sim_result=%ERRORLEVEL%"
if not "%sim_result%"=="0" pause
exit /b %sim_result%
