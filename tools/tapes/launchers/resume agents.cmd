@echo off
title Ito sprint 2 - resume agents
wsl.exe -d Ubuntu-26.04 -- bash /home/maarten/.unii/work/tonight/resume.sh
timeout /t 3 >nul
