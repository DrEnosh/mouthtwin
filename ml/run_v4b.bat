@echo off
REM MouthTwin round 4b: round-4 data, restorations protected (see train4b.py). Needs runs3\best.pt and round-4 prep done.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
echo Training 4b... > status4b.txt
set MT_STS_U=2650
set MT_BS=6
set MT_STEPS=500
%PY% -X faulthandler -u train4b.py 18 > train4b.log 2>&1
if errorlevel 1 goto fail
echo Exporting 4b... > status4b.txt
%PY% -X faulthandler -u export4b.py > export4b.log 2>&1
if errorlevel 1 goto fail
echo ALL_DONE > status4b.txt
echo Round 4b finished. You can close this window.
pause
exit /b 0
:fail
echo FAILED see logs > status4b.txt
echo A step failed. See the log files in this folder.
pause
