@echo off
REM MouthTwin round 2: harder OPGs + restoration detection. Needs round 1 finished (runs\best.pt).
REM Safe to run again: finished steps are skipped or resumed. Look at status2.txt and the *.log files.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
echo Installing packages... > status2.txt
%PY% -m pip install -q -c constraints.txt scipy remotezip pillow numpy matplotlib > pip2.log 2>&1
if not exist runs\teacher.pt copy runs\best.pt runs\teacher.pt
echo Downloading extra OPGs... > status2.txt
%PY% -X faulthandler -u prepare_more.py . > prep2.log 2>&1
if errorlevel 1 goto fail
echo Teacher labelling... > status2.txt
%PY% -X faulthandler -u pseudo_label.py > pseudo.log 2>&1
if errorlevel 1 goto fail
echo Training... > status2.txt
%PY% -X faulthandler -u train2.py 40 > train2.log 2>&1
if errorlevel 1 goto fail
echo Exporting... > status2.txt
%PY% -X faulthandler -u export2.py > export2.log 2>&1
if errorlevel 1 goto fail
echo ALL_DONE > status2.txt
echo.
echo Round 2 finished. You can close this window.
pause
exit /b 0
:fail
echo FAILED see logs > status2.txt
echo A step failed. See the log files in this folder.
pause
