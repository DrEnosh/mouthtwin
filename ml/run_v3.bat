@echo off
REM MouthTwin round 3: more varied adult OPGs (STS-2D-Tooth, AKU). Needs round 2 finished (runs2\best.pt).
REM Safe to run again: finished steps are skipped or resumed. Look at status3.txt and the *3.log files.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
echo Installing packages... > status3.txt
%PY% -m pip install -q -c constraints.txt pyarrow > pip3.log 2>&1
if not exist dentex\proc3\aku\meta.json (
  %PY% -m zipfile -e aku_part0.zip . > unzip3.log 2>&1
  %PY% -m zipfile -e aku_part1.zip . >> unzip3.log 2>&1
  %PY% -m zipfile -e aku_part2.zip . >> unzip3.log 2>&1
)
echo Downloading STS OPGs... > status3.txt
if exist dentex\proc3\meta3.json goto prepdone
%PY% -X faulthandler -u prepare3.py . > prep3.log 2>&1
if errorlevel 1 goto fail
:prepdone
echo Teacher labelling... > status3.txt
%PY% -X faulthandler -u pseudo3.py > pseudo3.log 2>&1
if errorlevel 1 goto fail
echo Training... > status3.txt
set MT_BS=6
set MT_STEPS=500
%PY% -X faulthandler -u train3.py 30 > train3.log 2>&1
if errorlevel 1 goto fail
echo Exporting... > status3.txt
%PY% -X faulthandler -u export3.py > export3.log 2>&1
if errorlevel 1 goto fail
echo ALL_DONE > status3.txt
echo.
echo Round 3 finished. You can close this window.
pause
exit /b 0
:fail
echo FAILED see logs > status3.txt
echo A step failed. See the log files in this folder.
pause
