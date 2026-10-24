@echo off
REM MouthTwin round 4: UFBA-425 (CC BY 4.0) + all STS OPGs. Needs round 3 (runs3\best.pt). Safe to rerun: finished steps are skipped.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
echo Preparing UFBA... > status4.txt
if exist dentex\proc4\ufba\meta.json goto prepdone
%PY% -X faulthandler -u prepare_ufba.py > prep4.log 2>&1
if errorlevel 1 goto fail
:prepdone
echo Teacher labelling the extra STS OPGs... > status4.txt
set MT_STS_U=2650
%PY% -X faulthandler -u pseudo3.py > pseudo4.log 2>&1
if errorlevel 1 goto fail
echo Training... > status4.txt
set MT_BS=6
set MT_STEPS=500
%PY% -X faulthandler -u train4.py 20 > train4.log 2>&1
if errorlevel 1 goto fail
echo Exporting... > status4.txt
%PY% -X faulthandler -u export4.py > export4.log 2>&1
if errorlevel 1 goto fail
echo ALL_DONE > status4.txt
echo Round 4 finished. You can close this window.
pause
exit /b 0
:fail
echo FAILED see logs > status4.txt
echo A step failed. See the log files in this folder.
pause
