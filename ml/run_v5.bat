@echo off
REM MouthTwin round 5: Humans-in-the-Loop + Khurshid restoration polygons + DentalOPG-1550 + Kairouan. Needs runs4b\best.pt.
REM Safe to rerun: prep is skipped once dentex\proc5\meta5.json exists.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
echo Preparing round-5 data... > status5.txt
if exist dentex\proc5\meta5.json goto prepdone
%PY% -X faulthandler -u prepare5.py > prep5.log 2>&1
if errorlevel 1 goto fail
:prepdone
echo Training 5... > status5.txt
set MT_STS_U=2650
set MT_BS=6
set MT_STEPS=500
%PY% -X faulthandler -u train5.py 24 > train5.log 2>&1
if errorlevel 1 goto fail
echo Exporting 5... > status5.txt
%PY% -X faulthandler -u export5.py > export5.log 2>&1
if errorlevel 1 goto fail
echo ALL_DONE > status5.txt
exit /b 0
:fail
echo FAILED see logs > status5.txt
exit /b 1
