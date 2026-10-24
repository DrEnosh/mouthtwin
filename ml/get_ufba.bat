@echo off
cd /d "%~dp0"
if not exist ufba mkdir ufba
cd ufba
echo Downloading UFBA-425 (CC BY 4.0, about 94 MB)...
curl.exe -L -o README.md https://ndownloader.figshare.com/files/56911730
curl.exe -L -o metadata.csv https://ndownloader.figshare.com/files/56911733
curl.exe -L -o segmentation_xrays.zip https://ndownloader.figshare.com/files/56911739
curl.exe -L -o polygon_masks.zip https://ndownloader.figshare.com/files/56911736
curl.exe -L -o numbering_xrays.zip https://ndownloader.figshare.com/files/56911727
for %%f in (segmentation_xrays polygon_masks numbering_xrays) do ..\.venv\Scripts\python.exe -m zipfile -e %%f.zip %%f
echo UFBA_DOWNLOADED > ..\status4.txt
echo Done. You can close this window.
pause
