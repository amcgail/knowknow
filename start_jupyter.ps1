$env:VENV = "C:/Users/amcga/envs/citation-deaths"
$env:KKDIR = "G:\My Drive\2020 ORGANISATION\1. PROJECTS\qualitative analysis of literature\110 CITATION ANALYSIS\010 analyses\bundle 100 - knowknow reboot 10-2020"
$env:VARDIR = "C:\Users\amcga\Desktop\knowknow_variables"
$env:NBDIR = $MyInvocation.MyCommand.Path



$env:PYTHONPATH = "$env:KKDIR;$env:PYTHONPATH"
& "$env:VENV/Scripts/Activate.ps1"
# pip install -r "$env:KKDIR/requirements.txt"

jupyter-lab