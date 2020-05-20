import sys
args = sys.argv[1:]

if not len(args):
    print("No command given. Assuming command 'python -m knowknow start'")
else:
    if args[0] != 'start':
        raise Exception("Command not valid. Use 'python -m knowknow start'")
        
import os
import knowknow
os.chdir(knowknow.BASEDIR)
os.system("jupyter lab")