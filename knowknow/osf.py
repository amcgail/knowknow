"""
NOTE: this is a failure...
I really have to manually upload new datasets, this command line bull isn't gonna work well
"""

from subprocess import Popen,PIPE

def run_command(command = 'dir'):
    pipe = Popen(command,stdout=PIPE,stderr=PIPE)    

    ret = []
    while True:         
        line = pipe.stdout.readline()
        if line:            
            ret.append(line.decode('utf8'))
        if not line:
            break

    return "".join(ret).replace("[\n\r]+","\n")

#run_command(["osf","ls"])