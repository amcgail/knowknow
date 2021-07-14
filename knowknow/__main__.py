from .env import GLOBS, cfile
from git.repo.base import Repo

DOCUMENTATION = """
To get started with knowknow, you need to 
    1) specify where knowknow should store data and code ("init") 
    2) either create a new project or copy an existing one, and 
    3) start a JupyterLab environment. 

The following commands will help you perform these actions, getting you started conducting or reproducing analyses using `knowknow`.

`python -m knowknow init`. 
    Run this command first. 
    It will prompt you for the directory to store data files and the directory where code will be stored.

`python -m knowknow start <REPO-NAME>`
    For instance, `python -m knowknow start citation-death`. 
    Start a JupyterLab notebook in a repository. 
    If the repository does not exist, a new folder will be created with a git repository.

`python -m knowknow clone <URL>`
    For instance, `python -m knowknow clone https://github.com/amcgail/lost-forgotten`.
    Clone someone else's repository. 

Note: 

Data files will be automatically downloaded during code execution, if they are not alredy in the *data* directory you specified with the `init` command. This may take up significant bandwidth -- the data files for the Sociology dataset are ~750MB.

Code specified by the `knowknow.reqiure` function will be automatically downloaded by knowknow into the *code* directory you specified with the `init` command. **Be sure you trust whoever wrote the code you download.** Running arbitrary code from random strangers on your computer is a security risk.
"""

"""
# THE FOLLOWING HAS BEEN DESTROYED -- USE GITHUB DESKTOP

`python -m knowknow push <REPO-NAME> <git args>`
    For instance, `python -m knowknow push citation-death`. 
    This "pushes" any updates you make to the code to a remote GitHub repository.
    By default, it will push to the repository which was "cloned," 
        but advanced users can add arguments to the underlying `git push` command using <git args>.

`python -m knowknow pull <REPO-NAME> <git args>`
    For instance, `python -m knowknow push citation-death`. 
    This "pulls" any updates made to a remote GitHub repository to your local machine.
    By default, it will pull from the repository which was "cloned,"
        but advanced users can add arguments to the underlying `git push` command using <git args>.

"""

if __name__ == '__main__':
    import sys
    from pathlib import Path

    args = sys.argv[1:]

    me = Path(__file__).parent

    if not len(args):
        print( DOCUMENTATION )

    elif args[0] == 'init':

        import yaml
        from os.path import expanduser
        home = expanduser("~")
        
        

        def dir_for( name, GLOB_KEY, default_name ):

            default = Path(home).joinpath("knowknow", default_name)

            if GLOB_KEY in GLOBS:
                default = Path(GLOBS[GLOB_KEY])

            while 1:
                chosen_dir = input(f"Enter the directory where knowknow will keep {name} <default: {default}> (Ctrl+C to exit) : ") or str(default)
                chosen_dir = Path(chosen_dir)
                if not chosen_dir.parent.exists():
                    ans = None
                    while ans not in "ynYN":
                        ans = input("The parent of this directory does not exist. Continue (y/n)?")
                    if ans in "nN":
                        continue

                # create directory if not exists...
                if not chosen_dir.exists():
                    print("Creating directory.")
                    chosen_dir.mkdir(parents=True)
                else:
                    #print("This directory exists.")
                    pass

                # make the update
                GLOBS[GLOB_KEY] = str(chosen_dir)
                
                with cfile.open('w') as f:
                    yaml.dump(GLOBS, f)

                print(f"Dataset directory updated: {GLOB_KEY} = {str(chosen_dir)}")
                break


        dir_for( 'data', 'kk_data_dir', 'data' )
        dir_for( 'code', 'kk_code_dir', 'code' )

    elif args[0] == 'clone':

        if 'kk_code_dir' not in GLOBS:
            raise Exception('You need to call `python -m knowknow init` first...')

        if len(args) < 3:
            print(args)
            raise Exception('`clone` command takes exactly 2 arguments')

        _, where, what = args

        from .code_sharing import clone
        status, fn = clone(where, what)
        if not status:
            print(f"""A module with that name already exists at {fn}. 
Either rename or remove that directory to continue.""")

        """
        elif args[0] == 'pull':
            print("Unfortunately `pull` hasn't been implemented yet. Just use GitHub Desktop...")
            #Repo.pull(url, dest_fn)
            
        elif args[0] == 'push':

            #print("Unfortunately `push` hasn't been implemented yet. Just use GitHub Desktop...")
            
            from git.repo.base import Repo

            fld = Path(GLOBS['kk_code_dir'])
            if not fld.exists():
                fld.mkdir()

            message = args[2]
            
            gtfld = Path(GLOBS['kk_code_dir'], args[1])
            if not gtfld.exists():
                print("No such folder exists in the knowknow code directory.")
            else:
                r = Repo( gtfld )
                
                r.index.add([str(fn) for fn in gtfld.glob("**/*") if (Path('.git') not in list(fn.parents) and fn.name != '.git')])
                r.index.commit(message)
                r.remotes.origin.push()

            # currently I get "fatal: the remote end hung up unexpectedly'" every time. idk why
        """        

    elif args[0] == 'start':
        if 'kk_code_dir' not in GLOBS:
            raise Exception('You need to call `python -m knowknow init` first...')

        if len(args) != 2:
            fs = list(Path(GLOBS['kk_code_dir']).glob("*"))
            if not len(fs):
                poss_args = "No code has been loaded, so there aren't valid commands. Either run `init` or `clone` to create/copy some data."
            else:
                poss_args = "Currently loaded: %s" %  ', '.join("`%s`"%x.name for x in fs)
            raise Exception('`start` command takes exactly 1 argument... %s' % poss_args)

        where = args[1]
        where = where.split("/")[-1]

        dr = Path(GLOBS['kk_code_dir'], where)

        import os
        os.chdir(dr)
        os.environ['NBDIR'] = str(dr)
        os.system('jupyter-lab')

    elif args[0] == 'help':
        print(DOCUMENTATION)

    else:
        print("Command not recognized.")
        print(DOCUMENTATION)