from .env import GLOBS, cfile

if __name__ == '__main__':
    import sys
    from pathlib import Path

    args = sys.argv[1:]

    me = Path(__file__).parent

    if not len(args):
        raise Exception("You need to specify a command: `init`, `clone`, `start`")

    if args[0] == 'init':

        import yaml
        from os.path import expanduser
        home = expanduser("~")
        default = Path(home).joinpath("knowknow")

        while 1:
            chosen_dir = input("Enter the directory where knowknow will keep data and code (will be created if doesn't exist) <default: %s> : " % default) or str(default)
            chosen_dir = Path(chosen_dir)
            if not chosen_dir.parent.exists():
                ans = None
                while ans not in "ynYN":
                    ans = input("The parent of this directory does not exist. Continue (y/n)?")
                if ans in "nN":
                    continue

            GLOBS['kkdir'] = str(chosen_dir)

            with cfile.open('w') as f:
                yaml.dump(GLOBS, f)

            if not chosen_dir.exists():
                print("Creating directory.")
                chosen_dir.mkdir(parents=True)
            else:
                print("This directory exists.")
            break

        print("Dataset directory updated")

    elif args[0] == 'clone':

        if len(args) != 2:
            raise Exception('`clone` command takes exactly 1 argument')


        if 'kkdir' not in GLOBS:
            raise Exception('You need to call `python -m knowknow init` first...')

        _, name = args

        if name[0] == '@':
            name = name[1:]

        if not( 'github.com' in name ):
            name = "https://github.com/%s" % name

        short_name = name.split("/")[-1]

        print("Cloning '%s' into '<kk>/code/%s' ..." % (name, short_name))

        from git.repo.base import Repo
        fld = Path(GLOBS['kkdir'], 'code')
        if not fld.exists():
            fld.mkdir()
        Repo.clone_from(name, fld.joinpath(short_name))

    elif args[0] == 'start':
        if len(args) != 2:
            fs = list(Path(GLOBS['kkdir'], 'code').glob("*"))
            if not len(fs):
                poss_args = "No code has been loaded, so there aren't valid commands. Either run `init` or `clone` to create/copy some data."
            else:
                poss_args = "Currently loaded: %s" %  ', '.join("`%s`"%x.name for x in fs)
            raise Exception('`start` command takes exactly 1 argument... %s' % poss_args)

        where = args[1]
        where = where.split("/")[-1]

        dr = Path(GLOBS['kkdir'], 'code', where)

        import os
        os.chdir(dr)
        os.environ['NBDIR'] = str(dr)
        os.system('jupyter-lab')

    elif args[0] == 'help':
        print("Possible commands are clone (with GitHub url), init (no args), and start (no args)")

    else:
        print("Command not recognized.")
        print("Possible commands are `clone` (with GitHub url), `init` (no args), and `start` (code directory)")