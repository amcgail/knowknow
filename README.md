This Python package, `knowknow`, is an attempt to make powerful, modern tools for analyzing the structure of knowledge open to anyone.
I recognize that parallel efforts exist along these lines, including [CADRE](https://cadre.iu.edu/), but this package is still the only resource for *anyone* to analyze Web of Science datasets, and the methods can be incorporated into CADRE by *anyone*.

<!--
I have included every inch of code here, leaving no stone unturned. With every `pip install knowknow-amcgail`, you download the following:

+ `creating variables`, a collection of pre-processing algorithms for cleaning and summarizing Web of Science search results, or JSTOR Data for Research data dumps.
+ `analyses`, a set of descriptive notebooks which illustrate these datasets
+ A connector to pre-computed cooccurrence sets, hosted on [OSF](https://osf.io/9vx4y/)
-->

# Projects built on knowknow

+ [amcgail/citation-death](https://github.com/amcgail/citation-death) applies the concept of 'death' to attributes of citations, and analyzes the lifecourse of cited works, cited authors, and the authors writing the citations, using the `sociology-wos-74b` dataset. 
+ [amcgail/lost-forgotten](https://github.com/amcgail/lost-forgotten) digs deeper into . An online appendix is available [here](www.alecmcgail.com/lost&forgotten/), and the paper published in *The American Sociologist* can be found [here](https://rdcu.be/cnSFG).

# Datasets built with knowknow

+ Sociology
    + `sociology-wos` ([Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GQGJLQ)) every paper in WoS in early 2020 whose journal is in the 'Sociology' category, and which have full data. 
    + *in progress* `sociology-jstor` in-text citations and their contexts were extracted from >90k full-text Sociology articles indexed in JSTOR. 

# Installation (from PyPI)

1. Install Python 3.7+
2. Install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
3. Run `pip install knowknow-amcgail`

# Installation (from GitHub)

1. Install Python 3.7+
2. Clone this repository to your computer
3. Create a virtualenv for `knowknow`
4. In the virtualenv, execute `pip install -r requirements`
    + On Windows, I needed to install the latest versions of `numpy`, `scikit-learn` and `scipy` via .whl
    + For Windows, download from [this site](https://www.lfd.uci.edu/~gohlke/pythonlibs/), install with `pip install <fn.whl>`

# Getting Started

To get started with knowknow, you need to 
    1) specify where knowknow should store data and code ("init") 
    2) either create a new project or copy an existing one, and 
    3) start a JupyterLab environment. 

The following commands will help you perform these actions, getting you started conducting or reproducing analyses using `knowknow`.

`python -m knowknow init`. 
    Run this command first. 
    It will prompt you for the directory to store data files and the directory where code will be stored.

`python -m knowknow start <PROJ-NAME>`
    For instance, `python -m knowknow start citation-death`. 
    Start a JupyterLab notebook in a knowknow code directory. 
    If the directory doesn't exist, knowknow creates the directory, and initiates it as a git repository.

# [Recommended] Interfacing with GitHub

In order to use the following commands you must install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
This allows you to use others' code, and to publish your own code for others to use.

`python -m knowknow clone <URL>`
    For instance, `python -m knowknow clone https://github.com/amcgail/lost-forgotten`.
    Clone someone else's repository. 

To make your own changes to others' code, or to share your code with the world, do the following:

1) Create a [GitHub](https://www.github.com/) account and log in.
2) Install [GitHub Desktop](https://desktop.github.com/), which is a simple connector between Git on your computer and GitHub, in the cloud.
3a) [Share your code] In GitHub Desktop, choose `File -> Create Repository`, navigate to the folder containing knowknow code. This folder was created by knowknow using the `start` command. Now press "Publish Repository" in the upper right to add this code to your GitHub account.
3b) [Contribute to others' code] In GitHub, `fork` the repository you would like to contribute to. This creates a personal copy of that repository in your GitHub account. Then clone this copy into knowknow's code directory using the `clone` command, or using GitHub desktop. Once you are satisfied with your updates, and they are pushed back to GitHub, submit a "pull request" to the original repository to ask them to review and merge your changes.

# Auto-downloading Data and Code

Data files will be automatically downloaded during code execution, if they are not alredy in the *data* directory you specified with the `init` command. This may take up significant bandwidth -- the data files for the Sociology dataset are ~750MB.

Code specified by the `knowknow.reqiure` function will be automatically downloaded by knowknow into the *code* directory you specified with the `init` command. **Be sure you trust whoever wrote the code you download.** Running arbitrary code from random strangers on your computer is a security risk.

# Developing

If you want to contribute edits of your own, fork this repository into your own GitHub account, make the changes, and submit a request for me to incorporate the code (a "pull request"). This process is really easy with GitHub Desktop ([tutorial here](https://www.youtube.com/watch?v=BYzriB5aTWU)).

There is a lot to do! If you find this useful to your work, and would like to contribute (even to the following list of possible next steps) but can't figure out how, please don't hesitate to reach out. My website is [here](http://www.alecmcgail.com), [Twitter here](https://twitter.com/SomeKindOfAlec). 

## Possible projects

+ The documentation for this project can always be improved. This is typically through people reaching out to me when they have issues. Please [feel free](https://twitter.com/SomeKindOfAlec).
+ **complete** An object-oriented model for handling context would prevent the need for so much variable-passing between functions, reduce total code volume, and improve readability.
+ *ongoing* Different datasets and sources could be incorporated, if you have the need, in addition to JSTOR and WoS.
+ **complete - you can now upload data files to Harvard's Dataverse** If you produce precomputed binaries and have an idea of how we could incorporate the sharing of these binaries within this library, please [DM me](https://twitter.com/SomeKindOfAlec) or something. That would be great.
+ *ongoing, future work* All analyses can be generalized to any counted variable of the citations. This wouldn't be tough, and would have a huge payout.
+ *huge project, uncertain payout* It would be amazing if we could make a graphical interface for this.
    + user simply imports data, chooses the analyses they want to run, fill in configuration parameters and press "go"
    + the output is a PDF with the code, visualizations, and explanations for a given analysis
    + behind the scenes, all this GUI does is run `nbconvert` 
    + also could allow users to regenerate any/all analyses for each dataset with the click of a button
    + could provide immediate access to online archives, either to download or upload similar count datasets
