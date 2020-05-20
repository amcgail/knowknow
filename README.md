This repository contains 
  the code and data I used to analyze the deaths of a million citations in Sociology articles.
The repository provides code and documentation for producing all analyses and figures in the final paper.

# Installation

1. Install Python 3.7
2. Install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
3. Run `pip install science2-amcgail`

# Quick start

The following command starts jupyterlab in the base directory of this repository. This is a good place to start.

`python -m science2 start`

# Developing

There is a lot to do! If you find this useful to your work, and would like to contribute (even to the following list of possible next steps) please don't hesitate to reach out. My website is [here](http://www.alecmcgail.com), [Twitter here](https://twitter.com/SomeKindOfAlec). If you want to contribute edits of your own, fork this repository into your own GitHub account, make the changes, and submit a request for me to incorporate the code (a "pull request"). This process is really easy with GitHub Desktop ([tutorial here](https://www.youtube.com/watch?v=BYzriB5aTWU)).

## Aimed completion by 5/22/2020 (ben rosche)

## Aimed completion by 5/29/2020 (committee)

## Aimed completion by 6/5/2020 (presentation)

+ Externalizing data from the Git repository, so it can be dynamically downloaded / uploaded via AWS

## Post-6/5/2020 (likely by others)

+ The documentation for this project can always be improved. This is typically through people reaching out to me when they have issues. Please [feel free](https://twitter.com/SomeKindOfAlec).
+ An object-oriented model for handling context would prevent the need for so much variable-passing between functions, reduce total code volume, and improve readability.
+ Different datasets and sources could be incorporated, if you have the need, in addition to JSTOR and WoS.
+ If you produce precomputed binaries and have an idea of how we could incorporate the sharing of these binaries within this library, please [DM me](https://twitter.com/SomeKindOfAlec) or something. That would be great.