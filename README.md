This Python package, `knowknow`, is an attempt to make powerful, modern tools for analyzing the structure of knowledge open to anyone.
Although I hope we can continue to improve the methods and documentation written here, and I intend that this grow larger than myself, this package acts as a stabilizing force for the field, giving us all access to the common methods and data for analyzing these structures.

I have included every inch of code here, leaving no stone unturned. With every `pip install knowknow-amcgail`, you download the following:

+ `creating variables`, a collection of pre-processing algorithms for cleaning and summarizing Web of Science search results, or JSTOR Data for Research data dumps.
+ `analyses`, a set of descriptive notebooks which illustrate these datasets
+ A connector to pre-computed cooccurrence sets, hosted on [OSF](https://osf.io/9vx4y/)

# Projects built on knowknow

+ [amcgail/citation-death](/amcgail/citation-death) applies the concept of 'death' to attributes of citations, and analyzes the lifecourse of cited works, cited authors, and the authors writing the citations, using the `sociology-wos` dataset.

# Datasets built with knowknow

+ Sociology ([hosted on OSF](https://osf.io/9vx4y/))
    + `sociology-wos` every paper in WoS in early 2020 whose journal is in the 'Sociology' category, and which have full data. See [the Web of Science counter for more information](knowknow/creating%20variables/counter%20-%20web%20of%20science%20(cnt).ipynb)
    + `sociology-jstor` in-text citations and their contexts were extracted from >90k full-text Sociology articles indexed in JSTOR. See [the JSTOR counter for more information](knowknow/creating%20variables/counter%20-%20jstor%20(cnt).ipynb)
    + `sociology-wos-all`

# Installation

1. Install Python 3.7
2. Install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
3. Run `pip install science2-amcgail`

# Quick start

The following command starts jupyterlab in the base directory of this repository. This is a good place to start.

`python -m science2 start`

# Developing

If you want to contribute edits of your own, fork this repository into your own GitHub account, make the changes, and submit a request for me to incorporate the code (a "pull request"). This process is really easy with GitHub Desktop ([tutorial here](https://www.youtube.com/watch?v=BYzriB5aTWU)).

There is a lot to do! If you find this useful to your work, and would like to contribute (even to the following list of possible next steps) but can't figure out how, please don't hesitate to reach out. My website is [here](http://www.alecmcgail.com), [Twitter here](https://twitter.com/SomeKindOfAlec). 

## Aimed completion by 5/22/2020 (ben rosche)

+ analyses complete, with explanations, annotations, and graphs

## Aimed completion by 5/29/2020 (committee)

+ literature review is tight, written, boom. everything down. finish it.

## Aimed completion by 6/5/2020 (presentation)

+ Externalizing data from the Git repository, so it can be dynamically downloaded / uploaded via AWS
+ trimming the paper and preparing it for publication

## Possible projects

+ The documentation for this project can always be improved. This is typically through people reaching out to me when they have issues. Please [feel free](https://twitter.com/SomeKindOfAlec).
+ An object-oriented model for handling context would prevent the need for so much variable-passing between functions, reduce total code volume, and improve readability.
+ Different datasets and sources could be incorporated, if you have the need, in addition to JSTOR and WoS.
+ If you produce precomputed binaries and have an idea of how we could incorporate the sharing of these binaries within this library, please [DM me](https://twitter.com/SomeKindOfAlec) or something. That would be great.
+ All analyses can be generalized to any counted variable of the citations. This wouldn't be tough, and would have a huge payout.
+ It would be amazing if we could make a graphical interface for this.
    + user simply imports data, chooses the analyses they want to run, fill in configuration parameters and press "go"
    + the output is a PDF with the code, visualizations, and explanations for a given analysis
    + behind the scenes, all this GUI does is run `nbconvert` 
    + also could allow users to regenerate any/all analyses for each dataset with the click of a button
    + could provide immediate access to online archives, either to download or upload similar count datasets
