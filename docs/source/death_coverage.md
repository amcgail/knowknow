# Citation death analysis

 This analysis is done using knowknow library ,It applies the concept of 'death' to attributes of citations, and analyzes the lifecourse of cited works, cited authors, and the authors writing the citations, using the sociology-wos dataset.
For details visit this [github repo](https://github.com/amcgail/citation-death)

## A study of death and demographics, of citation attributes

<!-- For full documentation on this study, visit [alecmcgail.com/knowknow/citation-death](http://alecmcgail.com/knowknow/citation-death) --> 
This study hinges on the analogy between the historical number of citations through time, and the life course of a person. In it I define death and rebirth, and produce basic demographic plots for arbitrary attributes of a citation. This study looks at web of science for sociology in particular.

## P( die in next 5 years | alive )


![whatev](https://github.com/amcgail/citation-death/blob/master/figures/How%20likely%20are%20living%20cited%20works%20to%20die%24_1%24%2C%20cohorts%201965%2D1980%20%28sociology%2Dwos%29.png)
![whatev](https://github.com/amcgail/citation-death/blob/master/figures/How%20likely%20are%20living%20cited%20works%20to%20die%24_1%24%2C%20cohorts%201980%2D1995%20%28sociology%2Dwos%29.png)


![whatev](https://github.com/amcgail/citation-death/blob/master/figures/How%20likely%20are%20living%20cited%20authors%20to%20die%24_1%24%2C%20cohorts%201965%2D1980%20%28sociology%2Dwos%29.png)
![whatev](https://github.com/amcgail/citation-death/blob/master/figures/How%20likely%20are%20living%20cited%20authors%20to%20die%24_1%24%2C%20cohorts%201980%2D1995%20%28sociology%2Dwos%29.png)

![whatev](https://github.com/amcgail/citation-death/blob/master/figures/How%20likely%20are%20living%20citing%20authors%20to%20die%24_1%24%2C%20cohorts%201965%2D1980%20%28sociology%2Dwos%29.png)
![whatev](https://github.com/amcgail/citation-death/blob/master/figures/How%20likely%20are%20living%20citing%20authors%20to%20die%24_1%24%2C%20cohorts%201980%2D1995%20%28sociology%2Dwos%29.png)

![whatev](https://github.com/amcgail/citation-death/blob/master/figures/How%20likely%20are%20living%20first%20citing%20authors%20to%20die%24_1%24%2C%20cohorts%201965%2D1980%20%28sociology%2Dwos%29.png)
![whatev](https://github.com/amcgail/citation-death/blob/master/figures/How%20likely%20are%20living%20first%20citing%20authors%20to%20die%24_1%24%2C%20cohorts%201980%2D1995%20%28sociology%2Dwos%29.png)


## Sleeping beauties

![fig2](figures/Sleeping%20beauties%20-%20between%20pub%20and%20first%20%28sociology%2Dwos%29.png)

## Probability dead by 2010

![fig3](figures/Probability%20dead%20by%202010%20%28c%2C%20sociology%2Dwos%2C%201980%2C%201990%29.png)
![fig3](figures/Probability%20dead%20by%202010%20%28ffa%2C%20sociology%2Dwos%2C%201980%2C%201990%29.png)
![fig3](figures/Probability%20dead%20by%202010%20%28ta%2C%20sociology%2Dwos%2C%201980%2C%201990%29.png)
![fig3](figures/Probability%20dead%20by%202010%20%28fa%2C%20sociology%2Dwos%2C%201980%2C%201990%29.png)

## There are more!

Feel free to explore all the figures included in this, and make your own. I hope this proves helpful!

## Reproduce the results

To reproduce these results, or modify them for your own purposes, follow one of the following instructions...

## Google's Colaboratory (Cloud)

You can run these algorithms completely free through your browser, thanks to Google.

1. Clone this repo into your own GitHub account. 
2. Open Google's [Colab](colab.research.google.com) environment.
3. From Colab's File menu, choose "Open," and open one of the notebooks in this repository from GitHub. The first cell should automatically install `knowknow` for you. If it doesn't, run `!pip install knowknow-amcgail` in the Colab notebook.

## Jupyter notebook (Locally)

1. Use [GitHub desktop](https://desktop.github.com/) to clone this repository and download it to your own desktop or laptop. 
2. Install Python 3 and `pip install knowknow-amcgail`
3. In the "citation-death" folder, execute `jupyter lab`.
4. Explore the .ipynb files to see how the analyses were run, and run yourself.
5. Google if you get stuck :) or DM me ([@someKindOfAlec](https://twitter.com/SomeKindOfAlec))

<!--
In a cell of a Colab notebook, run the following code:
```python
!pip install knowknow-amcgail
!python -m knowknow init
!git clone https://github.com/amcgail/citation-death
```
-->