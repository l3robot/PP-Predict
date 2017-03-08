PP-Predict: Prediction of Political Party from Speeches
===
Université Laval

### Authors :
- Louis-Émile Robitaille
- Martin Richard-Cerda
- Arnaud Cavrois

### What is this project?

Here is our implementation of a bag-of-words technique to predict the political party that wrote a speech with the text of the speech only. This project was done as part of our [Machine Learning](https://www.ulaval.ca/les-etudes/cours/repertoire/detailsCours/gif-4101-apprentissage-et-reconnaissance.html) course in the fall of 2016. 

### Description of the project



### Code Structure

- **bow** : bag-of-words implementation
- **others** : miscellaneous
- **stats** : statistics module
- **visual** : to make nice graphics
- **xp** : experiments code

### Raw Data Structure

For each file, you have to follow this structure:
```
***** [Speaker] [subject] [party]\n
[speech content]\n
***** [Next Speaker] [next subject] [next party]\n
[speech content]\n
...
...
...
```

### Results

See the [report](rapport.pdf) (in french) explaining the technique and describing our results. If you want to look at the data we used, don't hesitate to ask us.
