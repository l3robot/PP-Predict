PP-Predict: Prediction of Political Party from Speeches
===
Université Laval

### Authors :
- Louis-Émile Robitaille
- Martin Richard-Cerda
- Arnaud Cavrois

### What is this project?

Here is our implementation of a bag-of-words technique to predict the political party from speeches. This project was done as part of our [Machine Learning](https://www.ulaval.ca/les-etudes/cours/repertoire/detailsCours/gif-4101-apprentissage-et-reconnaissance.html) course in the fall of 2016. 

### Project description

We construct bag-of-words (BoW) from treated raw_data. Then, we can construct a vector associated with a speech and train a simple LinearSVM to calssified the speeched. With tested 3 versions of bag-of-words:

- **Classic BoW**: 1-Gram BoW. We cut the words with to high or to low overall frequence.
- **N-Gram BoW**: Different sizes of N to study the influence of N. We also cut the words with to high or to low overall frequence.
- **Norm-BoW**: We use an equation to weight the frequence of a word with respect of its overall frequence. Hence we do not have to cut the frequent and non-frequent words. 

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
