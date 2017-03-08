PP-Predict: Prediction du Parti Politique à partir de discours
===
Université Laval

### Auteurs :
- Louis-Émile Robitaille
- Martin Richard-Cerda
- Arnaud Cavrois

### Qu'est-ce que ce projet?

Voici notre implémentation d'une technique utilisant les _bag-of-words_ pour prédire le parti politique à l'aide d'un discours transcrit. Ce projet a été réalisé à l'automne 2016 dans le cadre de notre cours d'[apprentissage et reconnaissance](https://www.ulaval.ca/les-etudes/cours/repertoire/detailsCours/gif-4101-apprentissage-et-reconnaissance.html).

### Description du projet

Voir le [rapport](rapport.pdf).

### Structure du code

- **bow** : notre implémentation de _bag-of-words_
- **others** : varia
- **stats** : module de statistique
- **visual** : pour faire de beaux graphiques
- **xp** : code pour les expériences

### Strcuture des données brutes (_raw\_data_)

Pour chaque fichier de données, suivre cette structure:
```
***** [Parlementaire] [sujet] [parti]\n
[texte du discours]\n
***** [Prochain parlementaire] [prochain sujet] [prochain parti]\n
[texte du discours]\n
...
...
...
```

### Resultats

Voir le [rapport](rapport.pdf) pour avoir une description de notre technique et de nos résultats. Si vous êtes intéressés à voir les données utilisées pour le projet, n'hésitez pas à nous contacter.