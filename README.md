# P6_Classification_Biens_Consommation : Classifiez automatiquement des biens de consommation

#### Enoncé : 
• Surlaplacedemarché,desvendeurs proposent des articles à des acheteurs en postant une photo et une description.
• L'attributiondelacatégoried'unarticle est effectuée manuellement par les vendeurs, et est donc peu fiable.
• Levolumedesarticlesestpourl’instant très petit mais il est destiné à s’accroitre.

#### Mission : 

Réaliserunepremièreétudedefaisabilité d'un moteur de classification d'articles,
basé sur une image et une description, pour l'automatisation de l'attribution de la catégorie de l'article.

#### Livrables :

- Un notebook contenant les fonctions permettant le prétraitement des données textes et images ainsi que les résultats du clustering (en y incluant des représentations graphiques).
- Un support de présentation qui présente la démarche et les résultats du clustering.

#### Compétences évaluées : 

- Prétraiter des données image pour obtenir un jeu de données exploitable
- Prétraiter des données texte pour obtenir un jeu de données exploitable
- Mettre en œuvre des techniques de réduction de dimension
- Représenter graphiquement des données à grandes dimensions

#### Algoritmes utilisés:

Texte :

- deux approches de type “bag-of-words”, comptage simple de mots et Tf-idf ;
- une approche de type word/sentence embedding classique avec Word2Vec (ou Glove ou FastText) ;
- une approche de type word/sentence embedding avec BERT ;
- une approche de type word/sentence embedding avec USE (Universal Sentence Encoder). 

Image:

-  SIFT 
- ORB
- CNN Transfer Learning ( VGG16).
