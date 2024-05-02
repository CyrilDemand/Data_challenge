# Data challenge


- ca y est le code compile

# changements apportés

- epochs : 10 > 25 
(0.05 accuracy)
- epochs : 25 > 100 
(0.35 accuracy)
- epochs : 100 > 500, lr 0.0001 > 0.00001 
(0.80 max train, 0.31 accuracy test donc overfitting)
- lr 0.00001 > 0.0001, epochs 500 > 50, et on normalise les données (signal = librosa.util.normalize(signal))
(0.30 accuracy, guez)
- epochs 50 > 100 + force audio a 5sec
(0.34 accuracy, encore guez)
- nouvelle structure du réseau, ajout de couches (plus de couches de convolutions, BatchNormalization, GlobalAveragePooling2D et dense layers à la fin)
  (0.45 : c'est mieux, mais pas de plateau dans le graphe)
- on tente d'augmenter le lr 0.0001 > 0.00025, toujour sur 100 epochs
  (0.52 : encore mieux !)

A TESTER : 
renamme poids avec date génération pour pouvoir comparer les résultats
changer structure du réseau, ajouter des couches, etc