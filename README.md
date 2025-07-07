# Prototype rendement 


## YOLOv5 
Le code est basé sur YOLOv5, rien de nouveau à ce sujet.

## Rendement 
Il est probable qu'il soit nécessaire de faire des installations supplémentaires ! 

Est-ce que c'est possible sur jetson ?? 
```
pip install filterpy
pip install scipy
```

Il faut lancer le fichier proto.py de la sorte  :
```
python3 proto.py image_path --weights best.pt
```

Pour optimiser le code est profiter de la puissance de tensorrt sur jetson : 

```
python export.py --weights best.pt --include onnx engine
```

Puis, relancer le programme proto avec le .engine

```
python3 proto.py image_path --weights best.engine
```
