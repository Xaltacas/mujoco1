# Implémentation de NAF sur des environnement Gym/Mujoco

DDPG : [le papier](https://arxiv.org/pdf/1603.00748.pdf)   

Code fortement inspiré de [ce Depot](https://github.com/BY571/Normalized-Advantage-Function-NAF-)

## Environnements:

#### LunarLanderContinuous :
Lunar lander de base de gym;  
pas d'altération.  

#### BipedalWalker :
Bipedal walker de base de Gym  
pas d'altération.

#### custom_env :
Le fetch de MuJoCo custom;   
reward policy: `- 100*d + (done)*100 - (failure)*300`
avec d la distance à la cible.  
(exploite les env custom_fetchEnv et custom_robotEnv)

## Resolutions:
#### NAF_fetch:
Résolution de custon_env  

#### NAF_lunarLander
Résolution de lunarLanderContinuous

#### NAF_walker
Résolution de bipedalWalker  


##### Toutes les resolutions peuvent être éxecutés avec les argument suivants:
```
--save "nom"  
      crée une sauvegarde du réseau dans le dossier "savedir/nom/"  

--savemax "nom"
      crée une sauvegarde du réseau ayant eu la plus haute reward dans le dossier "savedir/nom/"

--load "nom"  
      charge une sauvegarde du réseau depuis le dossier "savedir/nom/"  
      ATTENTION !! la taille du réseau chargé et celui déclaré dans DDPG_robot doivent correspondre.  

--visu  
      visualisation de l'entrainement
```


### Observations sur fetch:

Multiplier l'action par une constante avant de l'envoyer a l'environement a l'air de causer des problèmes de convergence

### Observations sur lunarLander:

Après environ quelques centaines d'épisodes la loss commence à diverger, plus tard le model va lui aussi diverger.  
La loss reviens petit a petit... attendre plus longtemps pour voir si on peut revenir vers quelque chose de stable?  

---
### Requirements

- **`python`** - `3.7`
- **`pyTorch`** -  `1.5.1`
- **`MuJoCo`** -
