# Implémentation de DDPG sur des environnement Gym/Mujoco

DDPG : [le papier](https://arxiv.org/pdf/1509.02971v2.pdf)   

Code fortement inspiré de [ce Depot](https://github.com/shivaverma/OpenAIGym/)

## Environnements:

#### Continuous_CartPole :
le cartpole de gym en continu;  
reward policy: `-(theta**2 + 0.1*theta_dot**2 + x**2 + 0.1*x**2) + 1`
avec theta l'angle du bâton et x la position du cart.  


#### LunarLanderContinuous :
Lunar lander de base de gym;  
pas d'altération.  


#### custom_env :
le fetch de mujoco custom;   
reward policy: `-1 -d*d + (done) * 10`
avec d la distance à la cible.  
(exploite les env custom_fetchEnv et custom_robotEnv)

## Resolutions:
#### DDPG_robot:
Résolution de custon_env  

#### DDPG_lunarLander
Résolution de lunarLanderContinuous  


#### DDPG_Cartpole
Résolution de Continuous_CartPole  

##### Toutes les resolutions peuvent être éxecutés avec les argument suivants:
```
--save "nom"  
      crée une sauvegarde du réseau dans le dossier "savedir/nom/"  

--load "nom"  
      charge une sauvegarde du réseau depuis le dossier "savedir/nom/"  
      ATTENTION !! la taille du réseau chargé et celui déclaré dans DDPG_robot doivent correspondre.  

--loadBuff "nom"  
      charge une sauvegarde de buffer depuis le fichier "preTrain/nom.json.gz"

--demo  
      visualisation avec que du tryhard, doit etre fait avec --load

--visu  
      visualisation de l'entrainement
```

### Outils de résolution:
#### preTrain_Datagen:
Génère un buffer de "bonnes" actions pour la resolution de custom_env.  


      argument obligatoire:

      --save "nom"  
            crée une sauvegarde du buffer dans le fichier "preTrain/nom.json.gz"  

      optionnels:  

      --size "taille"
            définis la taille du buffer créé

      --mstep "nb"
            definis le nombre de step dans la simulation par action

#### preTrain_fit
Crée une sauvegarde du réseau entrainé de manière supervisée à partir d'un buffer donné.  


      arguments obligatoires:

      --save "nom"  
            crée une sauvegarde du réseau dans le dossier "savedir/nom/"  

      --loadBuff "nom"  
            charge une sauvegarde du buffer depuis le fichier "preTrain/nom.json.gz"

      optionnels:

      --mstep "nb"
            definis le nombre de step dans la simulation par   
            default : 1

      --ep "nb"
            definis le nombre d'episode sur lesquels le reseau sera entrainé  
            default : 10000

### Observations sur fetch:
Le learning rate et tau doivent être très bas pour avoir de la stabilité dans la convergence  
Des paramètres qui ont l'air de marcher :  
```
ep = 10000
tau = 0.0001
gamma = 0.99
min_batch = 32
actor_lr = 0.00001
critic_lr = 0.0001
buffer_size = 1000000
layers = [512,256]
```
~~(début de convergence en 300 ep avec un réseau pré train)~~
la convergence est toujours random c'est chiant

le réseau a tendance à diverger quelques centaines d'épisodes après avoir convergé...

~~Les gros réseau ont l'air de converger au bout d'un temps (prometteur:[1024,512])~~  
Les gros overfit ou ne convergent jamais... les petit sont sans doute mieux

La plupart des network ont l'air de tomber dans un minima local avec le bras en pleine extension a l'avant...  
_(corriger ca avec la RewardPolicy?)_

---

### Requirements

- **`python`** - `3.7`
- **`tensorflow`** -  `2.2.0`
- **`MuJoCo`** -
