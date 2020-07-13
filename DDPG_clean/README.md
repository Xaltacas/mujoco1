# Implémentation de DDPG sur des environnement Gym/Mujoco

DDPG : [le papier](https://arxiv.org/pdf/1509.02971v2.pdf)   

Code fortement inspiré de [ce Depot](https://github.com/shivaverma/OpenAIGym/)

## Environnements:

#### Continuous_CartPole :
le cartpole de gym en continu;  
reward policy: `-(theta**2 + 0.1*theta_dot**2 + x**2 + 0.1*x**2) + 1`
avec theta l'angle du baton et x la position du cart.  


#### LunarLanderContinuous :
Lunar lander de base de gym;  
pas d'alteration.  


#### custom_env :
le fetch de mujoco custon;   
reward policy: `-1 -d*d + (done) * 10`
avec d la distance à la cible.  
(exploite les env custom_fetchEnv et custom_robotEnv)

## Resolutions:
#### DDPG_robot:
Résolution de custon_env  


      arguments optionnels:  

      --save "nom"  
            crée une sauvegarde du réseau dans le dossier "savedir/nom/"  

      --load "nom"  
            charge une sauvegarde du réseau depuis le dossier "savedir/nom/"  
            ATTENTION !! la taille du réseau chargé et celui déclaré dans DDPG_robot doivent correspondre.  

      --loadBuff "nom"  
            charge une sauvegarde de buffer depuis le fichier "preTrain/nom.json.gz"

      --demo  
            visualisation avec que du tryhard, doit etre fait avec --load

      visu  
            visualisation de l'entrainement

##### DDPG_lunarLander
Résolution de lunarLanderContinuous  


##### DDPG_Cartpole
Résolution de Continuous_CartPole  

### Outils de résolution:
#### preTrain_Datagen:
Génère un buffer de "bonnes" actions pour la resolution de custom_env.  


      argument obligatoire:

      --save "nom"  
            crée une sauvegarde du buffer dans le fichier "preTrain/nom.json.gz"  

      optionnel:  

      --size "taille"
            définis la taille du buffer créé

#### preTrain_fit
Crée une sauvegarde du réseau entrainé de manière supervisée à partir d'un buffer donné.  


      arguments obligatoires:

      --save "nom"  
            crée une sauvegarde du reseau dans le dossier "savedir/nom/"  

      --loadBuff "nom"  
            charge une sauvegarde du buffer depuis le fichier "preTrain/nom.json.gz"
