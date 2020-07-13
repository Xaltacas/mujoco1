# Implémentation de DDPG sur des environnement Gym/Mujoco

DDPG : [le papier](https://arxiv.org/pdf/1509.02971v2.pdf)  

## Environnements:

#####Continuous_CartPole :
le cartpole de gym en continu;  
reward policy:''' -(theta**2 + 0.1*theta_dot**2 + x**2 + 0.1*x**2) + 1  '''
avec theta l'angle du baton et x la position du cart.  


#####LunarLanderContinuous :
Lunar lander de base de gym;  
pas d'alteration.  


#####custom_env :
le fetch de mujoco custon;   
reward policy: ''' -d * d + (done) * 10  +- 1 if d < previousD  '''
avec d la distance à la cible.  
(exploite les env custom_fetchEnv et custom_robotEnv)

## Resolutions:
DDPG_robot : pour custon_env  
      arguments:  

      --save "nom"  
            crée une sauvegarde du réseau dans le fichier "savedir/nom/"  

      --load "nom"  
            charge une sauvegarde du réseau depuis le fichier "savedir/nom/"  
            **!! la taille du réseau chargé et celui déclaré dans DDPG_robot doivent correspondre.**  

      --loadBuff "nom"  
            charge une sauvegarde de buffer depuis le fichier "preTrain/nom.json.gz"

DDPG_lunarLander : pour lunarLanderContinuous  


DDPG_Cartpole : pour Continuous_CartPole  
