# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:30:13 2022

@author: leo
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps

def load_image(filename):
    """
    Importation de l'image, passage en nuance de gris, puis passage dans [0,1]
    """
    my_img = Image.open(filename)
    img_gray = ImageOps.grayscale(my_img)
    img_gray = np.array(img_gray)
    img_gray_binary = np.where(img_gray > 0.5, 1, 0) #Passage en binaire
    return img_gray_binary


def tograyscale(mat):
    """
    Scale les nombres de la matrice dans [0,255] (pour permettre les images en nuance de gris)
    Mais ne prend pas encore en compte le négatif
    """
    scaled_mat = 255.0*mat/mat.max()
    return scaled_mat


def noise(img, MU, TAU):
    """
    Ajout d'un bruit gaussien aux pixels de l'image
    """
    img_noised = np.zeros(shape = img.shape)
    row, col = np.shape(img)
    for i in range(row):
        for j in range(col):
            img_noised[i,j] = np.random.normal(loc = MU[int(img[i,j])], scale = TAU**2)
    
    return img_noised

# Je ne sais pas comment décider de ces valeurs
ALPHA = 0.0125
BETA = 0.4

def sample(i, j, X, Y, maxi, maxj):
    """
    Echantillonne dans la distribution conditionnelle selon la formule que j'ai calculée
    """
    MU = [0,1]
    TAU = 0.4
    # Ajout des valeurs des 4 voisins du pixel considéré, si pertinent
    voisins = []
    if i != 0:
        voisins.append(X[i-1][j])
    if i!= maxi-1:
        voisins.append(X[i+1][j])
    if j!= 0:
        voisins.append(X[i][j-1])
    if j!= maxj-1:
        voisins.append(X[i][j+1])
        
    # Calcul du nombre de 0 et de 1 dans les voisins
    nb_0 = sum(voisins[i] == 0 for i in range(len(voisins)))
    nb_1 = sum(voisins[i] == 1 for i in range(len(voisins)))
    
    # Prise en considération de la variation du nombre de voisins (bords)
    pond = 4/len(voisins) 
    
    # Calcul de la probabilité de valoir 1
    exp_valeur_1 = -ALPHA + BETA * pond * (nb_0-nb_1) + 1/2*TAU*(MU[1]**2 - MU[0]**2 + 2*Y[i][j]*(MU[0]-MU[1]))
    # Calcul de la probabilité de valoir 0 (juste pour vérifier)
    exp_valeur_0 = ALPHA + BETA * pond * (nb_1-nb_0) + 1/2*TAU*(MU[0]**2 - MU[1]**2 + 2*Y[i][j]*(MU[1]-MU[0]))
    
    #print(1/(1+np.exp(exp_valeur_1)))
    #print(1/(1+np.exp(exp_valeur_0)))
    
    prob = 1/(1+np.exp(exp_valeur_1))
    
    return(int(np.random.rand() < prob)) # Renvoie 1 si l'uniforme est inférieure à la proba de valoir 1
    
    
def get_posterior(filename, burn_in, samples):
    """
    Output les fréquences de valoir 1 pour chaque pixel
    """
    Y = load_image(filename) # Chargement de l'image
    posterior = np.zeros(Y.shape)
    X = np.random.choice([1,0], size = Y.shape) #Initialisation de la sortie
    maxi = Y.shape[0]
    maxj = Y.shape[1]
    
    for step in tqdm(range(burn_in + samples)):
        
        # Ici on boucle de manière déterministe, idéalement il faudrait le faire de manière aléatoire
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = sample(i, j, X, Y, maxi, maxj)
                X[i,j] = x
                
                if x == 1 and step >= burn_in:
                    posterior[i][j] += 1
    
    # On a ajouté 1 aux pixels à chaque itération réelle où il valait 1
    # On fait la moyenne
    posterior = posterior / samples

    return posterior

def denoise_image(filename, burn_in, samples):
    """
    Ici on considère comme 1 valant 1 les pixels où le posterior 
    vaut 1 à une fréquence supérieure à seuil
    """
    
    posterior = get_posterior(filename, burn_in, samples)
    denoised = np.zeros(posterior.shape, dtype=np.float64)
    denoised[posterior > 0.5] = 1 # POur passer des probas à des valeurs 
    
    return denoised