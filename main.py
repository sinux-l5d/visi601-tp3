#!/usr/bin/env python3
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
from scipy.sparse.linalg import splu
import matplotlib.image as mpimg
import matplotlib.animation as animation
from math import sqrt, floor


class Grid:
    def __init__(self, image):
        """ Constructeur. Fabrique une grille avec `nbrow` lignes et `nbcol` colonnes.  """
        self.nbrow = image.shape[0]
        self.nbcol = image.shape[1]

        self.I = [i for i in range(self.nbrow)
                  for j in range(self.nbcol)
                  if image[i, j] != 0]

        self.J = [j for i in range(self.nbrow)
                  for j in range(self.nbcol)
                  if image[i, j] != 0]

        self.index = {(self.I[i], self.J[i]): i for i in range(self.size())}

    def getIndex(self, i, j):
        """ Retourne le numéro/indice du pixel `(i,j)`.  """
        return self.index.get((i, j), -1)

    def getIndexOr0(self, i, j):
        """ Retourne le numéro/indice du pixel `(i,j)` ou 0 (matrice creuse).  """
        return self.index.get((i, j), 0)

    def getRow(self, idx):
        """ Retourne la ligne du pixel d'indice `idx` """
        return self.I[idx]

    def getCol(self, idx):
        """ Retourne la colonne du pixel ayant pour indice `idx` """
        return self.J[idx]

    def size(self):
        """
        Taille n du vecteur u.
        """
        return len(self.I)

    def _neighbors_coords(self, i, j):
        yield i-1, j
        yield i+1, j
        yield i, j-1
        yield i, j+1

    def neighbors(self, idx):
        """
        Retourne la liste des voisins direct du sommet d'indice idx
        """
        N = []
        try:
            i = self.getRow(idx)
            j = self.getCol(idx)
        except IndexError:
            return []

        for ni, nj in self._neighbors_coords(i, j):
            if (ni, nj) in self.index:
                N.append(self.getIndex(ni, nj))

        return N

    def Identity(self):
        """
        Retourne la matrice identite de taille n*n
        """
        n = self.size()
        LIGS = []  # les lignes des coefficients
        COLS = []  # les colonnes des coefficients
        VALS = []  # les valeurs des coefficients
        for idx in self.index.values():
            LIGS.append(idx)
            COLS.append(idx)
            VALS.append(1.0)
        # print(LIGS, COLS, VALS )
        M = coo_matrix((VALS, (LIGS, COLS)), shape=(n, n))
        return M.tocsc()

    def implicitEuler(self, U0, T, dt):
        """"
        A partir du vecteur de valeurs U0, calcule U(T) en itérant des pas dt successifs.
        permet d'obtenir une marge d'erreur plus petite par rapport a explicitEuler
        """
        Id = self.Identity()
        U = np.array(U0)
        L = self.Laplacian()
        lu = splu(Id - dt * L)

        for _ in np.arange(0, T, dt):
            U = lu.solve(U)

        return U

    def Laplacian(self):
        """ Retourne le laplacien et retourne la matrice creuse """
        n = self.size()
        LIGS = []  # les lignes des coefficients
        COLS = []  # les colonnes des coefficients
        VALS = []  # les valeurs des coefficients

        # on parcourt les indices
        for k in self.index.values():
            # on calcule les voisins de l'indice
            voisins = self.neighbors(k)
            nbvoisins = len(voisins)

            # on met pour valeur - le nombre des voisins si on est sur (i,i)
            LIGS.append(k)
            COLS.append(k)
            VALS.append(-len(voisins))

            # on met pour valeur 1 si on est sur une colonne d'un voisin
            for v in voisins:
                LIGS.append(k)
                COLS.append(v)
                VALS.append(1.0)

        L = coo_matrix((VALS, (LIGS, COLS)), shape=(n, n))
        return L.tocsc()

    # laplacien de la matrice creuse avec methode Dirichlet
    def LaplacianD(self):
        """ Retourne le laplacien de Dirichlet et retourne la matrice creuse 
            --> juste changer les -(len.. ) en -4 constant
        """
        n = self.size()

        LIGS = []  # les lignes des coefficients
        COLS = []  # les colonnes des coefficients
        VALS = []  # les valeurs des coefficients

        # on parcourt les indices
        for k in self.index.values():
            # on calcule les voisins de l'indice
            voisins = self.neighbors(k)
            nbvoisins = len(voisins)

            # on met pour valeur - 4 si on est sur (i,i)
            LIGS.append(k)
            COLS.append(k)
            VALS.append(-4.0)

            # on met pour valeur 1 si on est sur une colonne d'un voisin
            for v in voisins:
                LIGS.append(k)
                COLS.append(v)
                VALS.append(1.0)

        L = coo_matrix((VALS, (LIGS, COLS)), shape=(n, n))
        return L.tocsc()

        # avec Dirichlet
    def implicitEulerD(self, U0, T, dt):
        """"
        A partir du vecteur de valeurs U0, calcule U(T) en itérant des pas dt successifs. 
        permet d'obtenir une marge d'erreur plus petite par rapport a explicitEuler
        avec l'utilistion du Laplacian D
        """
        Id = self.Identity()
        U = np.array(U0)
        L = self.LaplacianD()
        lu = splu(Id - dt * L)

        for _ in np.arange(0, T, dt):
            U = lu.solve(U)

        return U

    def vectorToImage(self, V):
        img = np.zeros((self.nbrow, self.nbcol))
        K = self.index.keys()
        I = self.index.values()
        for k, idx in zip(K, I):
            img[k[0], k[1]] = V[idx]
        return img

    # given a list of pixels [(i,j),...], return the vector U0 with a value 1 on these sources
    def sources(self, H):
        U0 = np.zeros(self.size())
        for i, j in H:
            U0[self.index[i, j]] = 1
        return U0

    # return U0 where its value is 1 on its boundary pixels
    def boundary(self):
        U0 = np.zeros(self.size())
        lap = self.Laplacian()

        for i in range(lap.shape[0]):
            if lap[i, i] > -4:
                U0[i] = 1

        return U0

# ne fonctionne pas./.. pourquoi ??
    def gradX(self):
        # POUR LES X :
        # on a 4 cas :
        # cas 1 : si (i,j-1) et (i,j+1) sont dans omega (pas noir) : alors derivee x =  (u(i,j+1) - u(i,j - 1))/2
        # cas 2 : si (i,j-1) PAS dans omega et (i,j+1) dans omega : alors derivee x = u(i,j+1) - u(i,j)
        # cas 3 : si (i,j-1) dans omega et (i,j+1) PAS dans omega : alors derivee x = u(i,j) - u(i,j-1)
        # cas 4 : sinon (les deux if(pas dans omega) : alors derivee x = 0
        # return grad
        n = self.size()
        LIGS = []  # les lignes des coefficients
        COLS = []  # les colonnes des coefficients
        VALS = []  # les valeurs des coefficients

        # on parcourt les indices
        for i in range(n):
            # on recupere les pixels
            pixelsous = self.getIndex(self.I[i], self.J[i]-1)
            pixelsur = self.getIndex(self.I[i], self.J[i]+1)
            # on verifie qu'ils sont bien dans omega
            if (pixelsous != -1 and pixelsur != -1):

                # on met pour valeur -1/2 si u(i,j-1)
                LIGS.append(i)
                COLS.append(pixelsur)
                VALS.append(1/2)

                LIGS.append(i)
                COLS.append(pixelsous)
                VALS.append(-1/2)

            elif (pixelsous != -1):
                LIGS.append(i)
                COLS.append(pixelsous)
                VALS.append(-1)

                LIGS.append(i)
                COLS.append(i)
                VALS.append(1)

            elif (pixelsur != -1):

                LIGS.append(i)
                COLS.append(pixelsur)
                VALS.append(-1)

                LIGS.append(i)
                COLS.append(i)
                VALS.append(1)
        L = coo_matrix((VALS, (LIGS, COLS)), shape=(n, n))
        return L.tocsc()

    def gradY(self):
        # POUR LES Y : # on a 4 cas :
        # cas 1 : si (i-1,j) et (i+1,j) sont dans omega (pas noir) : alors derivee x =  (u(i+1,j) - u(i-1,j))/2
        # cas 2 : si (i-1,j) PAS dans omega et (i+1,j) dans omega : alors derivee x = u(i+1,j) - u(i,j)
        # cas 3 : si (i-1,j) dans omega et (i+1,j) PAS dans omega : alors derivee x = u(i,j) - u(i-1,j)
        # cas 4 : sinon (les deux pas dans omega) : alors derivee x = 0

        n = self.size()
        LIGS = []  # les lignes des coefficients
        COLS = []  # les colonnes des coefficients
        VALS = []  # les valeurs des coefficients

        # on parcourt les indices
        for i in range(n):
            # on recupere les pixels
            pixelsous = self.getIndex(self.I[i]-1, self.J[i])
            pixelsur = self.getIndex(self.I[i]+1, self.J[i])
            # on calcule les voisins de l'indice
            if (pixelsous != -1 and pixelsur != -1):

                # on met pour valeur -1/2 si u(i,j-1)
                LIGS.append(pixelsur)
                COLS.append(i)
                VALS.append(1/2)

                LIGS.append(pixelsous)
                COLS.append(i)
                VALS.append(-1/2)

            elif (pixelsous != -1):
                LIGS.append(pixelsous)
                COLS.append(i)
                VALS.append(-1)

                LIGS.append(i)
                COLS.append(i)
                VALS.append(1)

            elif (pixelsur != -1):
                LIGS.append(i)
                COLS.append(i)
                VALS.append(-1)

                LIGS.append(pixelsur)
                COLS.append(i)
                VALS.append(1)

        L = coo_matrix((VALS, (LIGS, COLS)), shape=(n, n))
        return L.tocsc()

    # ne fonctioone pas : cf transfert avec sqrt dans q4 qui ne fonctionne pas ...
    def direction(self, U):
        gradU = (self.gradX()*U, self.gradY()*U)
        # print(f'{gradU=}')
        # return -1 * gradU / np.sqrt(np.multiply())
        return gradU

    # deuxieme methode: on calcule la direction du gradient et on append avec les racines carrées direction ici
    def direction2(self, U):
        vx = []
        vy = []
        xx = self.gradX()*U
        xy = self.gradY()*U
        for k in range(len(xx)):
            vx.append(-xx[k] / (sqrt(np.multiply(xx[k], xx[k]) +
                      np.multiply(xy[k], xy[k]))))
            vy.append(-xy[k] / (sqrt(np.multiply(xx[k], xx[k]) +
                      np.multiply(xy[k], xy[k]))))
        return vx, vy

    def divergence(self, vx, vy):
        return np.sum(vx) + np.sum(vy)

    # Ecrivez la méthode Grid.poisson qui résoud l’équation précédente. On peut utiliser la méthode PLU, par exemple via scipy.linalg.splu
    # ca marche paaaaas
    def poisson(self, B):
        # print(f'{b=}')
        lap = self.Laplacian()
        lu = splu(lap)
        bis = np.zeros((lu.shape[0], 1))
        bis.fill(B)
        dbis = lu.solve(bis)
        mini = dbis.min()
        d = dbis - mini
        return d

    def poissonD(self, B):
        # print(f'{b=}')
        lap = self.LaplacianD()
        lu = splu(lap)
        bis = np.zeros((lu.shape[0], 1))
        bis.fill(B)
        dbis = lu.solve(bis)
        mini = dbis.min()
        d = dbis - mini
        return d

    def heatMethod(self, U0, Lap='Neumann', T=100, dt=10):
        if (Lap == 'Neumann'):
            self.L = self.Laplacian()
            U = self.implicitEuler(U0, T, dt)
            ux, vx = self.direction2(U)
            B = self.divergence(ux, vx)
            D = self.poisson(B)

        else:
            self.L = self.Laplacian()
            U = self.implicitEulerD(U0, T, dt)
            ux, vx = self.direction2(U)
            B = self.divergence(ux, vx)
            D = self.poissonD(B)
        return D


# pour la question V
# n est le nombre de lignes de niveaux visibles, width est leur largeur.
def transformDistance(D, n=50, width=0.2):
    TD = D.copy()
    M = D.max()
    E = M / n
    for i in range(D.shape[0]):
        v = D[i] / E
        e = v-floor(v)
        if (e < width):
            TD[i] = 1.5*M
    return TD


def getG():
    img = mpimg.imread('exemple-1.png')
    red = img[:, :, 0]
    return Grid(red)


def q1():
    # test 1: pour charger une des images
    # load color image
    # img = mpimg.imread('exemple-1.png')
    # # image N&B entre 0 and 1, qui correspond au canal rouge
    # red = img[:, :, 0]
    # imgplot = plt.imshow(red, cmap='gray', vmin=0., vmax=1.)
    # plt.show()  # éventuellement

    # test 2: pour la definition du Laplacien
    # img = mpimg.imread('exemple-1.png')
    # # image N&B entre 0 and 1, qui correspond au canal rouge
    # red = img[:, :, 0]
    # G = Grid(red)
    # L = G.Laplacian()  # Laplacien associé à Neumann
    # print(f'{L=}')

    # test 3: pour verifier l'apparence du Laplacien
    A = np.zeros((5, 5))
    A[:, 1] = 1.
    A[2, :] = 1.
    A[3, 2] = 1.
    print(A)
    G = Grid(A)
    L = G.Laplacian()  # Laplacien pour les conditions de Neumann
    print(L)


def q2():
    # définir l’initialisation pour une source en (1,1
    G = getG()

    # définir l’initialisation pour  des sources sur tous les bords du domaine:
    U0 = G.boundary()
    Visu_U0 = G.vectorToImage(U0.flatten())
    imgplot = plt.imshow(Visu_U0, cmap='gray', vmin=0., vmax=1.0)
    plt.show()


def q3():
    G = getG()
    # diffuse jusqu'à T=1000, par pas dt=100
    U0 = G.sources([(1, 1)])

    fig = plt.figure()
    ims = []

    U = U0
    for _ in range(100):
        # for i in range(0, 10000, 1000):
        img = G.vectorToImage(U)
        im = plt.imshow(img, cmap='hot', vmin=0.0, vmax=1e-5, animated=True)
        ims.append([im])
        # U = G.implicitEuler(U0, i, 100)
        U = G.implicitEuler(U, 100, 100)

    ani = animation.ArtistAnimation(
        fig, ims, interval=50, blit=True, repeat_delay=1000)

    plt.show()


# fonctionne passssss
def q4():
    G = getG()
    U0 = G.sources([(1, 1)])
    U = G.implicitEuler(U0, 1000, 100)  # T=1000 recuperer U ?????
    Vx, Vy = G.direction(U)

    N = np.sqrt(np.multiply(Vx, Vx), np.multiply(Vy, Vy))
    Yx = -1.0 * Vx / N
    Yy = -1.0 * Vy / N

    visu_VX = G.vectorToImage(Yx)
    plt.imshow(visu_VX, cmap='viridis', vmin=-1., vmax=1.)
    plt.show()
    visu_VY = G.vectorToImage(Yy)
    plt.imshow(visu_VY, cmap='viridis', vmin=-1., vmax=1.)
    plt.show()

# fonctionne : changeemlnt d'utilisation de direction (on passe sur direction2)
# donc les gradX et gradY fonctionnent bien


def q4Test2():
    G = getG()
    U0 = G.sources([(1, 1)])
    U = G.implicitEuler(U0, 100, 100)  # T=1000 recuperer U ?????
    Vx, Vy = G.direction2(U)
    visu_VX = G.vectorToImage(Vx)
    plt.imshow(visu_VX, cmap='viridis', vmin=-1., vmax=1.)
    plt.show()
    visu_VY = G.vectorToImage(Vy)
    plt.imshow(visu_VY, cmap='viridis', vmin=-1., vmax=1.)
    plt.show()


def q4bis():
    G = getG()
    U0 = G.sources([(1, 1)])
    U = G.implicitEuler(U0, 100, 100)
    ux, vx = G.direction2(U)
    B = G.divergence(ux, vx)
    D = G.poisson(B)
    visu_D = G.vectorToImage(D)
    plt.imshow(visu_D, cmap='hot', vmin=-1., vmax=1.)
    plt.show()

# marche pas car poisson fonctionne pas ... why ? j'en sais rien


def q5():
    img = mpimg.imread('exemple-4.png')
    red = img[:, :, 0]
    G = Grid(red)
    U0 = G.sources([(200, 200)])
    U = G.implicitEuler(U0, 100, 100)
    ux, vx = G.direction2(U)
    B = G.divergence(ux, vx)
    D = G.poisson(B)
    D_bis = transformDistance(D)
    visu_D = G.vectorToImage(D_bis)
    plt.imshow(visu_D, cmap='hot')
    plt.show()


def q6():
    G = getG()
    U0 = G.sources([(1, 1)])
    D = G.heatMethod(U0, 'Neumann', 1000., 100.)
    TD = transformDistance(D, 50, 0.2)
    visu_TD = G.vectorToImage(TD)
    plt.imshow(visu_TD, cmap='hot')
    plt.show()


def main():
    # q1()
    # q2()
    # q3()
    #  q4()
    # q4Test2()
    # q4bis()
    q5()
    # q6()
    # print(G.boundary())


if __name__ == "__main__":
    main()
