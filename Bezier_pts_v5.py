from math import sqrt
from random import randint

ColorsDark=["blue","green","Maroon","DarkMagenta","Sienna","FireBrick",
    "DeepPink","brown","Purple","DarkOliveGreen","DarkSlateGrey","MidnightBlue",
    "Violet","Crimson","Fuchsia","DarkOrange","Teal","RoyalBlue","SeaGreen","red"]
ColorsLight=["PaleTurquoise","Tan","LightGray","PaleGoldenRod","LightCyan","Azure",
    "PaleGoldenRod","HoneyDew","Lavender","AntiqueWhite","MintCream","MistyRose",
    "LavenderBlush","LightYellow","LightPink","Beige","LightGoldenRodYellow","Linen",
    "WhiteSmoke","PeachPuff"]

def translat(P1,P2,lbda):
    """Renvoie le point image de P1 par translation de vecteur lbda fois
    le vecteur P1P2 où P1 et P2 sont des tuples : (x1,y1) et (x2,y2)."""
    Lt=[u + lbda*(v-u) for (u,v) in zip(P1,P2)]
    return Lt[0], Lt[1]

def bezierNPt(L,lbda):
    """Renvoie les coordonnées du point de la courbe de Bézier d'ordre n
    définie par la liste L des points de contrôle au paramètre lambda (lbda)
    L est une liste de couples de coordonnées (Liste de tuples)"""
    Ls = [k for k in L] # Reproduit la liste pour ne pas écraser l'originale
    assert len(Ls)>1 and len(Ls[0])==2 , "L n'est pas constitué de bons arguments"
    od = len(Ls) - 1
    for j in range(od):
        for k in range(od-j):
            Ls[k] = translat(Ls[k], Ls[k+1], lbda)
        Ls.pop(-1) # retire le dernier élément de la liste Ls
    R = Ls[0] # R est le point résultant
    del(Ls) # Supprime la liste temporaire
    return R

def CadreBezier(L, n=200, entier=True):
    """Renvoie les dimensions du cadre dans lequel se situe la courbe
    de Bézier définie par les points de contrôle de la liste L.
    Pour cela on calcule n points de la courbe. Si entier=False les
    dimensions du cadre ne sont pas arrondies"""
    xMin, yMin = L[0]
    xMax, yMax = L[0]
    for k in range(1, n+1):
        t = k/n
        xB, yB = bezierNPt(L,t)
        if xB < xMin :
            xMin = xB
        elif xB > xMax:
            xMax = xB
        if yB < yMin:
            yMin = yB
        elif yB > yMax:
            yMax = yB
    if entier:
        xMin = arrondi(xMin,False)
        yMin = arrondi(yMin,False)
        xMax = arrondi(xMax,True)
        yMax = arrondi(yMax,True)
    return xMin, xMax, yMin, yMax

def CadreBSV(L, n=200, entier=True):
    """Renvoie les dimensions du cadre dans lequel se situe une figure
    présentant un axe de symétrie vertical définie par les points de contrôle
    de la liste L. Pour cela on calcule n points de la courbe.
    Si entier=False les dimensions du cadre ne sont pas arrondies"""
    xMin, yMin = L[0]
    xMax, yMax = L[0]
    for k in range(1, n+1):
        t = k/n
        xB, yB = bezierNPt(L,t)
        if xB < xMin :
            xMin = xB
        elif xB > xMax:
            xMax = xB
        if yB < yMin:
            yMin = yB
        elif yB > yMax:
            yMax = yB
    if abs(xMin)>abs(xMax):
        xMax = -xMin
    else:
        xMin = -xMax
    if entier:
        xMin = arrondi(xMin,False)
        yMin = arrondi(yMin,False)
        xMax = arrondi(xMax,True)
        yMax = arrondi(yMax,True)
    return xMin, xMax, yMin, yMax

def CadreBS4(L,n=200,entier=True):
    """Renvoie les dimensions du cadre dans lequel se situe une figure
    2B3S4 définie par les paramètres de la liste L.
    Pour cela on calcule n points de la courbe. Si entier=False les
    dimensions du cadre ne sont pas arrondies"""
    Ls = [(L[0],L[0]),(L[0]+L[3],L[0]+L[4]),(L[1]+L[5],L[2]+L[6]),(L[1],L[2])]
    Lcmax = CadreBezier(Ls, n, entier)
    cmax1 = max(abs(k) for k in Lcmax)
    Ls = [(L[1],L[2]),(L[1]-L[5],L[2]-L[6]),(L[1]-L[5],L[6]-L[2]),(L[1],-L[2])]
    Lcmax = CadreBezier(Ls, n, entier)
    cmax2 = max(abs(k) for k in Lcmax)
    cmax = max(cmax1,cmax2)
    if entier:
        cmax = arrondi(cmax,True)
    return cmax

def RMaxFig(L, n=200, crx=True):
    """Renvoie la distance max à zéro de l'un des points de la figure
    définie par la liste L il peut s'agir d'une courbe de Bézier ou
    Si crx=True d'une croix L contenant alors les 7 paramètres de la croix:
    xA, xB, yB, xu, yu, xv, yv"""
    if crx:
        assert isinstance(L[0],(int,float)), 'erreur dans la liste'
        Ls = [(L[0],L[0]),(L[0]+L[3],L[0]+L[4]),(L[1]+L[5],L[2]+L[6]),(L[1],L[2])]
        r1 = RMaxBezier(Ls, n)
        Ls = [(L[1],L[2]),(L[1]-L[5],L[2]-L[6]),(L[1]-L[5],L[6]-L[2]),(L[1],-L[2])]
        r2 = RMaxBezier(Ls, n)
        del(Ls)
        return max(r1,r2)
    else:
        Ls = [k for k in L] # on duplique la liste L
        return RMaxBezier(Ls, n)

def RMaxBezier(L, n):
    """Renvoie la distance max à zéro de l'un des points de la courbe
    de Bézier définie par les points de contrôle de la liste L.
    Pour cela on calcule n points de la courbe."""
    rMax = sqrt(L[0][0]**2+L[0][1]**2)
    for k in range(1, n+1):
        t = k/n
        xB, yB = bezierNPt(L,t)
        r = sqrt(xB**2+yB**2)
        if  r > rMax:
            rMax = r
    return rMax

def BezierSvgN(L, t, x0=0, y0=0, mg=15):
    """Renvoie le code SVG pour l'affichage d'une courbe de Bézier
    de points définis dans la liste L de coordonnées (L est une
    liste de tuples des coordonnées des points).
    Met la courbe à l'échelle pour intégrer un carré de côte t en
    préservant une marge mg avec l'un des bords et la positionne
    aux coordonnées du repère image (x0, y0) point haut à gauche
    sans déformation de la courbe.
    """
    od = len(L)-1 # ordre de la courbe de Bézier
    assert od==2 or od==3, "Erreur le svg ne peut afficher l'ordre demandé"
    assert 2*mg < t , "Erreur la marge est trop grande  !"
    xMin, xMax, yMin, yMax = CadreBezier(L)
    a, b = xMax - xMin, yMax - yMin # dimensions du cadre origine
    if b==0 or a/b > 1:
        rh = round((t-2*mg)/a) # rapport homothétique (pixels par unité)
        ofx, ofy = x0, y0 + round(((t-2*mg)-b*rh)/2)
    else:
        rh = round((t-2*mg)/b)
        ofx, ofy = x0 + round(((t-2*mg)-a*rh)/2), y0
    L1 = [(rh*(x-xMin)+ofx+mg,rh*(yMax-y)+ofy+mg) for (x,y) in L]
    if od == 2:
        tg= " Q" # Type Bézier Quadratique
    else:
        tg = " C" # Type Bézier Cubique
    chem = "M " + str(L1[0][0]) + "," + str(L1[0][1]) + tg
    for k in range(1, od+1):
        chem += " " + str(L1[k][0]) + "," + str(L1[k][1])
    return chem

def SvgCodeFigS4N(Lpar, t, x0=0, y0=0, mg=15, pr=0):
    """Générateur de code svg pour la représentation d'une figure présentant
    les 4 axes de symétrie d'un carré. Cette figure est générée par deux courbes
    de Bézier paramétrées par l'abscisse de A (point de départ de la première
    courbe situé sur l'axe de symétrie y=x), le point B ( point de départ de la
    seconde courbe ) et deux vecteurs u et v pécisant les tangentes de la
    première courbe de Bézier en ces points. Lpar est une liste contenant
    (xA, xB, yB, xu, yu, xv, yv). La seconde courbe est totalement
    contrainte par la première et les contraintes de symétrie. L'algorithme
    met la courbe à l'échelle pour l'intégrer un carré de côte t en préservant
    une marge mg avec l'un des bords sans déformation de la figure. On peut
    choisir une meilleure précision dans le calcul des pts de contrôle (par
    défaut arrondi à l'entier) en donnant à pr le nombre de décimales.
    """
    assert len(Lpar)==7 and isinstance(Lpar[0],(int,float)) , "erreur de type de liste"
    c = 2*CadreBS4(Lpar,300,False) # coté de la croix en unité
    rh = (t-2*mg)/c
    L = PFigS4ToListePts(Lpar)
    if pr > 0:
        pr = 10**pr
        Lpp = [(int(pr*(rh*(x+c/2)+x0+mg))/pr,int(pr*(rh*(c/2-y)+y0+mg))/pr) for (x,y) in L]
    else:
        Lpp = [(int(rh*(x+c/2)+x0+mg),int(rh*(c/2-y)+y0+mg)) for (x,y) in L]
    p1 = Lpp.pop(0)
    Chem = "M" + str(p1[0]) + "," + str(p1[1])
    for k in range(len(Lpp)):
        if k%7 == 0:
            tg = "\nC "
        elif k%7 == 3 or k%7 == 5:
            tg = " S"
        else:
            tg= " "
        Chem += tg + str(Lpp[k][0]) + "," + str(Lpp[k][1])
    return Chem

def PFigS4ToListePts(L):
    """Renvoie la listes des points primitifs d'une figure S4 de paramètres
    situés dans la liste L"""
    Dep =[(L[0],L[0])]
    Ls = [(L[0]+L[3],L[0]+L[4]),(L[1]+L[5],L[2]+L[6]),(L[1],L[2]),
            (L[1]-L[5],L[6]-L[2]),(L[1],-L[2]),
            (L[0]+L[3],-L[0]-L[4]),(L[0],-L[0])]
    Lc = [k for k in Ls]
    for k in range(3):
        Lc = [(v,-u) for (u,v) in Lc]
        Ls += Lc
    del(Lc)
    return Dep+Ls

def SvgCodeS1B2B3(Lpar, t, x0=0, y0=0, mg=15, pr=0):
    """Générateur de code svg pour la représentation d'une figure présentant
    un axe de symétrie. Cette figure est générée par deux courbes de Bézier
    la première d'ordre 2 paramétrée par trois points (F, E et D), la seconde
    d'ordre 3 paramétrée par le point de départ (D) et celui d'arrivée (O) et
    leurs deux vecteurs u et v précisant les tangentes en ces points.
    Lpar liste les paramètres (xF, yF , xE, yE, xD, yD, xu, yu, xv, yv).
    L'algorithme met la courbe à l'échelle pour l'intégrer un carré de côte t
    en préservant une marge mg avec l'un des bords sans déformation. On peut
    choisir une meilleure précision dans le calcul des pts de contrôle (par
    défaut arrondi à l'entier) en donnant à pr le nombre de décimales.
    """
    assert len(Lpar)==10 and isinstance(Lpar[0],(int,float)) , "erreur de type de liste"
    Lpt = PS1B2B3ToListePts(Lpar)
    xMin1, xMax1, yMin1, yMax1 = CadreBSV(Lpt[:3])
    xMin2, xMax2, yMin2, yMax2 = CadreBSV(Lpt[2:6])
    xMin, yMin = min(xMin1,xMin2), min(yMin1, yMin2)
    xMax, yMax = max(xMax1,xMax2), max(yMax1, yMax2)
    a, b = xMax - xMin, yMax - yMin # dimensions du cadre origine
    if b==0 or a/b > 1:
        rh = (t-2*mg)/a # rapport homothétique (pixels par unité)
        ofx, ofy = x0, y0 + ((t-2*mg)-b*rh)/2
    else:
        rh = (t-2*mg)/b
        ofx, ofy = x0 + ((t-2*mg)-a*rh)/2, y0
    if pr > 0:
        pr = 10**pr
        Lpt = [(round(pr*(rh*(x-xMin)+ofx+mg))/pr,round(pr*(rh*(yMax-y)+ofy+mg))/pr) for (x,y) in Lpt]
    else:
        Lpt = [(round(rh*(x-xMin)+ofx+mg),round(rh*(yMax-y)+ofy+mg)) for (x,y) in Lpt]
    p1 = Lpt.pop(0)
    Chem = "M" + str(p1[0]) + "," + str(p1[1])
    for k in range(len(Lpt)):
        if k%8 == 0:
            tg = "Q "
        elif k == 2 or k == 5:
            tg = " C"
        else:
            tg= " "
        Chem += tg + str(Lpt[k][0]) + "," + str(Lpt[k][1])
    return Chem

def PS1B2B3ToListePts(L):
    """Renvoie la listes des points primitifs d'une figure S1B2B3 de paramètres
    situés dans la liste L"""
    assert len(L)==10, "Erreur de taille de la liste de paramètres"
    Ls = [(L[k], L[k+1]) for k in range(0,6,2)]
    Ls += [(L[4]+L[6],L[5]+L[7]),(L[8],L[9]),(0,0)]
    Lc = [k for k in Ls]
    for k in range(5):
        Ls += [(-Lc[4-k][0],Lc[4-k][1])]
    del(Lc)
    return Ls


def gridBezier(n, t=100, d=10):
    """Génère n fois n courbes de Bézier aléatoires chacune intégrée dans
    un carré de coté t pixels, les coordonnées des pts de contrôle
    sont choisis aléatoirement entre -d et d"""
    Lr=[]
    cSVG=""
    DimSvg = "<svg width=\""+str(n*t)+"\" height=\"" + str(n*t)+"\">"
    for i in range(n):
        for j in range(n):
            L = [(randint(-d,d),randint(-d,d)) for k in range(randint(3,4))]
            Lr.append(L)
            cSVG += "<path stroke-width=\"5\" stroke=\""+ ColorsDark[randint(0,19)]+"\" fill=\"none\" \nd=\""
            cSVG += BezierSvgN(L, t, i*t,j*t)+"\" />\n"
            cSVG +="<text x=\""+str(5+i*t)+"\" y=\""+str(10+j*t)+"\" >"+chr(65+i)+str(j+1)+"</text>\n"
    print(DimSvg)
    print(Lr)
    return cSVG

def gridfigS4(nl, nc, t=150, d=15):
    """Génère n fois n figures S4 aléatoires chacune intégrée dans
    un carré de coté t pixels, les coordonnées des pts de contrôle
    sont choisis aléatoirement entre -d et d"""
    cSVG=""
    DimSvg = "<svg width=\""+str(nc*t)+"\" height=\"" + str(nl*t)+"\">"
    for i in range(nc):
        for j in range(nl):
            Lpr = [randint(-d,d) for k in range(7)]
            snum = chr(65+i)+str(j+1)
            ref = snum + " : " +", ".join([ str(k) for k in Lpr])
            cSVG += "<g><path stroke-width=\"2\" stroke=\""+ ColorsDark[randint(0,19)]+"\" fill=\""+ColorsLight[randint(0,19)]+"\" \nd=\""
            cSVG += SvgCodeFigS4N(Lpr, t, i*t, j*t,10,2)+"Z\" />\n"
            cSVG +="<text x=\""+str(5+i*t)+"\" y=\""+str(10+j*t)+"\" >"+ref+"</text></g>\n"
    del(Lpr)
    print(DimSvg)
    return cSVG

def gridS1B2B3(nl, nc, t=150, d=15):
    """Génère n fois n figures S1B2B3 aléatoires chacune intégrée dans
    un carré de coté t pixels, les coordonnées des pts de contrôle
    sont choisis aléatoirement entre -d et d"""
    cSVG=""
    DimSvg = "<svg width=\""+str(nc*t)+"\" height=\"" + str(nl*t)+"\">"
    for i in range(nc):
        for j in range(nl):
            Lpr = [randint(-d,d) for k in range(10)]
            snum = chr(65+i)+str(j+1)
            ref = snum + " : " +", ".join([ str(k) for k in Lpr])
            cSVG += "<g><path stroke-width=\"2\" stroke=\""+ ColorsDark[randint(0,19)]+"\" fill=\""+ColorsLight[randint(0,19)]+"\" \nd=\""
            cSVG += SvgCodeS1B2B3(Lpr, t, i*t, j*t, 10, 3)+"Z\" />\n"
            cSVG +="<text x=\""+str(5+i*t)+"\" y=\""+str(10+j*t)+"\" >"+ref+"</text></g>\n"
    del(Lpr)
    print(DimSvg)
    return cSVG


def arrondi(x, exces):
    """Arrondi par défaut si exces = False
    Arrondi par excès si exces = True"""
    assert isinstance(exces,bool), "erreur d'argument dans arrondi"
    if abs(x-(int(x)))<10**-12: # x est alors déjà quasi entier
        return int(x)
    else:
        if x < 0:
            if exes:
                return int(x)
            else:
                return int(x)-1
        else:
            if exces:
                return int(x)+1
            else:
                return int(x)

def cruxS4(L,n=12,t=150,pas=0.25,amplit=2):
    """Génère n fois n figures S4 variant aléatoirement autour des paramètres L
    avec un pas et une amplitude donnée autour des valeurs saisies,
    chacune intégrée dans un carré de coté t pixels.
    """
    cSVG=""
    DimSvg = "<svg width=\""+str(n*t+100)+"\" height=\"" + str(n*t)+"\">"
    inc = int(amplit/(2*pas))
    for i in range(n):
        for j in range(n):
            Lpr = [k+pas*randint(-inc,inc) for k in L]
            snum = chr(65+i)+str(j+1)
            ref = snum + " : " +", ".join([ str(k) for k in Lpr])
            cSVG += "<g><path stroke-width=\"2\" stroke=\""+ ColorsDark[randint(0,19)]+"\" fill=\""+ColorsLight[randint(0,19)]+"\" \nd=\""
            cSVG += SvgCodeFigS4N(Lpr, t, i*t, j*t,10,2)+"Z\" />\n"
            cSVG +="<text x=\""+str(5+i*t)+"\" y=\""+str(10+j*t)+"\" >"+ref+"</text></g>\n"
    del(Lpr)
    print(DimSvg)
    return cSVG


# tests sur une Bézier d'ordre 4
L1=[(1,0),(3,2),(-2,4),(1,-2),(-5,3)]
assert bezierNPt(L1,0.5)==(0,1.6875), "erreur dans le programme BezierNPt"
assert CadreBezier(L1,200,False)==(-5,1.453125,0,3), "erreur dans le programme CadreBezier"
#print(CadreBezier(L1))

# tests sur une B3S1
#Lcoeur = [(0,-3),(4,0),(1,2),(0,0)]
#Lpic = [(1,-8),(-2,-2),(7,-13),(0,0)]
#Lcarreau = [(0,-3),(2,0),(2,-3),(0,0)]
#print(CadreBS1(Lcoeur,200,False))
#print(CadreBS1(Lcarreau))
#L1b=[(-1,2),(5,2),(-2,5),(3,0)]
#print(BezierSvgN(L1b, 400, 200,50))
# Les deux courbes génératrices d'une croix occitanne:
#L2 = [(1,1),(12,2),(3,6),(6,1)]
#L2p = [1, 6, 1, 11, 1, -3, 5]
#L3 = [(6,1),(9,-4),(9,4),(6,-1)]
#print(RMaxBezier(L3,100),RMaxFig(L2p))
#L2pts=PFigS4ToListePts(L2p)
#print(len(L2pts))
# croix Occitane
#print(SvgCodeFigS4(1, 6, 1, 11, 1, -3, 5, 800, 400))
# Rotor Star
#L5 = [(1,1),(-13,-6),(2,-2),(4,1)]
#L5p = [1, 4, 1, -14, -7, -2, -3]
#print(SvgCodeFigS4(1, 4, 1, -14, -7, -2, -3, 800, 400,50,3))

# Escavarotor
#L6=[(1,1),(4,-8),(10,-1),(6,-3)]
#print(SvgCodeFigS4N([1, 6, -3, 3, -9, 4, 2], 400, 200,50,3))
#print(gridBezier(7))
#print(gridfigS4(12,12))

#S1B2B3 [xF,yF,xE,yE,xD,yD,xv,yv,xu,yu]
Lcarreau = [0, -10, 0, -9, -2, -7, -5, 5, -6, -9]
Lcoeur = [0, -8, 0, -6, 2, -5, 6, 3, 4, 4]
Lpic = [1, -6, 0, -3, 1, -4, 2, -2, 4, -2]
Ltrefle = [-3, -18, 4, 0, 9, -7, -1, -10, -10, -4]

#print(PS1B2B3p7ToListePts(Ltrefle))
"""print(SvgCodeS1B2B3(Ltrefle, 200))
print((SvgCodeS1B2B3(Lcoeur, 200, 200,0)))
print((SvgCodeS1B2B3(Lpic,200, 0, 200)))
print((SvgCodeS1B2B3(Lcarreau,200, 200, 200)))
print("\n","\n")
print(gridS1B2B3(12,12))"""

Loccitv3 = [1, 5, 1, 7, 1, -2, 3]
#print(PFigS4ToListePts(Loccitv3))
#print(SvgCodeFigS4N(Loccitv3, 400,200,100,30,3))
#print(CadreBezier([(0, 5), (2, -2), (7, -2), (1, -3)] ,300,False))

Loccitint = [[-1,-2,-13,1,12,0,-1], [1,8,1,-12,-9,-10,5],   [0,1,0,-4,-1,11,4],
            [-1,-9,-4,-11,3,14,0],  [3,-6,1,-4,0,-14,-4],   [4,-10,-3,-8,-3,9,-1],
            [-7,9,-2,8,3,-10,-2],   [3,5,-1,6,9,-10,-1],    [5,0,-9,-10,0,0,-6],
            [3,-10,-1,9,-2,8,7],    [2,-9,2,8,3,9,-9],      [0,1,0,-4,-1,11,4],
            [2,9,3,1,-10,6,10],     [1,4,-13,3,-8,2,-2],    [5,14,9,-7,-1,-1,-9],
            [8,-15,-3,0,-1,9,4],    [3,-6,-14,4,3,-11,-11], [-8,-12,1,15,15,-10,-11],
            [0,8,4,8,-2,8,10],      [1,-1,2,-3,-5,9,8],     [-3,5,1,9,0,-10,3],
            [2,1,9,-1,-5,3,3],      [9,-10,1,-3,7,7,-8],    [8,2,-1,-3,-4,-14,-3],
            [-5,0,9,2,4,1,-3],      [-5,1,14,4,15,-1,-14],  [-4,-3,4,3,2,12,15],
            [-12,-2,-2,-9,-2,11,-4],[0,13,-2,8,-3,9,12],    [11,10,-4,11,11,-15,-12]]

#print(cruxS4(Loccitint[29]))
#print(SvgCodeFigS4N([-0.75,0.5,0,-4,-1.75,11.5,4.25], 400,200,100,70,3))

Lpuzl=[[5,14,9,-7,-1,-1,-9],[1, -1, 2, -3, -5, 9, 8],[-9, 5, 1, 0, 4, 7, -5],
        [5, 0, -9, -10, 0, 0, -6],[-8, -9, -3, 10, -5, 6, -8],[2,-13,11,13,-11,10,-11],
        [-9, 4, 10, -3, 1, 5, -5],[9, -9, -6, -2, -3, -1, 9],[3,1,-2,-14,3,1,9]]

# Découpage d'un rectangle en carrés
Lcoord=[(33,32),(0,0,9),(9,0,10),(19,0,14),(0,9,8),(8,9,1),(8,10,7),(15,10,4),(0,17,15),(15,14,18)]

def SvgCodS4_Dehn(LS4,Lcoo,u=30):
    cSVG=""
    a,b = Lcoo.pop(0)
    DimSvg = "<svg width=\""+str(u*a)+"\" height=\"" + str(u*b)+"\">"
    id = 0
    for Lpr in LS4:
        xi, yi, t = (u*k for k in Lcoo[id])
        ref = chr(65+id) + " : " +", ".join([ str(k) for k in Lpr])
        id+=1
        cSVG += "<g><path stroke-width=\"2\" stroke=\""+ ColorsDark[randint(0,19)]+"\" fill=\""+ColorsLight[randint(0,19)]+"\" \nd=\""
        cSVG += SvgCodeFigS4N(Lpr, t, xi, yi,10,2)+"Z\" />\n"
        cSVG +="<text x=\""+str(5+xi)+"\" y=\""+str(10+yi)+"\" >"+ref+"</text></g>\n"
    del(Lpr)
    print(DimSvg)
    return cSVG

print(SvgCodS4_Dehn(Lpuzl,Lcoord))
#print(CadreBS4(Loccitv3,300,False))
#print(SvgCodeFigS4N([-8, -9, -3, 10, -5, 6, -8], 80,230,250,5,3))