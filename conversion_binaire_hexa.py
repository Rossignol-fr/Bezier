# Créé par Tony Paintoux, le 23/11/2020 en Python 3.40
Dbibi=['HO', 'HA', 'HE', 'HI', 'BO', 'BA', 'BE', 'BI',
    'KO', 'KA', 'KE', 'KI', 'DO', 'DA', 'DE', 'DI']
def base2dec(s,b):
    """Converti un nombre écrit en base b en décimal (un entier)
    Le nombre s est sous forme de chaîne de caractères"""
    Ls, l = format_base(s,b)
    Lp = [ int(Ls[k])*b**k if ord(Ls[k])<65 else (ord(Ls[k])-55)*b**k for k in range(l)]
    return sum(Lp)


def format_base(s,b):
    """Verifie si les caractères contenus dans la chaîne de caractères sont
    bien tous dans la base b, et renvoie la listes de ces caractères"""
    assert isinstance(s,str), "erreur d'argument on attends une chaîne de caractères"
    assert b < 37, "La base considérée doit être inférieure à 37"
    s = s.upper()
    Lb = [ str(k) if k <10 else chr(55+k) for k in range(b)]
    sb = sum([ 0  if k in Lb else 1 for k in s])
    assert sb==0, "des caractères ne sont pas dans la base b"
    l = len(s)
    Ls = [ s[l-k-1] for k in range(l)]
    return Ls, l


def dec2base(n,b):
    """Converti l'entier n dans la base b choisie, renvoie une chaîne de
    caractères. La base b doit être inférieure à 37 (0 -> 9, A -> Z)"""
    assert b < 37, "La base considérée doit être inférieure à 37"
    if n<b:
        if n<10:
            return str(n)
        else:
            return chr(55+n)
    else:
        Lr =[]
        r = n
        nch = 1 # Variable contenant le nombre de chiffres necéssaires
        while n>=b**nch:
            nch+=1
        for k in range(nch):
            p = nch - k - 1 # puissance courante de la base
            if r < b**p:
                Lr.append(0)
            else:
                q = r//(b**p)
                Lr.append(q)
                r -= q*b**p
        Ls = [ str(k) if k <10 else chr(55+k) for k in Lr]+["0"]*p
        return "".join(Ls)

def base2base(s1,b1,b2):
    """Converti un nombre codé en b1 dans la base b2"""
    return dec2base(base2dec(s1,b1),b2)

def NewBase(s):
    """Défini une nouvelle base en s'appuyant sur les caractères contenus
    dans la chaîne de caractère et en renvoie le codage"""
    assert len(s)>1 and s[0]!=s[1], "Il doit y avoir au moins deux caractères !"

def dec2bibi(n):
    """Renvoie le code bibi-binaire (Boby Lapointe) du nombre n"""
    codeH = dec2base(n,16)
    Lb = [Dbibi[base2dec(k,16)] for k in codeH]
    return "".join(Lb)


# --- Tests unitaires
s1="424d"
s2="3A41"
s3="413A"
assert base2dec(s1, 16) == 16973, "Problème de conversion décimale!"
assert base2dec(s2, 16) == 14913, "Problème de conversion décimale!"
assert base2dec(s3, 16) == 16698, "Problème de conversion décimale!"

n1 = 1789
n2 = 6021976
n3 = 251659024
assert dec2base(n1,16) == '6FD', "Problème de conversion dans la base!"
assert dec2base(n1,2) == '11011111101', "Problème de conversion dans la base!"
assert dec2base(n3,16) == 'F000310', "Problème de conversion dans la base!"

# --- Exemple d'utilisation
for k in range(2,37):
    print(n2, " s'écrit en base "+str(k)+":", dec2base(n2,k))