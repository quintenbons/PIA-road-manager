def getLength(pos1:tuple[float, float], pos2:tuple[float, float]) -> float:
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5


def circle_collision(pos1, pos2, size1, size2):
    dist = getLength(pos1, pos2)
    if dist < size1 + size2:
        return True
    return False

def norm(v) -> float:
    return (v[0]**2+v[1]**2)**0.5

def vecteur(pos1, pos2):
    return (pos2[0] - pos1[0], pos2[1] - pos1[1])

def vecteur_norm(pos1, pos2):
    vec = vecteur(pos1, pos2)
    n = norm(vec)
    return (vec[0]/n, vec[1]/n)

def scalaire(u, v):
    return u[0]*v[0] + u[1]*v[1]

# def possible_collision(src1, dest1, src2, dest2, size1, size2):
#     u = normalized(src1, dest1)
#     v = normalized(src2, dest2)

#     p1, p2 = boxes_point(src1, size1, u)
#     q1, q2 = boxes_point(src2, size2, v)

#     #TODO factoriser le code dessous

#     a1, b1, c1 = linear_function(p1, u)
#     a2, b2, c2 = linear_function(p2, u)
    
#     a3, b3, c3 = linear_function(q1, v)
#     a4, b4, c4 = linear_function(q2, v)
    
#     A = intersec(a1, b1, c1, a3, b3, c3)
#     B = intersec(a1, b1, c1, a4, b4, c4)

#     C = intersec(a2, b2, c2, a3, b3, c3)
#     D = intersec(a2, b2, c2, a4, b4, c4)

#     vec1 = normalized(p1, A)
#     vec2 = normalized(p1, B)
#     vec3 = normalized(p2, C)
#     vec4 = normalized(p2, D)

#     wec1 = normalized(q1, A)
#     wec2 = normalized(q1, C)
#     wec3 = normalized(q2, B)
#     wec4 = normalized(q2, D)

#     sca1 = transf_scalaire(vec1, u)
#     sca2 = transf_scalaire(vec2, u)
#     sca3 = transf_scalaire(vec3, u)
#     sca4 = transf_scalaire(vec4, u)

#     tca1 = transf_scalaire(wec1, v)
#     tca2 = transf_scalaire(wec2, v)
#     tca3 = transf_scalaire(wec3, v)
#     tca4 = transf_scalaire(wec4, v)

#     l_sca = [sca1, sca2, sca3, sca4]
#     l_tca = [tca1, tca2, tca3, tca4]

#     if 1 in l_sca and -1 in l_sca:
#         pass
#     elif -1 in l_sca:
#         pass


# def scalaire(a, b):
#     return a[0]*b[0]+a[1]*b[1]

# def transf_scalaire(a, b):
#     sca = scalaire(a, b)
#     if sca > 0:
#         return 1
#     else:
#         return -1
# def boxes_point(point, size, u):
#     ux, uy = u
#     p1 = (point[0] - size/2*uy + size*ux, point[1] + size/2*ux + size*uy)
#     p2 = (point[0] + size/2*uy + size*ux, point[1] - size/2*ux + size*uy)

#     return p1, p2

# def linear_function(point, vec):
#     # return ax+by+c = 0
#     ux, uy = vec
#     x, y = point
#     if ux == 0:
#         a, b, c = 1, 0, -x
#     else:
#         a = uy/ux
#         b = -1
#         c = y - a*x

#     return a, b, c

# def intersec(a1, b1, c1, a2, b2, c2):
#     #a1*x+b1*y+c1 = 0
#     #a2*x+b2*y+c2 = 0

#     if(b1 == 0 and b2==0):
#         return float("inf")
#     if(b1 == 0):
#         #a1*x+c1=0
#         #a2*x+b2*y+c2=0

#         #x=-c1/a1
#         #-c1*a2/a1+b2*y+c2=0
#         #b2*y+c2=c1*a2/a1
#         #y=(c1*a2/a1-c2)/b2
#         return -c1/a1, (c1*a2/a1-c2)/b2
#     if(b2 == 0):
#         return -c2/a2, (c2*a1/a2-c1)/b1
#     if(a1 == a2):
#         return float("inf")

#     #a1*x+b1*y+c1=0
#     #a2*x+b2*y+c2=0

#     #a1*x+c1=-b1*y
#     #a2*x+c2=-b2*y

#     #-a1/b1*x-c1/b1=y
#     #-a1/b2*x-c2/b2=y
#     a3, b3 = -a1/b1, -c1/b1
#     a4, b4 = -a2/b2, -c2/b2
#     #a3*x+b3=y
#     #a4*x+b4=y

#     #a3*x+b3=a4*x+b4
#     #(a3-a4)*x=(b4-b3)
#     x=(b4-b3)/(a3-a4)
    
#     return x, a3*x+b3

# def normalized(src, dest):
#     ux, uy = dest[0] - src[0], dest[1] - src[1]

#     n_u = (ux**2+uy**2)**0.5
#     ux /= n_u
#     uy /= n_u

#     return ux, uy

# if __name__ == "__main__":
#     #small tests
#     possible_collision((0, 20), (80, 0), (0, 40), (80, 0), 5, 5)