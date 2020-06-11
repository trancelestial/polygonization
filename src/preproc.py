import math
import shapely
import collections
import numpy as np
from sklearn.model_selection import train_test_split
from util import draw_hist

# Encode point set input to sequence of oriented angles and lengths
def encode_point_set(polygon):
    vertices = polygon.exterior.coords[:-1]
    langles = []
    n = len(vertices)
    v1 = (vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1])
    v2 = (vertices[-1][0] - vertices[0][0], vertices[-1][1] - vertices[0][1])
    angle = math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
    langles.append(angle)
    for i in range(1,n-1):
        v1 = (vertices[i+1][0] - vertices[i][0], vertices[i+1][1] - vertices[i][1])
        v2 = (vertices[i-1][0] - vertices[i][0], vertices[i-1][1] - vertices[i][1])
        angle = math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
        langles.append(angle)
    v1 = (vertices[0][0] - vertices[n-1][0], vertices[0][1] - vertices[n-1][1])
    v2 = (vertices[n-2][0] - vertices[n-1][0], vertices[n-2][1] - vertices[n-1][1])
    angle = math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
    langles.append(angle)
    langles = [x+2*math.pi if x<0 else x for x in langles]
    # angles in degrees
    dlangles = [angle*180/math.pi for angle in langles]
    
    ledges = []
    for i in range(1,n):
        l = math.sqrt((vertices[i][0] - vertices[i-1][0])**2 + (vertices[i][1] - vertices[i-1][1])**2)
        ledges.append(l)
    l = math.sqrt((vertices[0][0] - vertices[n-1][0])**2 + (vertices[0][1] - vertices[n-1][1])**2)
    ledges.append(l)
    # scaled
    C = sum(ledges)
    sledges = [edge/C for edge in ledges]
    if len(ledges) != len(langles):
        raise Exception("Wrong number of components, bad calculation!")
    return [langles,sledges]

def transform_data(geometry, merge_multi=False):
    polygons = []
    for geom in geometry:
        if isinstance(geom, shapely.geometry.multipolygon.MultiPolygon):
            if not merge_multi:
                for i in range(len(geom)):
                    polygons.append(encode_point_set(geom[i]))
            else:
                multi = [encode_point_set(g) for g in geom]
                polygons.append(multi)
        else:
            polygons.append(encode_point_set(geom) if not merge_multi else [encode_point_set(geom)])
    return polygons

# aids stratification by number of edges (strat not possible if there is a single polygon in class)
def fix_single_class(X, edgeno):
    '''
    
    :param X: polygon list 
    :param edgeno: number of edges for each polygon
    :return: polygon list with appended edge number columng suitable for stratification
    '''
    # conv A to list<string> to become hashable
    ahash = [str(a[0]) for a in edgeno.tolist()]
    # make hashable
    count = collections.Counter(ahash)
    # sorted key list
    scount = [int(a) for a in sorted(count, key=lambda x : int(x))]
    
    for i in range(len(edgeno)):
        if count[str(edgeno[i,0])] == 1:
            nscount = len(scount)
            si = scount.index(edgeno[i,0])
            if si == 0:
                count[str(scount[1])] += 1
                edgeno[i,0] = scount[1]
            elif si == nscount-1:
                count[str(scount[-2])] += 1
                edgeno[i,0] = scount[-2]
            elif abs(si - scount[si-1]) <= abs(si - scount[si+1]):
                count[str(scount[si-1])] += 1
                edgeno[i,0] = scount[si-1]
            else:
                count[str(scount[si+1])] += 1
                edgeno[i,0] = scount[si+1]
            count.pop(str(scount[si]))
            scount.pop(si)
            nscount-=1
    
    return np.hstack((X,edgeno))

# stratify by number of edges separately for each class
def stratify_nested(polygons, lbl):
    # split by class
    Ty_i, Fy_i  = np.where(lbl==1), np.where(lbl==0)
    Tx, Ty = polygons[Ty_i], lbl[Ty_i]
    Fx, Fy = lbl[Fy_i], polygons[Fy_i]

    # stratify train+val and test
    tX_tv, tX_test, ty_tv, ty_test = train_test_split(Tx, Ty, test_size=0.2, stratify=Tx[:,3])
    fX_tv, fX_test, fy_tv, fy_test = train_test_split(Fx, Fy, test_size=0.2, stratify=Fx[:,3])
    X_tv = np.vstack((tX_tv,fX_tv))
    X_test = np.vstack((tX_test,fX_test))
    y_tv = np.concatenate((ty_tv,fy_tv))
    y_test = np.concatenate((ty_test,fy_test))

    # stratify train val
    tX_train, tX_val, ty_train, ty_val = train_test_split(tX_tv, ty_tv, test_size=0.2, stratify=tX_tv[:,3])
    fX_train, fX_val, fy_train, fy_val = train_test_split(fX_tv, fy_tv, test_size=0.2, stratify=fX_tv[:,3])
    X_train = np.vstack((tX_train,fX_train))
    X_val = np.vstack((tX_val,fX_val))
    y_train = np.concatenate((ty_train,fy_train))
    y_val = np.concatenate((ty_val,fy_val))

    return X_train, X_val, X_test, X_test, y_train, y_val, y_test



