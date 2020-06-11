import matplotlib.pyplot as plt
import shapely.wkt as sw

def test_dim(testlist, dim=0):
   if isinstance(testlist, list):
      if testlist == []:
          return dim
      dim = dim + 1
      dim = test_dim(testlist[0], dim)
      return dim
   else:
      if dim == 0:
          return -1
      else:
          return dim

def draw_hist(list, num_bins=100, color='red', xlabel='', ylabel=''):
    '''
    
    :param list: number list of shape (n, 1)
    :param num_bins: number of bins for histogram
    :param color: histogram color
    :param xlabel: x axis label
    :param ylabel: y axis label
    :return: void
    '''
    n, bins, patches = plt.hist(list, num_bins, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# loading data in *.wkt format
def load_wkt(path, predef_class=None):
    geometry = []
    labels = []
    with open('path', 'r') as f:
        for line in f:
            pol, lbl = line.rstrip().split('\t') if predef_class == None else line.rstrip(), predef_class 
            geometry.append(sw.loads(pol))
            labels.append((0 if lbl == 'bad' else 1) if predef_class == None else predef_class)
    return geometry, labels

def test_angle_edge_consist(geometry, polygons):
    n_check = [len(g.exterior.coords[:-1]) for g in geometry]
    assert len(n_check) == len(polygons), 'Edges number of POLYGONS doesn\'t match!'
    for i in range(len(polygons)):
        assert len(polygons[i][0]) == len(polygons[i][1]), f'Angle and edge numers don\'t match at index {i}!'
        assert len(polygons[i][0]) == n_check[i], f'Angle numers don\'t match at index {i}!'
