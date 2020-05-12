import random
class Graph:
	"""
	Simple graph with adjacency list that initializes from files.
	"""
	def __init__(self, filename):
		f = open(filename, 'r')
		l = f.readline()
		self.n = int(l)
		self.V = list(range(self.n))
		self.E = [{} for i in range(self.n)]
		e = f.readline().split()
		while e:
			u, v, w = int(e[0]), int(e[1]), float(e[2])
			self.E[u][v] = w
			self.E[v][u] = w
			e = f.readline().split()
		f.close()
		

def generate(n, k, d = 0, fname = None):
	"""
	Makes a random tree on n vertices and attempts to add k edges with maximum degree of any vertex as d
	"""
	assert d >= 2 + k/n, "invalid degree limit"
	deg = [0 for _ in range(n)]
	V = list(range(n))
	C = [0]
	U = list(range(1,n))
	E = [{} for i in range(n)]
	v1, v2, w = 0, 0, 0
	for _ in range(n+k):
		degs = degrees(E)
		if len(C) < n:
			(v1, v2, w) = chooseValidEdge(C,U,degs,d)
			C.append(v2)
			U.remove(v2)
		else:
			(v1, v2, w) = chooseValidEdge(V,V,degs,d)
		E[v1][v2] = w
		E[v2][v1] = w
	if fname != None:
		f = open(fname,'w')
		f.write(str(n))
		for i in range(n):
			for j in range(i+1,n):
				if j in E[i]:
					f.write('\n'+str(i) + ' ' + str(j) + ' ' + str(E[i][j]))
		f.close()
	return E
	
def chooseValidEdge(V1,V2,degs,d):
	"""
	Args:
		V1, V2: Iterables of vertices (ints)
		degs: Array of degrees of vertices
		d: degree limit to obey
	Returns:
		A 3-tuple for a weighted edge
	"""
	v1 = random.choice(V1)
	v2 = random.choice(V2)
	while v1 == v2 or degs[v1] >= d or degs[v2] >= d:
		v1 = random.choice(V1)
		v2 = random.choice(V2)
	w = round(random.uniform(0,100),3)
	while w == 0 or w == 100:
		w = round(random.uniform(0,100),3)
	return (v1, v2, w)
	
def degrees(E):
	return [len(e) for e in E]