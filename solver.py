import networkx as nx
import random
from networkx.algorithms import approximation as nxapprox
from parse import read_input_file, write_output_file, read_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
from heapq import heappush, heappop, heapify
from itertools import count
from statistics import median, mean
import sys
import os

#Corresponds to regular dijkstras
DEFAULT_PARAMS = [1,0,0,0]

def heuristic_explore():
	"""
	Explores a relatively large space of heuristics based on parameters.
	"""
	vals = [0,1,3]
	results = {}
	for p1 in vals:
		for p2 in vals:
			for p3 in vals:
				for p4 in vals:
					results[[p1,p2,p3,p4]] = heuristic_loss([p1,p2,p3,p4])
	for params in results:
		print(str(params)+ ": " +results[params])

def coordinate_descent(iters=10):
	"""
	Simple form of coordinate descent to improve parameters for heuristic.
	Loss function not differentiable so convergence is done by shrinking steps over time.
	"""
	#Starts with regular dijkstra's
	params = [1,0,0,0]
	params_step = [1,1,1,1]
	results = {}
	for k in range(iters):
		for i in range(len(params)):
			#Stores best value for step, defaulting to 0
			Step, Loss = 0, 0
			if params in results:
				Loss = results[params]
			else:
				Loss = heuristic_loss(params)
			for step in [-1,-3/4,-1/2,-1/4,1/4,1/2,3/4,1]:
				update = params.copy()
				update[i] = params[i]+step*params_step[i]
				#Negative parameters are not considered
				if update[i] >= 0:
					#Only recalculates if not already known
					loss = 0
					if update in results:
						loss = results[update]
					else:
						loss = heuristic_loss(update)
						results[update] = loss
					if loss < Loss:
						Step = step
						Loss = loss
			#Step
			params[i] = params[i]+Step*params_step[i]
			#If step was small, reduce step size
			if abs(Step) <= 1/2:
				params_step[i] /= 2
	sorted_results = sorted(results.keys(), key = lambda p: results[p])
	for params in sorted_results:
		print(str(params)+ ": " +results[params])
			
def heuristic_loss(params):
	"""
	General loss function to replace if I want to try some other loss.
	"""
	return heuristic_avg_cost(params)
			
def heuristic_avg_cost(params):
	"""
	Loss function of average cost over training graphs.
	"""
	total_cost, count = 0, 0
	for input in os.listdir('training/'):
		G = read_input_file('training/'+input)
		T = dijkstra_two_solve(G,[p1,p2,p3,p4])
		assert is_valid_network(G, T)
		total_cost += average_pairwise_distance_fast(T)
		count += 1
	return total_cost/count
	
def evolving_trees_solve(G,iters=100):
	"""
	Generates trees by random mutations to dijkstra_two trees
	Initial probabilities seeded by shortest path trees
	Uses temp parameter as an idea from simulated annealing to determine chance of taking locally bad decisions
	"""
	temp = 0.33
	#important to make sure that indexing remains consistent
	edge_index, index_edge, count = {}, [0 for _ in range(len(G.edges))], 0
	for e in G.edges():
		edge_index[e] = count
		index_edge[count] = e
		count += 1
	#Weights will change multiplicatively
	P = {edge_index[e]:1 for e in G.edges()}
	Trees = dijkstra_two_solve(G,k=5,all_trees=True)
	for T in Trees:
		#To prevent errors later in code from dividing by cost and because no improvement possible
		if average_pairwise_distance_fast(T) == 0:
			return T
		for e in T.edges():
			P[edge_index[e]] *= 2
	#Choose initial starting tree
	T = min(Trees, key=lambda t: average_pairwise_distance_fast(t))
	E = [edge_index[e] for e in T.edges()]
	cost = average_pairwise_distance_fast(T)
	for i in range(iters):
		#Chooses edge index to remove inversely proportional to weight
		remove_prob = {i:1/P[i] for i in P if i in E}
		trim_index = random.choices(list(remove_prob.keys()), weights = [1/p for p in remove_prob.values()])
		E.remove(trim_index)
		
		#Chooses an edge to reconncet tree
		components_iter = nx.connected_components(T)
		V1 = components_iter.next()
		V2 = components_iter.next()
		cross_edges = [(u,v) for (u,v) in G.edges() if (u in V1 and v in V2) or (u in V2 and v in V1)]
		cross_prob = {i:P[i] for i in P if index_edge[i] not in E}
		add_index = random.choices(list(cross_prob.keys()), weights = cross_prob.values())
		E.add(add_index)
		
		#Create new tree
		T2 = G.edge_subgraph([index_edge[i] for i in E])
		cost2 = average_pairwise_distance_fast(T2)
		#Percentage performance change used to penalize chance of making locally bad decision
		err = (cost2 - cost)/cost 
		if err < 0 or random.random() + err < temp:
			#update weights
			P[trim_index] /= 2
			P[add_index] = min(P[add_index]*2,1024)
			cost = cost2
			T = T2
		else:
			E.add(trim_index)
			E.remove(add_index)
			P[trim_index] = min(P[add_index]*2,1024)
			P[add_index] /= 2 
	return T
		
		
		
		
					
def dijkstra_two_solve(G,params=DEFAULT_PARAMS,k=2,all_trees=False):
	"""
	Wrapper method that generats solutions using dijkstra_two
	Args:
		G: nx.Graph()
		params: allows Dijkstras to account for several properties of edges
		k: tries upto k source vertices if k distinct vertices exist
		all_trees: Flags whether to return best tree or all of them
	Returns:
		If all_trees, then it returns an array of all the generated trees.
		Else, it returns the single best shortest path tree.
	"""
	n = len(G.nodes())
	k = min(n,k)
	#Trivial cases that otherwise error
	if n==1:
		return G
	elif n==2:
		T = nx.Graph()
		T.add_node(0)
		return T
	#Uses dijkstras to find metric closure distances
	D = [[0 for u in G.nodes()] for v in G.nodes()]
	all_paths_iter = nx.all_pairs_dijkstra(G)
	for u, (distance, path) in all_paths_iter:
		for v in distance:
			D[u][v] = distance[v]
	avgD = [mean(d) for d in D]
	starts = sorted(G.nodes(), key= lambda v: avgD[v])[0:k]
	if all_trees:
		Trees = []
		for s in starts:
			T = dijkstra_two(G,s,D=D,params=params)
			cost = average_pairwise_distance_fast(T)
			Trees.append(T)
		return Trees
	else:
		minval, mintree = 1000000, None
		for s in starts:
			T = dijkstra_two(G,s,D=D,params=params)
			cost = average_pairwise_distance_fast(T)
			if cost < minval:
				minval = cost
				mintree = T
		return mintree
	
def dijkstra_two(G,source,targets=None,D=None,params=DEFAULT_PARAMS):
	"""
	Lots of credit to the original networkx dijkstras
	Args:
		G: nx.Graph()
		source: starting vertex
		targets: Optional list of vertices it will stop after finding. By default will be all vertices
		D: optional 2D array of distances between vertices
		params: allows for heuristic costs rather than standard dijkstra
	Returns:
		Shortest path tree as an nx.Graph()
	"""
	def weight(x,y,d,S):
		"""
		Heuristic cost of adding edge x to y with weight d['weight'] having seen vertices S before
		"""
		if D == None:
			return d['weight'] 
		else:	
			w = d['weight'] * params[0]
			w-= G.degree[v]* params[1]
			w+= mean([D[y][v] for v in S]) * params[2]
			w+= median(D[y]) * params[3]
			return max(0,w)
	if targets == None:
		targets = list(G.nodes())
	pred = {} # dictionary of predessors
	dist = {}  # dictionary of final distances
	seen = {}
	# fringe is heapq with 3-tuples (distance,c,node)
	# use the count c to avoid comparing nodes
	c = count()
	fringe = []
	#Single source
	seen[source] = 0
	#Tracks which of targets is not seen
	not_seen = targets.copy()
	heappush(fringe, (0, next(c), source))
	while fringe and len(not_seen) > 0:
		(d, _, v) = heappop(fringe)
		if v in dist:
			continue  # already searched this node.
		dist[v] = d
		for u, e in G.adj[v].items():
			cost = weight(v, u, e, seen)
			if cost is None:
				continue
			vu_dist = dist[v] + cost
			if u in dist:
				if vu_dist < dist[u]:
					raise ValueError('Contradictory paths found:',
									 'negative weights?')
			elif u not in seen or vu_dist < seen[u]:
				seen[u] = vu_dist
				if u in not_seen:
					not_seen.remove(u)
				heappush(fringe, (vu_dist, next(c), u))
				if pred is not None:
					pred[u] = v
	E = []
	targeted = set()
	while len(targets) > 0:
		v = targets.pop()
		targeted.add(v)
		if v in pred:
			E.append((v,pred[v]))
			if pred[v] not in targeted:
				targets.append(pred[v])
	if len(E) == 0:
		T = nx.Graph()
		T.add_node(source)
		return T
	return G.edge_subgraph(E)

def dijkstra_solve(G, starts=None, targets=None):
	"""
	Finds shortest path trees from each start to targets and trims them. By default goes from all to all.
	"""
	n = len(G.nodes())
	#Trivial cases that otherwise error
	if n==1:
		return G
	elif n==2:
		T = nx.Graph()
		T.add_node(0)
		return T
	minval, mintree = 1000000, None
	if start == None:
		start = G.nodes()
	if targets == None:
		targets = G.nodes()
	for v in starts:
		T = dijkstra_tree(G,v, targets)
		Tn = trim_tree(G,T)
		if average_pairwise_distance_fast(Tn) < minval:
			#print("Best is now rooted at " + str(v))
			#print(is_valid_network(G,T))
			mintree = Tn
			minval = average_pairwise_distance_fast(Tn)
	return mintree
	
def mst_solve(G):
	"""
	Uses prims to find a minimum spanning tree and then trims it.
	Prims performs much better than kruskal's.
	"""
	T = T=nx.minimum_spanning_tree(G, algorithm='prim')
	Tn = trim_tree(G,T)
	return Tn
		
	
def dijkstra_tree(G,v, targets = None):
	return dijkstra_two(G,v,targets=targets,params=[1,0,0,0])
		
	
def pre_trim_dijkstra(G):
	"""
	Runs dijkstra solve attempting to target a trimmed set of vertices
	"""
	D = pre_trimming(G)
	if len(D) == 1:
		T = nx.Graph()
		T.add_nodes_from(D)
		return T
	return dijkstra_solve(G,D,D)
	
		
def pre_trimming(G):
	"""
	Returns a dominating set greedily pruned by removing vertices with high median distance to other vertices.
	Updates distances as vertices are pruned so you only need the max each time rather than succesive mins.
	"""
	dist = {v:{} for v in G.nodes()}
	all_paths_iter = nx.all_pairs_dijkstra(G)
	for u, (distance, path) in all_paths_iter:
		for v in distance:
			dist[u][v] = distance[v]
	#current sub vertices
	sub_nodes = set(G.nodes())
	median_dist = {v:median(dist[v].values) for v in sub_nodes}
	#Only to track remaining attempts
	Q = list(G.nodes())
	while len(Q) > 0:
		x = max(Q, key = lambda v: median_dist[v])
		Q.remove(x)
		sub_nodes.remove(x)
		if not nx.is_dominating_set(G,sub_nodes):
			sub_nodes.add(x)
		else:
			dist.pop(x)
			for v in dist:
				dist[v].pop(x)
			median_dist = {v:median(dist[v]) for v in sub_nodes}
	return sub_nodes

def pre_trimming_fast(G):
	"""
	Returns a dominating set greedily pruned by removing vertices with high median distance to other vertices.
	Faster by using priority queue and using original distances rather than update as vertices are pruned. Sorting would work here too.
	"""
	n = len(G.nodes())
	dist = [[0 for _ in range(n)] for _ in range(n)]
	all_paths_iter = nx.all_pairs_dijkstra(G)
	for u, (distance, path) in all_paths_iter:
		for v in distance:
			dist[u][v] = distance[v]
	sub_nodes = set(G.nodes())
	#Negated because python heaps are min heaps
	Q = [(-median(dist[v]),v) for v in sub_nodes]
	heapify(Q)
	while len(Q) > 0:
		d,v = heappop(Q)
		sub_nodes.remove(v)
		if not nx.is_dominating_set(G,sub_nodes):
			sub_nodes.add(v)
	return sub_nodes
	
	
	
def trim_tree(G,T):
	"""
	Trims tree T as solution for graph G.
	Valid solutions for the problem are trees whose vertices are dominating sets.
	"""
	leaves = [l for l in T.nodes() if T.degree()[l]==1]
	subTreeNodes = list(T.nodes())
	cost = average_pairwise_distance_fast(T)
	while len(leaves) > 0:
		l, w = leaves[0],100
		for v in leaves:
			for u in T.adj[v]:
				if T.adj[v][u]['weight'] <= w:
					l, w = v, T.adj[v][u]['weight']
		leaves.remove(l)
		toCheck = T.adj[l]
		subTreeNodes.remove(l)
		Tn = T.subgraph(subTreeNodes)
		if len(subTreeNodes) > 0 and is_valid_network(G, Tn):
			costn = average_pairwise_distance_fast(Tn)
			if costn < cost:
				leaves += [v for v in toCheck if Tn.degree[v] == 1]
				T = Tn
				cost = costn
			else:
				subTreeNodes.append(l)
		else:
			subTreeNodes.append(l)
	return T
				
def random_steiner_tree(G):
	D = set()
	#remainder of vertices
	R = list(G.nodes())
	W = [G.degree[v] for v in remainder]
	while not nx.is_dominating_set(G,D):
		x = random.choices(R, weights=W)
		D.add(x)
	return dijkstra_solve(G,D,D)				

def dom_set_solver(G):
	"""
	Finds a smallish dominating set and uses Dijkstra's to find a tree on it.
	Superceded by pre_trimming
	"""
	D = nxapprox.dominating_set.min_weighted_dominating_set(G)
	return dijkstra_solve(G,D,D)

def weighted_dom_set_solver(G):
	"""
	Adds wieghts to vertices based on weight of lightest edges, as there are only a few vertices
	Superceded by pre_trimming
	"""
	base = 0
	for v in G.nodes():
		sortAdj = sorted([G.adj[v][i]['weight'] for i in G.adj[v]])
		#sortAdj = sorted(adj, key=lambda x: adj.get(x)['weight'])
		if len(sortAdj) > 1:
			G.nodes[v]['weight'] = base+150 - sortAdj[0] - sortAdj[1]/2
		else:
			G.nodes[v]['weight'] = base+100 - sortAdj[0]
	D = nxapprox.dominating_set.min_weighted_dominating_set(G)
	return dijkstra_solve(G,D,D)
	
def grow_dom_set_solver(G):
	"""
	Generates a dominating set by adding vertices with small median distance to all other vertices
	"""
	n = len(G.nodes())
	if n < 10:
		return dijkstra_solve(G)
	dist = [[0 for i in range(n)] for i in range(n)]
	all_paths_iter = nx.all_pairs_dijkstra(G)
	for u, (distance, path) in all_paths_iter:
		for v in distance:
			dist[u][v] = distance[v]
	median_dist = [median(dist[v]) for v in range(n)]
	#current sub vertices
	sub_nodes = set(G.nodes())
	#queue for vertices, sorted in ascending order I guess
	Q = sorted(range(n), key=lambda v: median_dist[v])
	while len(Q) > 0:
		v = Q.pop()
		sub_nodes.remove(v)
		if not nx.is_dominating_set(G,sub_nodes):
			sub_nodes.add(v)
	assert nx.is_dominating_set(G,sub_nodes)
	if len(sub_nodes) == 1:
		T = nx.Graph()
		T.add_nodes_from(sub_nodes)
		return T
	else:
		return dijkstra_solve(G, sub_nodes, sub_nodes)

def dijkstra_tree_old(G,v, targets = None):
	"""
	Deprecated method for finding shortest path tree
	"""
	edgeSet = set()
	T = nx.Graph()
	T.add_nodes_from(range(len(G.nodes())))
	dists, paths = nx.single_source_dijkstra(G, v)
	if targets != None:
		paths = {v:paths[v] for v in paths if v in targets}
	for p in paths.values():
		for i in range(len(p)-1):
			edgeSet.add((p[i],p[i+1]))
	T.add_edges_from(edgeSet)
	for e in T.edges():
		T.edges[e]['weight'] = G.edges[e]['weight']
	#remove vertices of degree 0 because they're not targets
	if targets != None:
		T.remove_nodes_from([v for v in T.nodes() if T.degree()[v] == 0])
	return T