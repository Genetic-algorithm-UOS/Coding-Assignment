from copy import deepcopy
from math import prod
import numpy as np
import matplotlib.pyplot as plt



numeral = lambda x: any(isinstance(x, c) for c in (int, float, complex, bool))



class View:
    DEFAULT_ATTRIBUTES = ['type', 'reference', 'mode', 'graph']
    THRESHOLD = 1e-4

    def __init__(self, mode, graph):
        self.mode = mode
        self.graph = graph
        self.type = type(self.reference)
        assert any(isinstance(self.reference, structure) for structure in (dict, list, set, tuple))


    def __iter__(self):
        if self.mode == 'node':
            for node in self.graph._nodes:
                yield node
        elif self.mode == 'edge':
            visited = []
            for n_i in self.graph.nodes:
                for n_j in self.graph.nodes:
                    edge = self.graph[n_i, n_j]
                    if edge and edge not in visited:
                        visited.append(edge)
                        yield edge





    def __getitem__(self, key):
        return self.reference[key]


    def __setitem__(self, key, val):
        self.reference[key] = val


    def __getattr__(self, attr):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if isinstance(self.reference, dict):
                if attr not in self.reference:
                    self.graph._registery(mode=self.mode, attr=attr)
                return self.reference[attr]
            else:
                if attr not in self.reference:
                    self.graph._registery(mode=self.mode, attr=attr)
                return self.type(getattr(item, attr) for item in self.reference)
        elif attr == 'reference':
            if self.mode == 'node':
                return self.graph._node_values
            elif self.mode == 'edge':
                return self.graph._edge_values
        else:
            return super().__getattr__(attr)


    def __setattr__(self, attr, val):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if isinstance(self.reference, dict):
                if numeral(val):
                    self.reference[attr] = np.full_like(self.value, val).astype(type(val))
                elif isinstance(val, np.ndarray):
                    if not self.graph.bidirectional and np.sum(np.abs(val - val.T))/prod(val.shape) > self.THRESHOLD:
                        raise Exception("The graph is undirected, but the np.matrix is not transposition invariant!")
                    self.reference[attr] = val
                elif self.mode == 'node' and (isinstance(val, tuple) or isinstance(val, list)) and len(val) == len(self.graph):
                    self.reference[attr] = val
                else:
                    if self.mode == 'node':
                        self.reference[attr] = [deepcopy(val) for _ in self.graph]
                    elif self.mode == 'edge':
                        self.reference[attr] = [[deepcopy(val) for _ in self.graph] for _ in self.graph]
            else:
                for item in self.reference:
                    setattr(item, attr, deepcopy(val))
        else:
            if numeral(val):
                super().__setattr__(attr, np.full_like(self.value, val).astype(type(val)))
            else:
                super().__setattr__(attr, val)


    def __str__(self):
        if self.mode == 'node':
            return 'Nodes[' + ', '.join(str(node) for node in self.graph) + ']'
        elif self.mode == 'edge':
            return 'Edges[' + ', '.join(str(self.graph[a, b]) for a in self.graph for b in self.graph if self.graph[a, b]) + ']'


    def __repr__(self):
        return 'View ' + str(self)



class Node:
    DEFAULT_ATTRIBUTES = ['graph', 'reference', 'name']

    def __init__(self, graph, reference, name=None):
        self.graph = graph
        self.reference = reference
        self.name = name or reference


    def __iter__(self):
        for node in self.graph:
            if self.graph[self, node]:
                yield node


    def __getattr__(self, attr):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if attr not in self.graph._node_values:
                self.graph._registery(mode='node', attr=attr)
            return self.graph._node_values[attr][self.reference]
        else:
            super().__getattr__(attr)
    

    def __setattr__(self, attr, val):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if attr not in self.graph._node_values:
                self.graph._registery(mode='node', attr=attr, val=val)
            self.graph._node_values[attr][self.reference] = val
        else:
            super().__setattr__(attr, val)


    def __repr__(self):
        if self.graph._node_values:
            return f"Node⟨{', '.join(key + ':' + str(getattr(self, key)) for key in self.graph._node_values.keys() if not key.startswith('_'))}⟩"
        else:
            return f"⟨{self.name}⟩"


    def __str__(self):
        return f"⟨{self.name}⟩"


    def __int__(self):
        return self.reference


    def __eq__(self, other):
        return self.reference == other.reference and self.graph == other.graph


    def __hash__(self):
        return hash(str(self))



class Edge:
    DEFAULT_ATTRIBUTES = ['graph', 'source', 'target']

    def __init__(self, graph, source, target):
        self.graph = graph
        self.source = source
        self.target = target


    def __getattr__(self, attr):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if attr not in self.graph._edge_values:
                self.graph._registery(mode='edge', attr=attr)
            try:
                return self.graph._edge_values[attr][int(self.source), int(self.target)]
            except:
                return self.graph._edge_values[attr][int(self.source)][int(self.target)]
        else:
            super().__getattr__(attr)


    def __setattr__(self, attr, val):
        if attr not in self.DEFAULT_ATTRIBUTES:
            if attr not in self.graph._edge_values:
                self.graph._registery(mode='edge', attr=attr, val=val)
            try:
                self.graph._edge_values[attr][int(self.source), int(self.target)] = val
                if not self.graph.bidirectional:
                    self.graph._edge_values[attr][int(self.target), int(self.source)] = val
            except:
                self.graph._edge_values[attr][int(self.source)][int(self.target)] = val
                if not self.graph.bidirectional:
                    self.graph._edge_values[attr][int(self.target)][int(self.source)] = val
        else:
            super().__setattr__(attr, val)


    def __repr__(self):
        return f"Edge{str(self)} ⟨{', '.join(key + ':' + str(getattr(self, key)) for key in self.graph._edge_values.keys() if not key.startswith('_'))}⟩"


    def __str__(self):
        return f"{str(self.source)}⟝{str(self.target)}"


    def __eq__(self, other):
        directed = self.source == other.source and self.target == other.target and self.graph == other.graph
        undirected = self.source == other.target and self.target == other.source
        return directed or not self.graph.bidirectional and undirected


    def __hash__(self):
        return hash(str(self))



class Graph:
    def __init__(self, nodes=None, edges=None, default_value=0, bidirectional=False):
        """
        creates an instance of a Graph
        :param nodes: int, np.ndarray
        :param edges: None, np.ndarray
        :param default_value:
        """
        self.default_value = default_value
        self.bidirectional = bidirectional
        self._node_values = {}
        self._edge_values = {}
        # For visualization:
        self.bg_img = plt.imread("osm_germany.png")
        # CONSTRUCTION
        if nodes is not None:
            if isinstance(nodes, int):
                self._nodes = [Node(self, n) for n in range(nodes)]
                self._edges = np.full((len(nodes), len(nodes)), default_value)
            elif isinstance(nodes, np.ndarray):
                if isinstance(edges, np.ndarray):
                    if nodes.shape + nodes.shape == edges.shape:
                        self._nodes = nodes
                        self._edges = edges
                    else:
                        raise Warning(f"For nodes of shape (n,) edges must be of shape (n, n). Got {nodes.shape} and {edges.shape} instead!")
                else:
                    self._nodes = nodes
                    self._edges = np.full(nodes.shape + nodes.shape, default_value)
        elif isinstance(edges, np.ndarray):
            self._nodes = [Node(self, n) for n in range(edges.shape[0])]
            self._edges = edges
        else:
            self._nodes = []
            self._edges = None
        self.edges.value = edges


    def __getitem__(self, key):
        """
        returns Node or Edge or False corresponding to the key
        :param key: Node, int, Edge, (int, int)
        """
        if isinstance(key, int):
            try:
                return self._nodes[key]
            except:
                raise IndexError(f"Index {key} is out of range {len(self)}!")
        elif isinstance(key, tuple):
            if len(key) == 2:
                try:
                    source, target = key
                    if isinstance(source, int) and isinstance(target, int):
                        if self._edges[source, target]:
                            return Edge(self, self[source], self[target])
                        else:
                            return None
                    elif isinstance(source, Node) and isinstance(target, Node):
                        if self._edges[int(source), int(target)]:
                            return Edge(self, source, target)
                        else:
                            return None
                    else:
                        raise TypeError(f"Keys must be of type int or Node not {type(source), type(target)}")
                except:
                    raise IndexError(f"Index {key} is out of range {len(self), len(self)}!")
            else:
                raise IndexError(f"Index for Edges expects 2 keys but {len(key)} where given!")
        elif isinstance(key, Node) or isinstance(key, Edge):
            return key
        else:
            raise Exception(f"Index must be of type int, tuple, Node or Edge not {type(key)}")


    def __getattr__(self, attr):
        if attr == 'nodes':
            return View('node', self)
        elif attr == 'edges':
            return View('edge', self)
        else:
            try:
                return super().__getattr__(self, attr)
            except:
                raise Exception(f"{attr} is currently not implemented")


    def __iter__(self):
        for node in self._nodes:
            yield node


    def __len__(self):
        return len(self._nodes)


    def __str__(self):
        return f"""GRAPH ({len(self)})
{self._edges}"""


    def __repr__(self):
        return str(self)


    def __contains__(self, other):
        if isinstance(other, Node):
            return other in self._nodes
        elif isinstance(other, Edge):
            return bool(self[other])
        else:
            return False


    def _registery(self, mode, attr, val=None):
        """
        All Graph data are stored in the Graph instance. 
        Attribute initialization on Nodes and Edges call 
        the registery to update the Graph data.
        :param mode: str ['node'|'edge']
        :param attr: str
        :param val:
        """
        val = val if val is not None else self.default_value
        if mode == 'node':
            if numeral(val):
                self._node_values[attr] = np.full((len(self),), type(val)())
            else:
                self._node_values[attr] = [type(val)() for _ in range(len(self))]
        elif mode == 'edge':
            if numeral(val):
                self._edge_values[attr] = np.full((len(self), len(self)), type(val)())
            else:
                self._edge_values[attr] = [[type(val)() for _ in range(len(self))] for _ in range(len(self))]



class TSP(Graph):
    def __init__(self, n_nodes=None, coordinates=None, min_distance=0, max_distance=1):
        if n_nodes is not None:
            distances = np.random.rand(n_nodes, n_nodes)
            distances = min_distance + distances * (max_distance - min_distance)
            distances = (distances + distances.T)/2
            distances[np.eye(n_nodes).astype(bool)] = 0
            super().__init__(edges=distances, bidirectional=False)
        elif coordinates is not None:
            distances = np.zeros((len(coordinates), len(coordinates)))
            coordinates = [np.array(c) for c in coordinates]
            for i, c_i in enumerate(coordinates):
                for j, c_j in enumerate(coordinates):
                    distances[i, j] = np.linalg.norm(c_i - c_j)
            super().__init__(edges=distances, bidirectional=False)
            self.nodes.coordinates = coordinates
        else:
            raise Exception("Either coordinates or n_nodes must be specified!")


    def route(self):
        """
        returns the shortest path as well as its length.
        """
        def rec(self, path, length):
            source = path[-1]
            min_length = float('inf')
            min_path = None
            for target in source:
                if target not in path:
                    edge = self[source, target]
                    rec_path, rec_length = rec(self, path + [target], length + edge.value)
                    if rec_length < min_length:
                        min_length, min_path = rec_length, rec_path
            if min_path is not None:
                return min_path, min_length
            else:
                return path + [path[0]], length + self[path[-1], path[0]].value
        return rec(self, [self[0]], 0)



class Germany(TSP):
    def __init__(self, cities={'Osnabrück':   (235, 234),
                               'Hamburg':     (324, 137),
                               'Hanover':     (312, 226),
                               'Frankfurt':   (264, 391),
                               'Munich':      (396, 528),
                               'Berlin':      (478, 215),
                               'Leipzig':     (432, 302),
                               'Düsseldorf':  (175, 310),
                               'Kassel':      (302, 305), 
                               'Cottbus':     (521, 274), 
                               'Bremen':      (270, 173), 
                               'Karlsruhe':   (251, 469),
                               'Nürnberg':    (373, 437),
                               'Saarbrücken': (187, 452)}):
        # coordinates for osna, hamburg, hanover, frankfurt, munich, berlin and leipzig, kassel, Düsseldorf
        coordinates = list(cities.values())
        super().__init__(coordinates=coordinates)
        self.plt = plt
        self.fig, self.ax = self.plt.subplots(figsize=(8, 8))
        #self.plt.ion()


    def visualize(self):
        self.ax.set_title('Route')
        if 'pheromone' in self._edge_values.keys():
            pheromone = self.edges.pheromone
        else:
            raise Warning(f"Edges of the graph must have the attribute 'pheromone'. Graph only got {', '.join(self._edge_values.keys())}")
        coords = np.array([node.coordinates for node in self])
        # Normalize the pheromone levels
        max_phero = np.max(pheromone[pheromone != 1.0])
        min_phero = np.min(pheromone[pheromone != 1.0])
        for n_i in self:
            for n_j in self:
                # Draw the connection of the two nodes based on the pheromone concentration
                if not n_i == n_j:
                    x_values = [n_i.coordinates[0], n_j.coordinates[0]]
                    y_values = [n_i.coordinates[1], n_j.coordinates[1]]
                    # Normalize pheromones for better visibility
                    normed_pheromone = (pheromone[n_i.name, n_j.name] - min_phero) / (max_phero - min_phero)
                    #print(normed_pheromone)
                    # Draw the connection based on determined pheromone level
                    self.ax.plot(x_values, y_values, '-', color=[0, 0, 0, max(min(normed_pheromone, 1), 0)])
        # Mark all cities with a dot
        self.ax.scatter(coords[:, 0], coords[:, 1])
        self.ax.imshow(plt.imread("osm_germany.png"))
        self.plt.axis('off')
        self.plt.tight_layout()
        self.plt.show()



class GridWorld(Graph):
    WALLS = ['w', 'wall']
    FOOD = ['f', 'food']
    FOOD_REWARD = 10.
    def __init__(self, size=None, world=None, values={'f': 'food', 'w': 'wall', ' ': ''}):
        if size is None and world is None and values is None:
            raise Exception("There must either size or world and values be defined")
        elif size is not None and (world is not None or values is not None):
            raise Exception("Either size or world and values must be defined, but not both")
        elif size is not None:
            self.i_size, self.j_size = size
        elif world is not None and values is not None:
            rows = world.split('\n')
            self.i_size, self.j_size = len(rows), len(rows[0])
        self.size = self.i_size, self.j_size
        n_nodes = self.i_size * self.j_size
        edges = np.zeros((n_nodes, n_nodes))
        for i in range(self.i_size):
            for j in range(self.j_size):
                # CHECKING FOR NORTH BORDER
                if i != 0 and values is not None and values[rows[i-1][j]].lower() not in self.WALLS:
                    edges[i * self.i_size + j, (i - 1) * self.i_size + j] = 1.0
                # CHECKING FOR EAST BORDER
                if j != self.j_size - 1 and values is not None and values[rows[i][j+1]].lower() not in self.WALLS:
                    edges[i*self.i_size + j, i * self.i_size + j + 1] = 1.0
                # CHECKING FOR SOUTH BORDER
                if i != self.i_size - 1 and values is not None and values[rows[i+1][j]].lower() not in self.WALLS:
                    edges[i * self.i_size + j, (i + 1) * self.i_size + j] = 1.0
                # CHECKING FOR WEST BORDER
                if j != 0 and values is not None and values[rows[i][j-1]].lower() not in self.WALLS:
                    edges[i * self.i_size + j, i * self.i_size + j - 1] = 1.0
        super().__init__(edges=edges, bidirectional=True)
        self.coordinates = {(i, j): i * self.i_size + j for i in range(self.i_size) for j in range(self.j_size)}
        self.reverse_coordinates = {val: key for key, val in self.coordinates.items()}
        if values is not None:
            for i in range(self.i_size):
                for j in range(self.j_size):
                    # CHECKING FOR NORTH BORDER
                    if i != 0 and values is not None and values[rows[i-1][j]].lower() in self.FOOD:
                        self[(i, j), (i - 1, j)].heuristic = self.FOOD_REWARD
                    # CHECKING FOR EAST BORDER
                    if j != self.j_size - 1 and values is not None and values[rows[i][j+1]].lower() in self.FOOD:
                        self[(i, j), (i, j + 1)].heuristic = self.FOOD_REWARD
                    # CHECKING FOR SOUTH BORDER
                    if i != self.i_size - 1 and values is not None and values[rows[i+1][j]].lower() in self.FOOD:
                        self[(i, j), (i + 1, j)].heuristic = self.FOOD_REWARD
                    # CHECKING FOR WEST BORDER
                    if j != 0 and values is not None and values[rows[i][j-1]].lower() in self.FOOD:
                        self[(i, j), (i, j - 1)].heuristic = self.FOOD_REWARD
        for node in self:
            i, j = self.reverse_coordinates[int(node)]
            node.name = f"{i}|{j}"


    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            a, b = key
            if isinstance(a, tuple) and isinstance(b, tuple):
                return super().__getitem__((self.coordinates[a], self.coordinates[b]))
            elif isinstance(a, int) and isinstance(b, int):
                return super().__getitem__(self.coordinates[key])
            else:
                return super().__getitem__(key)
        else:
            return super().__getitem__(key)



class AntWorld(GridWorld):
    def __init__(self, path='world_0.ant'):
        with open(path, 'r') as file:
            super().__init__(world=file.read())


    def visualize(self):
        if 'pheromone' not in self._edge_values.keys():
            raise Exception("You should set the edges pheromone attribute for visualization.")
        pheromones = np.zeros(self.size)
        for node in self:
            for neighbor in node:
                idx = self.reverse_coordinates[int(neighbor)]
                pheromones[idx] += self[node, neighbor].pheromone
        pheromones[ 0,:] *= 1.5
        pheromones[-1,:] *= 1.5
        pheromones[:, 0] *= 1.5
        pheromones[:,-1] *= 1.5
        plt.imshow(pheromones)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.0001)



"""
USECASES:
    Graph[x] -> Node(x)
        x: index, Node
    Graph[x] -> Edge(x)
        x: Edge
    Graph[x, y] -> Edge(x, y)
        x, y: index, Edge

    Node(x).attr = val -> Graph[x].attr = val
        attr: known, unknown
        val: Any
    Edge(x, y).attr = val -> Graph[x, y].attr = val
        attr: known, unknown
        val: Any

    Node(x).attr -> Graph[x].attr
        âttr: known, unknown
    Edge(x, y).attr -> Graph[x, y].attr
        âttr: known, unknown

    Graph.nodes.attr -> [N0.attr, N1.attr ... Nn.attr]
        attr: known
    Graph.nodes.attr = [numeral, ..., numeral] -> Graph.nodes.attr = [N0.attr, N1.attr ... Nn.attr]
        attr: known, unknown
"""



if __name__ == '__main__':
    G = Germany()
    for item in G.edges:
        print(item)
    exit()
    G = AntWorld()
    print(G.nodes)
    print(G.edges)
    print(G[(3, 4)])
    print(G.edges.heuristic)
    G.visualize()
