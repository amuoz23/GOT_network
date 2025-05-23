import networkx as nx
import numpy as np
import igraph as ig
from igraph import plot
import matplotlib.pyplot as plt
import matplotlib.cm as cm



def info(G):
    info = f"Tipo de grafo: {type(G)}\n"
    info += f"Número de nodos: {G.number_of_nodes()}\n"
    info += f"Número de aristas: {G.number_of_edges()}\n"
    info += f"Es dirigido: {G.is_directed()}\n"
    
    # Métricas de grado
    grados = [d for n, d in G.degree()]
    info += f"Grado medio: {np.mean(grados):.2f}\n"
    info += f"Grado máximo: {max(grados)}\n"
    
    # Densidad
    info += f"Densidad: {nx.density(G):.4f}\n"
    
    # Coeficiente de clustering promedio
    info += f"Coeficiente de clustering promedio: {nx.average_clustering(G):.4f}\n"
    
    # Diámetro (solo para grafos conectados)
    if nx.is_connected(G):
        info += f"Diámetro: {nx.diameter(G)}\n"
    else:
        info += "Diámetro: No aplica (grafo no conectado)\n"
    
    return info



