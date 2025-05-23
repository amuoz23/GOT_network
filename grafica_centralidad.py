import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize

def graficar_centralidad_igraph(G_ig, centrality, titulo="Centralidad", layout="fr"):
    """
    Visualiza centralidad usando igraph con mejor calidad visual
    
    Parameters:
    -----------
    G_ig : igraph.Graph
        El grafo de igraph
    centrality : dict, list o array
        Valores de centralidad para cada nodo
    titulo : str
        Título del gráfico
    layout : str
        Algoritmo de layout ('fr', 'kk', 'circle', 'grid', 'random', 'drl')
    """
    # Convertir centralidad a array
    if isinstance(centrality, dict):
        cent_values = np.array(list(centrality.values()))
    else:
        cent_values = np.array(centrality)
    
    # Crear layout
    if layout == "fr":
        pos = G_ig.layout_fruchterman_reingold(niter=500)
    elif layout == "kk":
        pos = G_ig.layout_kamada_kawai()
    elif layout == "circle":
        pos = G_ig.layout_circle()
    elif layout == "grid":
        pos = G_ig.layout_grid()
    elif layout == "drl":
        pos = G_ig.layout_drl()
    else:
        pos = G_ig.layout_auto()
    
    # Configurar tamaños y colores
    if len(np.unique(cent_values)) == 1:
        vertex_sizes = [30] * len(cent_values)
        colors = [0.5] * len(cent_values)
        vmin = vmax = cent_values[0]
    else:
        min_val, max_val = cent_values.min(), cent_values.max()
        # Tamaños de nodos (15-60)
        vertex_sizes = 15 + 45 * (cent_values - min_val) / (max_val - min_val)
        colors = (cent_values - min_val) / (max_val - min_val)
        vmin, vmax = min_val, max_val
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    
    # Configurar colormap
    cmap = cm.plasma
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Dibujar el grafo
    ig.plot(G_ig, 
            target=ax,
            layout=pos,
            vertex_size=vertex_sizes,
            vertex_color=[cmap(norm(c)) for c in colors],
            vertex_frame_width=1,
            vertex_frame_color="white",
            vertex_label=range(G_ig.vcount()),
            vertex_label_size=10,
            vertex_label_color="white",
            vertex_label_dist=0,
            edge_width=1.5,
            edge_color="rgba(100,100,100,0.3)",
            edge_curved=0.1,
            margin=50)
    
    # Título
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=30)
    cbar.set_label('Valor de Centralidad', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas
    print(f"\n📊 Estadísticas de {titulo}:")
    print(f"   Mínimo: {cent_values.min():.4f}")
    print(f"   Máximo: {cent_values.max():.4f}")
    print(f"   Promedio: {cent_values.mean():.4f}")
    print(f"   Desv. estándar: {cent_values.std():.4f}")

def convertir_nx_a_igraph(G_nx):
    """
    Convierte un grafo de NetworkX a igraph
    """
    # Obtener lista de aristas
    edges = list(G_nx.edges())
    
    # Crear grafo igraph
    G_ig = ig.Graph()
    G_ig.add_vertices(len(G_nx.nodes()))
    
    # Mapear nodos si no son secuenciales
    node_to_id = {node: i for i, node in enumerate(G_nx.nodes())}
    edge_list = [(node_to_id[u], node_to_id[v]) for u, v in edges]
    
    G_ig.add_edges(edge_list)
    G_ig.vs['name'] = list(G_nx.nodes())
    
    return G_ig, node_to_id

def graficar_centralidad_premium(G_nx, centrality, titulo="Centralidad", layout="fr"):
    """
    Versión premium que convierte automáticamente de NetworkX a igraph
    
    Parameters:
    -----------
    G_nx : networkx.Graph
        Grafo de NetworkX
    centrality : dict, list o array
        Valores de centralidad
    titulo : str
        Título del gráfico
    layout : str
        Algoritmo de layout
    """
    # Convertir a igraph
    G_ig, node_mapping = convertir_nx_a_igraph(G_nx)
    
    # Si centrality es dict con nombres de nodos, reordenar
    if isinstance(centrality, dict):
        cent_ordenada = [centrality[node] for node in G_nx.nodes()]
    else:
        cent_ordenada = centrality
    
    # Usar la función de igraph
    graficar_centralidad_igraph(G_ig, cent_ordenada, titulo, layout)

def comparar_centralidades_igraph(G_nx, centralidades_dict, layouts=None):
    """
    Compara múltiples centralidades usando igraph con layouts personalizados
    
    Parameters:
    -----------
    G_nx : networkx.Graph
        Grafo de NetworkX
    centralidades_dict : dict
        Diccionario con tipo de centralidad como clave
    layouts : list o None
        Lista de layouts para cada subplot, o None para usar 'fr' en todos
    """
    n_plots = len(centralidades_dict)
    cols = min(3, n_plots)
    rows = (n_plots - 1) // cols + 1
    
    if layouts is None:
        layouts = ['fr'] * n_plots
    
    # Convertir a igraph una sola vez
    G_ig, node_mapping = convertir_nx_a_igraph(G_nx)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), facecolor='white')
    if n_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes) if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, ((tipo, centrality), layout) in enumerate(zip(centralidades_dict.items(), layouts)):
        ax = axes[i]
        
        # Preparar datos
        if isinstance(centrality, dict):
            cent_values = np.array([centrality[node] for node in G_nx.nodes()])
        else:
            cent_values = np.array(centrality)
        
        # Layout
        if layout == "fr":
            pos = G_ig.layout_fruchterman_reingold(niter=300)
        elif layout == "kk":
            pos = G_ig.layout_kamada_kawai()
        elif layout == "circle":
            pos = G_ig.layout_circle()
        else:
            pos = G_ig.layout_auto()
        
        # Colores y tamaños
        if len(np.unique(cent_values)) == 1:
            vertex_sizes = [25] * len(cent_values)
            colors = [0.5] * len(cent_values)
            vmin = vmax = cent_values[0]
        else:
            min_val, max_val = cent_values.min(), cent_values.max()
            vertex_sizes = 10 + 30 * (cent_values - min_val) / (max_val - min_val)
            colors = (cent_values - min_val) / (max_val - min_val)
            vmin, vmax = min_val, max_val
        
        # Colormap diferente para cada tipo
        cmaps = ['plasma', 'viridis', 'coolwarm', 'magma', 'inferno']
        cmap = getattr(cm, cmaps[i % len(cmaps)])
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Dibujar
        ig.plot(G_ig,
                target=ax,
                layout=pos,
                vertex_size=vertex_sizes,
                vertex_color=[cmap(norm(c)) for c in colors],
                vertex_frame_width=0.5,
                vertex_frame_color="white",
                vertex_label=range(G_ig.vcount()),
                vertex_label_size=8,
                vertex_label_color="black",
                edge_width=1,
                edge_color="rgba(150,150,150,0.4)",
                edge_curved=0.05,
                margin=30)
        
        ax.set_title(f'{tipo.capitalize()} Centrality', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Mini colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.6)
    
    # Ocultar subplots vacíos
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Función de conveniencia con estilos predefinidos
def graficar_estilo_moderno(G_nx, centrality, estilo="neon"):
    """
    Estilos predefinidos modernos
    
    Estilos disponibles: 'neon', 'pastel', 'dark', 'academic'
    """
    G_ig, _ = convertir_nx_a_igraph(G_nx)
    
    if isinstance(centrality, dict):
        cent_values = np.array([centrality[node] for node in G_nx.nodes()])
    else:
        cent_values = np.array(centrality)
    
    # Configuraciones por estilo
    estilos = {
        'neon': {
            'facecolor': 'black',
            'cmap': cm.plasma,
            'edge_color': 'rgba(0,255,255,0.3)',
            'vertex_frame_color': 'cyan',
            'vertex_label_color': 'white'
        },
        'pastel': {
            'facecolor': '#f8f9fa',
            'cmap': cm.Set3,
            'edge_color': 'rgba(200,200,200,0.5)',
            'vertex_frame_color': 'white',
            'vertex_label_color': 'black'
        },
        'dark': {
            'facecolor': '#2c3e50',
            'cmap': cm.viridis,
            'edge_color': 'rgba(255,255,255,0.2)',
            'vertex_frame_color': 'white',
            'vertex_label_color': 'white'
        },
        'academic': {
            'facecolor': 'white',
            'cmap': cm.Blues,
            'edge_color': 'rgba(100,100,100,0.6)',
            'vertex_frame_color': 'navy',
            'vertex_label_color': 'white'
        }
    }
    
    config = estilos.get(estilo, estilos['neon'])
    
    # Preparar datos
    min_val, max_val = cent_values.min(), cent_values.max()
    if max_val > min_val:
        vertex_sizes = 20 + 40 * (cent_values - min_val) / (max_val - min_val)
        colors = (cent_values - min_val) / (max_val - min_val)
    else:
        vertex_sizes = [30] * len(cent_values)
        colors = [0.5] * len(cent_values)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=config['facecolor'])
    
    # Layout
    pos = G_ig.layout_fruchterman_reingold(niter=500)
    
    # Dibujar
    ig.plot(G_ig,
            target=ax,
            layout=pos,
            vertex_size=vertex_sizes,
            vertex_color=[config['cmap'](c) for c in colors],
            vertex_frame_width=2,
            vertex_frame_color=config['vertex_frame_color'],
            vertex_label=range(G_ig.vcount()),
            vertex_label_size=10,
            vertex_label_color=config['vertex_label_color'],
            edge_width=2,
            edge_color=config['edge_color'],
            edge_curved=0.1,
            margin=60)
    
    ax.set_title(f'Centralidad - Estilo {estilo.capitalize()}', 
                fontsize=16, fontweight='bold', 
                color='white' if config['facecolor'] != 'white' else 'black')
    ax.axis('off')
    
    # Colorbar
    norm = Normalize(vmin=min_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=config['cmap'], norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Centralidad', fontsize=12)
    
    plt.tight_layout()
    plt.show()