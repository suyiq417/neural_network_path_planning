from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from evacuation_entity import EvacuationEntity, EntityType
import networkx as nx
import os
import seaborn as sns
import pandas as pd
from collections import defaultdict
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 常量定义
FLOOR_HEIGHT = 3.5  # 楼层高度（米）
NODE_SIZE_2D = 150  # 2D图中节点的默认大小
NODE_SIZE_3D = 100  # 3D图中节点的默认大小
EDGE_WIDTH = 1.0    # 边的默认宽度
ALPHA = 0.7         # 透明度
DPI = 300           # 图像DPI

# 实体类型的颜色映射
ENTITY_COLOR_MAP = {
    EntityType.Door: '#1f77b4',     # 蓝色
    EntityType.Exit: '#d62728',     # 红色
    EntityType.Corridor: '#ff7f0e', # 橙色
    EntityType.Stair: '#9467bd',    # 紫色
    EntityType.Room: '#2ca02c',     # 绿色
}

# 实体类型的标记映射
ENTITY_MARKER_MAP = {
    EntityType.Door: 's',     # 方形
    EntityType.Exit: '*',     # 星形
    EntityType.Corridor: 'o', # 圆形
    EntityType.Stair: '^',    # 三角形
    EntityType.Room: 'p',     # 五边形
}

class TopologyVisualization:
    """
    建筑物实体空间拓扑图可视化类
    """
    def __init__(self, all_entities: List[EvacuationEntity]):
        """
        初始化拓扑可视化
        
        参数:
            all_entities: 所有建筑物实体的列表
        """
        self.all_entities = all_entities
        self.entity_map = {entity.id: entity for entity in all_entities}
        self.floors = sorted(set(entity.floor for entity in all_entities))
        self.entity_types = set(entity.entity_type for entity in all_entities)
        
        # 为每种实体类型创建颜色映射
        self.color_palette = ENTITY_COLOR_MAP
        
        # 创建NetworkX图
        self.graph = self._build_graph()
    
    def _build_graph(self) -> nx.Graph:
        """构建NetworkX图对象"""
        G = nx.Graph()
        
        # 添加节点
        for entity in self.all_entities:
            G.add_node(entity.id, 
                      pos=entity.center, 
                      floor=entity.floor, 
                      entity_type=entity.entity_type,
                      name=entity.name,
                      area=entity.area)
        
        # 添加边
        for entity in self.all_entities:
            for connected_id in entity.connected_entity_ids:
                if connected_id in self.entity_map:
                    # 计算连接实体之间的距离
                    distance = EvacuationEntity.calculate_distance(
                        entity, 
                        self.entity_map[connected_id],
                        FLOOR_HEIGHT
                    )
                    G.add_edge(entity.id, connected_id, weight=distance)
        
        return G
    
    def get_floor_subgraph(self, floor: int) -> nx.Graph:
        """获取特定楼层的子图"""
        nodes = [node for node, attrs in self.graph.nodes(data=True) 
                if attrs.get('floor') == floor]
        return self.graph.subgraph(nodes)
    
    def visualize_2d_topology(self, 
                             output_dir: str,
                             filename_prefix: str = "topology_2d",
                             show_labels: bool = True,
                             highlight_nodes: List[int] = None,
                             highlight_edges: List[Tuple[int, int]] = None,
                             figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        生成2D拓扑图可视化
        
        参数:
            output_dir: 输出目录
            filename_prefix: 输出文件名前缀
            show_labels: 是否显示节点标签
            highlight_nodes: 要高亮显示的节点列表
            highlight_edges: 要高亮显示的边列表
            figsize: 图像大小
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个楼层生成图像
        for floor in self.floors:
            print(f"生成第{floor}层拓扑图...")
            subgraph = self.get_floor_subgraph(floor)
            
            if not subgraph.nodes:
                print(f"  第{floor}层没有节点，跳过")
                continue
            
            # 创建图和坐标
            fig, ax = plt.subplots(figsize=figsize)
            pos = {n: self.graph.nodes[n]['pos'] for n in subgraph.nodes()}
            
            # 按实体类型绘制节点
            for entity_type in self.entity_types:
                node_list = [n for n in subgraph.nodes() 
                            if self.graph.nodes[n]['entity_type'] == entity_type]
                if node_list:
                    nx.draw_networkx_nodes(
                        subgraph, 
                        pos, 
                        nodelist=node_list,
                        node_color=self.color_palette.get(entity_type, 'gray'),
                        node_shape=ENTITY_MARKER_MAP.get(entity_type, 'o'),
                        node_size=NODE_SIZE_2D,
                        alpha=ALPHA,
                        ax=ax
                    )
            
            # 绘制边
            nx.draw_networkx_edges(
                subgraph, 
                pos, 
                width=EDGE_WIDTH,
                alpha=0.5,
                ax=ax
            )
            
            # 高亮显示指定的节点
            if highlight_nodes:
                highlight_nodes_in_floor = [n for n in highlight_nodes if n in subgraph.nodes()]
                if highlight_nodes_in_floor:
                    nx.draw_networkx_nodes(
                        subgraph, 
                        pos, 
                        nodelist=highlight_nodes_in_floor,
                        node_color='yellow',
                        node_size=NODE_SIZE_2D * 1.5,
                        linewidths=2,
                        edgecolors='black',
                        ax=ax
                    )
            
            # 高亮显示指定的边
            if highlight_edges:
                highlight_edges_in_floor = [(u, v) for u, v in highlight_edges 
                                          if u in subgraph.nodes() and v in subgraph.nodes()]
                if highlight_edges_in_floor:
                    edge_pos = [(pos[u], pos[v]) for u, v in highlight_edges_in_floor]
                    line_segments = LineCollection(
                        edge_pos,
                        colors='red',
                        linewidths=2,
                        alpha=0.8,
                        zorder=1
                    )
                    ax.add_collection(line_segments)
            
            # 添加标签
            if show_labels:
                labels = {n: f"{n}" for n in subgraph.nodes()}
                nx.draw_networkx_labels(
                    subgraph, 
                    pos, 
                    labels=labels, 
                    font_size=8,
                    ax=ax
                )
            
            # 创建图例
            legend_elements = []
            for entity_type in self.entity_types:
                color = self.color_palette.get(entity_type, 'gray')
                marker = ENTITY_MARKER_MAP.get(entity_type, 'o')
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color='w', 
                          markerfacecolor=color, markersize=10, 
                          label=entity_type.name)
                )
            
            # 添加高亮图例
            if highlight_nodes:
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='yellow', markersize=10,
                          markeredgecolor='black', markeredgewidth=2,
                          label='高亮节点')
                )
            
            if highlight_edges:
                legend_elements.append(
                    Line2D([0], [0], color='red', lw=2, label='高亮连接')
                )
            
            # 设置图例和标题
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1))
            ax.set_title(f'第{floor}层建筑物拓扑图')
            ax.axis('off')
            plt.tight_layout()
            
            # 保存图像
            output_file = os.path.join(output_dir, f"{filename_prefix}_floor_{floor}.png")
            plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            print(f"  已保存到: {output_file}")
    
    def visualize_3d_topology(self, 
                             output_dir: str,
                             filename: str = "topology_3d.png",
                             show_labels: bool = True,
                             highlight_nodes: List[int] = None,
                             highlight_edges: List[Tuple[int, int]] = None,
                             figsize: Tuple[int, int] = (15, 12),
                             elevation: float = 30,
                             azimuth: float = -60) -> None:
        """
        生成3D拓扑图可视化
        
        参数:
            output_dir: 输出目录
            filename: 输出文件名
            show_labels: 是否显示节点标签
            highlight_nodes: 要高亮显示的节点列表
            highlight_edges: 要高亮显示的边列表
            figsize: 图像大小
            elevation: 仰角
            azimuth: 方位角
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("生成3D建筑物拓扑图...")
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取所有节点的3D坐标
        pos_3d = {}
        for node in self.graph.nodes():
            x, y = self.graph.nodes[node]['pos']
            floor = self.graph.nodes[node]['floor']
            z = floor * FLOOR_HEIGHT
            pos_3d[node] = (x, y, z)
        
        # 按楼层和实体类型绘制节点
        for floor in self.floors:
            for entity_type in self.entity_types:
                node_list = [n for n in self.graph.nodes() 
                            if self.graph.nodes[n]['floor'] == floor 
                            and self.graph.nodes[n]['entity_type'] == entity_type]
                
                if node_list:
                    x = [pos_3d[n][0] for n in node_list]
                    y = [pos_3d[n][1] for n in node_list]
                    z = [pos_3d[n][2] for n in node_list]
                    
                    ax.scatter(
                        x, y, z,
                        c=self.color_palette.get(entity_type, 'gray'),
                        marker=ENTITY_MARKER_MAP.get(entity_type, 'o'),
                        s=NODE_SIZE_3D,
                        alpha=ALPHA,
                        label=f"{entity_type.name}" if floor == self.floors[0] else "_nolegend_"
                    )
        
        # 绘制边
        for u, v in self.graph.edges():
            x = [pos_3d[u][0], pos_3d[v][0]]
            y = [pos_3d[u][1], pos_3d[v][1]]
            z = [pos_3d[u][2], pos_3d[v][2]]
            
            # 检查是否为跨楼层连接
            is_cross_floor = self.graph.nodes[u]['floor'] != self.graph.nodes[v]['floor']
            
            ax.plot(
                x, y, z,
                color='red' if is_cross_floor else 'gray',
                linewidth=2.0 if is_cross_floor else EDGE_WIDTH,
                alpha=0.7 if is_cross_floor else 0.5,
                linestyle='-' if is_cross_floor else '--'
            )
        
        # 高亮显示指定的节点
        if highlight_nodes:
            x = [pos_3d[n][0] for n in highlight_nodes if n in self.graph.nodes()]
            y = [pos_3d[n][1] for n in highlight_nodes if n in self.graph.nodes()]
            z = [pos_3d[n][2] for n in highlight_nodes if n in self.graph.nodes()]
            
            if x:  # 确保有要绘制的节点
                ax.scatter(
                    x, y, z,
                    color='yellow',
                    s=NODE_SIZE_3D * 1.5,
                    edgecolor='black',
                    linewidth=2,
                    alpha=1.0,
                    label='高亮节点'
                )
        
        # 高亮显示指定的边
        if highlight_edges:
            for u, v in highlight_edges:
                if u in self.graph.nodes() and v in self.graph.nodes():
                    x = [pos_3d[u][0], pos_3d[v][0]]
                    y = [pos_3d[u][1], pos_3d[v][1]]
                    z = [pos_3d[u][2], pos_3d[v][2]]
                    
                    ax.plot(
                        x, y, z,
                        color='darkred',
                        linewidth=3.0,
                        alpha=1.0
                    )
        
        # 添加标签
        if show_labels:
            for node in self.graph.nodes():
                x, y, z = pos_3d[node]
                ax.text(
                    x, y, z,
                    f"{node}",
                    size=8,
                    zorder=1,
                    color='black'
                )
        
        # 设置坐标轴标签和标题
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_zlabel('Z坐标 (高度)')
        ax.set_title('3D建筑物拓扑图')
        
        # 添加楼层平面
        floor_alphas = [0.1] * len(self.floors)  # 设置楼层平面透明度
        floor_colors = ['lightgray'] * len(self.floors)  # 设置楼层平面颜色
        
        # 找出每层的X和Y范围
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        
        for node in self.graph.nodes():
            x, y = self.graph.nodes[node]['pos']
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
        
        # 添加边距
        margin = max((x_max - x_min), (y_max - y_min)) * 0.1
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        # 绘制每层楼的平面
        for i, floor in enumerate(self.floors):
            z = floor * FLOOR_HEIGHT
            
            # 创建平面的四个角
            xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
            zz = np.ones_like(xx) * z
            
            # 绘制平面
            surf = ax.plot_surface(
                xx, yy, zz,
                color=floor_colors[i],
                alpha=floor_alphas[i],
                shade=False
            )
            
            # 添加楼层标签
            ax.text(
                x_min + (x_max - x_min) * 0.05,
                y_min + (y_max - y_min) * 0.05,
                z,
                f"第{floor}层",
                size=10,
                color='black'
            )
        
        # 添加跨楼层连接的图例项
        handles, labels = ax.get_legend_handles_labels()
        cross_floor_line = Line2D([0], [0], color='red', linewidth=2.0, label='跨楼层连接')
        handles.append(cross_floor_line)
        
        if highlight_edges:
            highlight_edge_line = Line2D([0], [0], color='darkred', linewidth=3.0, label='高亮连接')
            handles.append(highlight_edge_line)
        
        # 设置图例
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1))
        
        # 设置视角
        ax.view_init(elev=elevation, azim=azimuth)
        
        # 设置坐标轴比例相等
        ax.set_box_aspect([1, 1, (z_max := max(self.floors) * FLOOR_HEIGHT) / max(x_max - x_min, y_max - y_min)])
        
        # 添加网格线
        ax.grid(True, alpha=0.3)
        
        # 保存图像
        output_file = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  已保存到: {output_file}")
    
    def generate_topology_statistics(self, output_dir: str, filename: str = "topology_stats.csv") -> None:
        """
        生成拓扑图统计信息并保存为CSV
        
        参数:
            output_dir: 输出目录
            filename: 输出文件名
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("生成拓扑图统计信息...")
        
        # 基本图统计
        graph_stats = {
            "总节点数": self.graph.number_of_nodes(),
            "总连接数": self.graph.number_of_edges(),
            "总楼层数": len(self.floors),
            "平均度数": np.mean([d for _, d in self.graph.degree()]),
            "最大度数": max([d for _, d in self.graph.degree()]) if self.graph.degree() else 0,
            "最小度数": min([d for _, d in self.graph.degree()]) if self.graph.degree() else 0,
            "是否连通": nx.is_connected(self.graph),
            "连通分量数": nx.number_connected_components(self.graph),
        }
        
        # 每层统计
        floor_stats = {}
        for floor in self.floors:
            subgraph = self.get_floor_subgraph(floor)
            floor_stats[f"第{floor}层节点数"] = subgraph.number_of_nodes()
            floor_stats[f"第{floor}层连接数"] = subgraph.number_of_edges()
            floor_stats[f"第{floor}层是否连通"] = nx.is_connected(subgraph)
            floor_stats[f"第{floor}层连通分量数"] = nx.number_connected_components(subgraph)
        
        # 实体类型统计
        entity_type_counts = defaultdict(int)
        for entity in self.all_entities:
            entity_type_counts[entity.entity_type.name] += 1
        
        # 跨楼层连接统计
        cross_floor_connections = 0
        for u, v in self.graph.edges():
            if self.graph.nodes[u]['floor'] != self.graph.nodes[v]['floor']:
                cross_floor_connections += 1
        
        cross_floor_stats = {
            "跨楼层连接数": cross_floor_connections,
            "跨楼层连接比例": cross_floor_connections / self.graph.number_of_edges() if self.graph.number_of_edges() > 0 else 0
        }
        
        # 中心性指标
        try:
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            centrality_stats = {
                "最高度中心性节点": max(degree_centrality.items(), key=lambda x: x[1])[0] if degree_centrality else None,
                "最高度中心性值": max(degree_centrality.values()) if degree_centrality else 0,
                "最高介数中心性节点": max(betweenness_centrality.items(), key=lambda x: x[1])[0] if betweenness_centrality else None,
                "最高介数中心性值": max(betweenness_centrality.values()) if betweenness_centrality else 0,
                "最高接近中心性节点": max(closeness_centrality.items(), key=lambda x: x[1])[0] if closeness_centrality else None,
                "最高接近中心性值": max(closeness_centrality.values()) if closeness_centrality else 0,
            }
        except:
            centrality_stats = {
                "中心性计算": "图不连通或其他原因导致无法计算"
            }
        
        # 合并所有统计信息
        all_stats = {**graph_stats, **floor_stats, **entity_type_counts, **cross_floor_stats, **centrality_stats}
        
        # 转换为DataFrame并保存
        stats_df = pd.DataFrame([all_stats])
        output_file = os.path.join(output_dir, filename)
        stats_df.to_csv(output_file, index=False)
        print(f"  统计信息已保存到: {output_file}")
        
        return all_stats
    
    def visualize_centrality_measures(self, 
                                     output_dir: str,
                                     filename_prefix: str = "centrality",
                                     figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        可视化节点的中心性度量
        
        参数:
            output_dir: 输出目录
            filename_prefix: 输出文件名前缀
            figsize: 图像大小
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("计算并可视化中心性度量...")
        
        # 计算中心性
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)
        
        centrality_measures = {
            "度中心性": degree_centrality,
            "介数中心性": betweenness_centrality,
            "接近中心性": closeness_centrality
        }
        
        # 为每个中心性度量绘制热力图
        for measure_name, centrality in centrality_measures.items():
            print(f"  绘制{measure_name}热力图...")
            
            for floor in self.floors:
                subgraph = self.get_floor_subgraph(floor)
                if not subgraph.nodes:
                    continue
                
                fig, ax = plt.subplots(figsize=figsize)
                
                # 获取该楼层节点的位置和中心性值
                pos = {n: self.graph.nodes[n]['pos'] for n in subgraph.nodes()}
                node_colors = [centrality.get(n, 0) for n in subgraph.nodes()]
                
                # 绘制节点，颜色根据中心性值变化
                nodes = nx.draw_networkx_nodes(
                    subgraph,
                    pos,
                    node_color=node_colors,
                    cmap=plt.cm.viridis,
                    node_size=NODE_SIZE_2D,
                    alpha=0.8,
                    ax=ax
                )
                
                # 绘制边
                nx.draw_networkx_edges(
                    subgraph,
                    pos,
                    alpha=0.3,
                    ax=ax
                )
                
                # 添加标签
                labels = {n: f"{n}" for n in subgraph.nodes()}
                nx.draw_networkx_labels(
                    subgraph,
                    pos,
                    labels=labels,
                    font_size=8,
                    ax=ax
                )
                
                # 添加颜色条
                plt.colorbar(nodes, ax=ax, label=measure_name)
                
                # 添加实体类型图例
                legend_elements = []
                for entity_type in self.entity_types:
                    color = self.color_palette.get(entity_type, 'gray')
                    marker = ENTITY_MARKER_MAP.get(entity_type, 'o')
                    legend_elements.append(
                        Line2D([0], [0], marker=marker, color='w',
                              markerfacecolor=color, markersize=10,
                              label=entity_type.name)
                    )
                
                # 设置图例和标题
                ax.legend(handles=legend_elements, title="实体类型",
                         loc='upper left', bbox_to_anchor=(1.01, 1))
                ax.set_title(f'第{floor}层 - {measure_name}')
                ax.axis('off')
                plt.tight_layout()
                
                # 保存图像
                output_file = os.path.join(output_dir, f"{filename_prefix}_{measure_name.replace(' ', '_')}_floor_{floor}.png")
                plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
                plt.close(fig)
                print(f"    已保存到: {output_file}")
    
    def analyze_critical_paths(self, 
                              source_types: List[EntityType] = [EntityType.Room],
                              target_types: List[EntityType] = [EntityType.Exit],
                              output_dir: str = None,
                              filename: str = "critical_paths.png",
                              figsize: Tuple[int, int] = (15, 12)) -> Dict:
        """
        分析关键路径并可视化
        
        参数:
            source_types: 源节点类型列表（默认为房间）
            target_types: 目标节点类型列表（默认为出口）
            output_dir: 输出目录
            filename: 输出文件名
            figsize: 图像大小
            
        返回:
            关键路径分析结果字典
        """
        print("分析关键路径...")
        
        # 找出所有源节点和目标节点
        source_nodes = [entity.id for entity in self.all_entities 
                       if entity.entity_type in source_types]
        target_nodes = [entity.id for entity in self.all_entities 
                       if entity.entity_type in target_types]
        
        if not source_nodes or not target_nodes:
            print("  没有找到源节点或目标节点，无法分析关键路径")
            return {}
        
        # 计算所有最短路径
        all_paths = []
        edge_usage = defaultdict(int)
        node_usage = defaultdict(int)
        
        for source in source_nodes:
            for target in target_nodes:
                try:
                    # 使用Dijkstra算法找最短路径
                    if nx.has_path(self.graph, source, target):
                        path = nx.shortest_path(self.graph, source, target, weight='weight')
                        all_paths.append(path)
                        
                        # 统计节点和边的使用频率
                        for node in path:
                            node_usage[node] += 1
                            
                        for i in range(len(path) - 1):
                            edge = tuple(sorted([path[i], path[i+1]]))
                            edge_usage[edge] += 1
                except nx.NetworkXNoPath:
                    pass
        
        # 找出最常用的边和节点（关键路径组件）
        if edge_usage:
            most_used_edges = sorted(edge_usage.items(), key=lambda x: x[1], reverse=True)
            most_used_nodes = sorted(node_usage.items(), key=lambda x: x[1], reverse=True)
            
            top_edges = most_used_edges[:min(10, len(most_used_edges))]
            top_nodes = most_used_nodes[:min(10, len(most_used_nodes))]
            
            # 打印关键路径分析结果
            print(f"  共分析了 {len(all_paths)} 条路径")
            print(f"  最常用的前{len(top_edges)}条边:")
            for edge, count in top_edges:
                print(f"    边 {edge}: 使用 {count} 次 (占比 {count/len(all_paths)*100:.1f}%)")
                
            print(f"  最常用的前{len(top_nodes)}个节点:")
            for node, count in top_nodes:
                node_type = self.graph.nodes[node]['entity_type'].name
                print(f"    节点 {node} ({node_type}): 使用 {count} 次 (占比 {count/len(all_paths)*100:.1f}%)")
        else:
            print("  未找到任何有效路径")
            return {}
        
        # 生成并保存可视化
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 3D可视化
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # 获取所有节点的3D坐标
            pos_3d = {}
            for node in self.graph.nodes():
                x, y = self.graph.nodes[node]['pos']
                floor = self.graph.nodes[node]['floor']
                z = floor * FLOOR_HEIGHT
                pos_3d[node] = (x, y, z)
            
            # 绘制所有节点（半透明）
            for node in self.graph.nodes():
                x, y, z = pos_3d[node]
                entity_type = self.graph.nodes[node]['entity_type']
                color = self.color_palette.get(entity_type, 'gray')
                marker = ENTITY_MARKER_MAP.get(entity_type, 'o')
                
                ax.scatter(
                    x, y, z,
                    color=color,
                    marker=marker,
                    s=NODE_SIZE_3D * 0.7,
                    alpha=0.3
                )
            
            # 绘制所有边（半透明）
            for u, v in self.graph.edges():
                x = [pos_3d[u][0], pos_3d[v][0]]
                y = [pos_3d[u][1], pos_3d[v][1]]
                z = [pos_3d[u][2], pos_3d[v][2]]
                
                ax.plot(
                    x, y, z,
                    color='gray',
                    linewidth=EDGE_WIDTH * 0.7,
                    alpha=0.2
                )
            
            # 高亮显示最常用的边
            for (u, v), count in top_edges:
                x = [pos_3d[u][0], pos_3d[v][0]]
                y = [pos_3d[u][1], pos_3d[v][1]]
                z = [pos_3d[u][2], pos_3d[v][2]]
                
                # 根据使用频率设置颜色
                normalized_count = count / max(c for _, c in top_edges)
                color = plt.cm.hot(normalized_count)
                linewidth = EDGE_WIDTH * (1 + normalized_count * 2)
                
                ax.plot(
                    x, y, z,
                    color=color,
                    linewidth=linewidth,
                    alpha=0.9
                )
            
            # 高亮显示最常用的节点
            for node, count in top_nodes:
                x, y, z = pos_3d[node]
                
                # 根据使用频率设置大小
                normalized_count = count / max(c for _, c in top_nodes)
                size = NODE_SIZE_3D * (1 + normalized_count * 3)
                
                ax.scatter(
                    x, y, z,
                    color='red',
                    s=size,
                    alpha=0.9,
                    edgecolor='black'
                )
                
                # 添加标签
                ax.text(
                    x, y, z,
                    f"{node}\n({count}次)",
                    size=8,
                    zorder=1,
                    color='black'
                )
            
            # 设置图表
            ax.set_xlabel('X坐标')
            ax.set_ylabel('Y坐标')
            ax.set_zlabel('Z坐标 (高度)')
            ax.set_title('建筑物关键路径分析')
            
            # 添加图例
            source_line = Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=self.color_palette.get(source_types[0], 'gray'),
                               markersize=10, label='源点')
            target_line = Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=self.color_palette.get(target_types[0], 'gray'),
                               markersize=10, label='目标点')
            critical_node = Line2D([0], [0], marker='o', color='w',
                                markerfacecolor='red', markersize=10,
                                markeredgecolor='black', label='关键节点')
            critical_edge = Line2D([0], [0], color='orange', lw=2, label='关键边')
            
            ax.legend(handles=[source_line, target_line, critical_node, critical_edge],
                    loc='upper left', bbox_to_anchor=(1.01, 1))
            
            # 设置视角
            ax.view_init(elev=30, azim=-60)
            
            # 保存图像
            output_file = os.path.join(output_dir, filename)
            plt.tight_layout()
            plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            print(f"  已保存到: {output_file}")
        
        # 返回分析结果
        return {
            "路径总数": len(all_paths),
            "最常用边": top_edges,
            "最常用节点": top_nodes
        }

def main():
    """主函数，用于演示和测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='建筑物空间拓扑图可视化')
    parser.add_argument('--data_file', type=str, required=True,
                       help='建筑物数据Excel文件路径')
    parser.add_argument('--output_dir', type=str, default='topology_output',
                       help='输出目录')
    parser.add_argument('--highlight_nodes', type=str, default='',
                       help='高亮显示的节点ID列表，用逗号分隔')
    args = parser.parse_args()
    
    try:
        # 加载实体数据
        print(f"从 {args.data_file} 加载建筑物实体数据...")
        all_entities = EvacuationEntity.from_xlsx(args.data_file)
        print(f"成功加载 {len(all_entities)} 个实体")
        
        # 创建拓扑可视化对象
        topology_viz = TopologyVisualization(all_entities)
        
        # 准备高亮节点
        highlight_nodes = []
        if args.highlight_nodes:
            highlight_nodes = [int(id.strip()) for id in args.highlight_nodes.split(',')]
        
        # 生成2D拓扑图
        topology_viz.visualize_2d_topology(
            args.output_dir,
            highlight_nodes=highlight_nodes
        )
        
        # 生成3D拓扑图
        topology_viz.visualize_3d_topology(
            args.output_dir,
            highlight_nodes=highlight_nodes
        )
        
        # 生成统计信息
        topology_viz.generate_topology_statistics(args.output_dir)
        
        # 可视化中心性度量
        topology_viz.visualize_centrality_measures(args.output_dir)
        
        # 分析关键路径
        topology_viz.analyze_critical_paths(output_dir=args.output_dir)
        
        print("拓扑图可视化完成！")
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()