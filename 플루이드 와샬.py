import pandas as pd
import networkx as nx
from termData import data
import time
import copy
import tracemalloc
import heapq

# 모든 역의 목록 추출
df = pd.DataFrame(data, columns=['Line1', 'Station1', 'Line2', 'Station2', 'Time'])

# Create a graph using the DataFrame
G = nx.Graph()
for index, row in df.iterrows():
    station1 = (row['Line1'], row['Station1'])
    station2 = (row['Line2'], row['Station2'])
    G.add_edge(station1, station2, weight=row['Time'])
    
    
nodes = list(G.nodes)
adj_list = {node: [(neighbor, G[node][neighbor]['weight']) for neighbor in G.neighbors(node)] for node in G.nodes}

#print(adj_list)
def floyd_warshall_with_path(nodes, adj_list):
    # 노드 이름을 인덱스로 매핑
    node_index = {node: i for i, node in enumerate(nodes)}
    index_node = {i: node for node, i in node_index.items()}
    V = len(nodes)
    
    # 무한대 값 정의
    INF = float('inf')
    
    # 인접 행렬 초기화 및 경로 추적 행렬 초기화
    dist = [[INF] * V for _ in range(V)]
    next_node = [[None] * V for _ in range(V)]
    
    # 자신에게 가는 비용은 0으로 설정
    for i in range(V):
        dist[i][i] = 0
        next_node[i][i] = i
    
    # 인접 리스트를 인접 행렬로 변환
    for u in adj_list:
        for v, w in adj_list[u]:
            u_index = node_index[u]
            v_index = node_index[v]
            dist[u_index][v_index] = w
            next_node[u_index][v_index] = v_index
    
    # 플로이드-워셜 알고리즘 수행
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    return dist, next_node, node_index, index_node

def construct_path(u, v, next_node, index_node):
    if next_node[u][v] is None:
        return None
    path = []
    while u != v:
        path.append(index_node[u])
        u = next_node[u][v]
    path.append(index_node[v])
    return path

# 예제 그래프

#nodes = ["A", "B", "C", "D"]
#adj_list = {
#    "A": [("B", 3), ("D", 5)],
#    "B": [("A", 3), ("C", 7), ("D", 4)],
#    "C": [("B", 7), ("D", 2)],
#    "D": [("A", 5), ("B", 4), ("C", 2)]
#}

# 플로이드-워셜 알고리즘 실행
dist, next_node, node_index, index_node = floyd_warshall_with_path(nodes, adj_list)

# 최단 경로 행렬 출력
"""
print("최단 경로 행렬:")
for row in dist:
    print(row)

#인덱스와 노드 매핑 출력
print("노드 인덱스 매핑:")
for node, index in node_index.items():
    print(f"{node}: {index}")
"""
print("두 노드 간의 경로:")
u, v = ('1', '서울역') , ('3', '대화')
u_index = node_index[u]
v_index = node_index[v]

path = construct_path(u_index, v_index, next_node, index_node)
if path:
    print(u,"에서 ",v,"로 가는 경로:",path)
else:
    print(f"{u}에서 {v}로 가는 경로가 존재하지 않습니다.")
    
print("소요시간", dist[u_index][v_index],"min")