import pandas as pd
import networkx as nx
from termData import data
import time
import copy
import tracemalloc
import heapq

# 직행 노선의 이동 시간은 2분; 환승은 5분 소요
start_station = ('1', '서울역')
end_station = ('4', '과천')

# DataFrame 생성
df = pd.DataFrame(data, columns=['Line1', 'Station1', 'Line2', 'Station2', 'Time'])

# DataFrame을 사용하여 그래프 생성
subway_graph = nx.Graph()
for index, row in df.iterrows():
    station1 = (row['Line1'], row['Station1'])
    station2 = (row['Line2'], row['Station2'])
    subway_graph.add_edge(station1, station2, weight=row['Time'])

# dijkstra
def dijkstra(graph, start_station, end_station):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_station] = 0
    predecessors = {node: None for node in graph.nodes}

    priority_queue = [(0, start_station)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node == end_station:
            break

        # 각 인접 노드 탐색
        for neighbor, properties in graph[current_node].items():
            distance = current_distance + properties['weight']

            # 새로운 경로가 더 나은 경우에만 고려
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    path = []
    step = end_station
    while step:
        path.append(step)
        step = predecessors[step]
    path.reverse()

    return path, distances[end_station]


# subway_graph가 정의되었고, start_station이 지정되었다고 가정
path, total_time = dijkstra(subway_graph, start_station, end_station)

# 시작점에서 모든 도달 가능한 역까지의 최단 경로 출력
print(f"Shortest path in dijkstra: {path}, Total travel time: {total_time} minutes")


# -----------------------시간, 공간 복잡도 비교--------------------
# MST 내에서 최단 경로를 찾는 함수
def find_shortest_path_mst(graph, start, end):
    path = nx.shortest_path(graph, source=start, target=end, weight='weight')
    path_length = nx.shortest_path_length(graph, source=start, target=end, weight='weight')
    return path, path_length

sort_functions = [find_shortest_path_mst, dijkstra]

def measure_memory_and_time(path_func, graph, start, end):
    tracemalloc.start()
    start_time = time.perf_counter()
    graph_copy = copy.deepcopy(graph)
    path_func(graph_copy, start, end)
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return end_time - start_time, peak / 1024


for sort_func in sort_functions:
    time_taken, peak_memory_kb = measure_memory_and_time(sort_func, subway_graph, start_station, end_station)
    print(f"{sort_func.__name__} took {time_taken:.6f} seconds and used {peak_memory_kb:.2f} KB at peak.")