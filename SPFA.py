import pandas as pd
from termData import data
import networkx as nx
from collections import deque
import time
import numpy as np

# 데이터프레임 생성
df = pd.DataFrame(data, columns=['Line1', 'Station1', 'Line2', 'Station2', 'Time'])

# 그래프 생성
subway_graph = nx.Graph()
for index, row in df.iterrows():
    station1 = (row['Line1'], row['Station1'])  # 첫 번째 역 (호선, 역 이름) 튜플
    station2 = (row['Line2'], row['Station2'])  # 두 번째 역 (호선, 역 이름) 튜플
    subway_graph.add_edge(station1, station2, weight=row['Time'])  # 두 역 사이에 간선 추가, 가중치는 시간

# SPFA 알고리즘 정의
# string으로 접근하고 상수시간복잡도로 접근할 수 있어 딕셔너리 형태 사용
def spfa(graph, start_station, end_station):
    distances = {node: float('inf') for node in graph}  # 모든 노드의 초기 거리를 무한대로 설정
    is_in_queue = {node: False for node in graph}  # 노드가 큐에 있는지 여부를 저장하는 딕셔너리
    predecessors = {node: None for node in graph}  # 최단 경로 추적을 위한 이전 노드 저장
    distances[start_station] = 0  # 시작 노드의 거리는 0으로 설정
    queue = deque([start_station])  # 시작 노드를 큐에 추가
    is_in_queue[start_station] = True  # 시작 노드를 큐에 있다고 표시

    # 큐가 빌 때까지 반복
    while queue:
        cur_node = queue.popleft()  # 큐에서 노드를 꺼냄
        is_in_queue[cur_node] = False  # 노드가 큐에 없다고 표시
        # 현재 노드에서 인접 노드까지 거리 계산해서 현재 알려진 거리 보다 짧으면 업데이트
        for neighbor, time in graph[cur_node].items():
            distance = distances[cur_node] + time['weight']  # 인접 노드까지의 거리 계산
            if distance < distances[neighbor]:  # 새로운 거리가 기존 거리보다 짧으면
                distances[neighbor] = distance  # 거리 업데이트
                predecessors[neighbor] = cur_node  # 이전 노드 업데이트
                # 인접 노드가 큐에 없으면 추가
                if not is_in_queue[neighbor]:
                    queue.append(neighbor)
                    is_in_queue[neighbor] = True

    # 최단 경로 추적
    path = []
    step = end_station
    while step:
        path.append(step)  # 경로에 노드 추가
        step = predecessors[step]  # 이전 노드로 이동
    path.reverse()  # 경로를 역순으로

    return path, distances[end_station]  # 최단 경로와 총 거리를 반환

# 출발역과 도착역 입력 받기
start = tuple(map(str.strip, input("출발역의 호선과 이름을 입력하세요 EX) 1 시청: ").split()))  # 출발역 입력 받기
end = tuple(map(str.strip, input("도착역의 호선과 이름을 입력하세요 EX) 1 시청: ").split()))  # 도착역 입력 받기

time_list = []
for i in range(100):
    start_time = time.time()  # 시작 시간 기록
    path, total_time = spfa(subway_graph, start, end)  # SPFA 알고리즘 실행
    end_time = time.time()  # 종료 시간 기록
    sub = end_time - start_time  # 실행 시간 계산
    time_list.append(sub)  # 실행 시간 리스트에 추가

# 최단 경로와 시간 출력
print(f"최적의 경로: {path},\n예상 시간: {total_time}분")  # 최단 경로와 총 이동 시간 출력
print(f"실제 실행 시간: {np.mean(time_list):.6f}")  # 평균 실행 시간 출력