import numpy as np
import matplotlib
matplotlib.use('TkAgg')


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import time



class AtomTypeException(Exception):
    def __str__(self):
        return "원자의 종류는 C 혹은 H 여야 합니다."

class AtomConnectionException(Exception):
    def __str__(self):
        return "4개 이상의 원자가 부착되려 합니다."

class SimpleMM:
    def __init__(self, carbon_matrix):
        """
        carbon_matrix: N x N 행렬 (연결 상태)
        N개의 탄소 원자의 연결 상태
        """

        # parameters
        self.params = {
            'C-C': {'r0': 1.54, 'k': 300.0, 'r_vanderwalls':3.55, 'eps':0.07},
            'C-H': {'r0': 1.09, 'k': 350.0, 'r_vanderwalls':2.99, 'eps':0.05},
            'H-H': {'r0': 0.64, 'r_vanderwalls':2.42, 'eps':0.03},
            'C' : {'r0': 0.77},
            'H' : {'r0': 0.32}
        }

        self.adj, self.atom = self.adj_matrix_CH(carbon_matrix) # 탄소원자 행렬을 수소원자가 포함된 행렬과 그 원자 종류로 반환
        self.pos = self.position(self.adj) # position method 를 활용해 행렬을 바탕으로 기본 원자 배열 생성

# 더 좋은 방법이 있어서 새로 짤 예정
    # def adj_matrix_CH(self, carbon_matrix):
    #     '''
    #     :param carbon_matrix: 탄소 원자의 연결 구조를 행렬로 받는다.
    #     :return: 수소 원자의 연결 구조를 포함하여 행렬로 반환한다., 행렬의 원자 종류 또한 반환한다.
    #     '''
    #
    #     N=len(carbon_matrix)
    #     bonds = (np.count_nonzero(carbon_matrix==1)-N)//2
    #
    #     N_containH = 4*N-2*bonds
    #     adj_matrix_with_H = carbon_matrix.copy()
    #     atoms = ['C']*N
    #
    #     atom_added = 0
    #     for i in range(N):
    #         adj_atom_num = 0
    #         for j in range(i+1, N):
    #             if carbon_matrix[i][j] == 1:
    #                 adj_atom_num += 1
    #         for j in range(3-adj_atom_num):
    #             insert_list = np.zeros(N+atom_added)
    #             insert_list[i+atom_added] = 1
    #             adj_matrix_with_H = np.insert(adj_matrix_with_H, i+atom_added+1,insert_list, axis=0)
    #
    #             insert_list = np.zeros(N+atom_added+1)
    #             insert_list[i+atom_added] = 1
    #             insert_list[i+atom_added+1] = 1
    #             adj_matrix_with_H = np.insert(adj_matrix_with_H, i+atom_added+1,insert_list, axis=1)
    #             atom_added += 1
    #
    #     return adj_matrix_with_H, atoms

    def adj_matrix_CH(self, carbon_matrix):
        """
        :param carbon_matrix: 탄소 원자의 연결 구조를 행렬로 받는다.
        :return: 수소 원자의 연결 구조를 포함하여 행렬로 반환한다., 행렬의 원자 종류 또한 반환한다.
        """

        # make length parametrs and assigng matrix to add hydrogen atom
        total_carbon=len(carbon_matrix)
        total_bond = (np.count_nonzero(carbon_matrix==1)-total_carbon)//2
        total_atom = 5*total_carbon - 2*total_bond
        total_atom_mat = np.zeros((total_atom, total_atom))
        total_atom_mat[0:total_carbon, :total_carbon] = carbon_matrix
        total_hydrogen = total_atom - total_carbon
        atom_list = ['C']*total_carbon + ['H'] * total_hydrogen

        #add Hydrogen atom
        num_connected_to_carbon = np.sum(carbon_matrix, axis=1) - 1

        #hydrogen comes next to carbon matrix
        h_indx_counter = total_carbon

        #add and connect hydrogen and carbon
        for i in range(total_carbon):
            for j in range(4 -num_connected_to_carbon[i]):
                total_atom_mat[h_indx_counter, h_indx_counter] = 1
                total_atom_mat[i,h_indx_counter] = 1
                total_atom_mat[h_indx_counter, i] = 1
                h_indx_counter += 1


        return total_atom_mat, atom_list

    def position(self, adj_matrix):
        """
        adj_matrix 를 바탕으로 다이아몬드 격자 구조를 바탕으로 초기 원자 좌표 생성
        isomer: 은 고려하지 않은 상태로 탄소 격자 생성하며 빈 공간에는 수소를 채워 넣는다
        결과적으로 N' 개의 행렬을 받아 N'*3 행렬을 반환한다.
        입력 행렬의 모양은 n개의 탄소 행렬, k개의 수소 행렬이 연결되어 있는 모양이다.
        """
        # connected_matrix 를 새로고침하면서 원자 연결을 기억

        #각 원자 좌표를 넣을 행렬 생성
        N = len(adj_matrix)
        atomic_position = np.zeros((N,3)) #n, 3 atomic coordinate matrix
        connected_matrix = np.zeros((N,5)) # 0~ 3 is is it connected, 4 is type of connected. it assume diamond-type connection that have positive and negative direction pyramidal shpe connection

        #BFS
        visited = np.zeros(N, dtype=bool)
        start_node = 0
        visited[start_node] = True
        connected_matrix[start_node][4] = 1

        #make queue
        queue = deque([start_node])

        #loop while queue is empty
        while queue:
            current = queue.popleft()
            neighbors = np.where(adj_matrix[current] == 1)[0]

            for neighbors in neighbors:
                if not visited[neighbors]:
                    self._add_position(current, neighbors, atomic_position, connected_matrix)
                    visited[neighbors] = True
                    queue.append(neighbors)

        return atomic_position


    def _add_position(self, original_atom_idx, added_atom_idx, atomic_position, connected_matrix):
        original_atom_coord = atomic_position[original_atom_idx] #1st coord
        added_atom_coord = original_atom_coord.copy() #2nd coord (will change)
        original_atom_type = self.atom[original_atom_idx] # C or H
        added_atom_type = self.atom[added_atom_idx] # C or H

        r = self.params[original_atom_type]['r0'] + self.params[added_atom_type]['r0'] #calculate interatomic distance
        theta = np.arccos(-1.0 / 3.0) #rad, 109.5 degree
        direction = connected_matrix[original_atom_idx][4] #direction of pyrimid

        #dz = - r * cos(theta)
        #dy for 1 = - r * sin(theta)
        #dy for 2

        if connected_matrix[original_atom_idx][0] == 0:
            added_atom_coord[1] -= r * np.sin(theta) * direction # change y coordinate
            added_atom_coord[2] -= r * np.cos(theta) * direction#0 position, only change of y, z coordinate
            connected_matrix[original_atom_idx][0] = 1
            connected_matrix[added_atom_idx][0] = 1
            connected_matrix[added_atom_idx][4] = -direction

        elif connected_matrix[original_atom_idx][1] == 0:
            added_atom_coord[0] += r * np.sin(theta) * np.sqrt(3) / 2 * direction
            added_atom_coord[1] += r * np.sin(theta) * np.sqrt(2) / 2 * direction
            added_atom_coord[2] -= r * np.cos(theta) * direction
            connected_matrix[original_atom_idx][1] = 1
            connected_matrix[added_atom_idx][1] = 1
            connected_matrix[added_atom_idx][4] = -direction

        elif connected_matrix[original_atom_idx][2] == 0:
            added_atom_coord[0] -= r * np.sin(theta) * np.sqrt(3) / 2 * direction
            added_atom_coord[1] += r * np.sin(theta) * np.sqrt(2) / 2 * direction
            added_atom_coord[2] -= r * np.cos(theta) * direction
            connected_matrix[original_atom_idx][2] = 1
            connected_matrix[added_atom_idx][2] = 1
            connected_matrix[added_atom_idx][4] = -direction

        elif connected_matrix[original_atom_idx][3] == 0:
            added_atom_coord[2] += r * direction
            connected_matrix[original_atom_idx][3] = 1
            connected_matrix[added_atom_idx][3] = 1
            connected_matrix[added_atom_idx][4] = -direction

        else:
            raise AtomTypeException

        atomic_position[added_atom_idx] = added_atom_coord + np.random.uniform(low=0, high=0.01, size=3)

        return

    def get_distance(self, i, j):
        distance = np.linalg.norm(self.pos[i] - self.pos[j])
        distance = np.clip(distance, a_min=0.01, a_max=np.inf)
        return distance

    def calculate_energy(self): # test requier
        total_energy = 0.0
        N = len(self.pos)

        # 1. Bond Energy (연결된 쌍)
        for i in range(N):
            for j in range(i + 1, N):
                if self.adj[i][j] == 1: # 결합이 있으면
                    dist = self.get_distance(i, j)
                    # 원자 타입에 맞는 파라미터 로드
                    adj_atom_type = ''
                    if self.atom[i] == 'C' and self.atom[j] == 'H':
                        adj_atom_type = 'C-C'
                    elif self.atom[i] == 'C' and self.atom[j] == 'C':
                        adj_atom_type = 'C-H'
                    else:
                        raise AtomTypeException #if atom are no H or C, raise error

                    r0 = self.params[adj_atom_type]['r0']
                    k = self.params[adj_atom_type]['k']

                    total_energy += 0.5 * k * (dist - r0)**2 #spring-type repulsion
        # 2. Repulsion Energy (연결 안 된 쌍)
        # 루프 돌면서 Lennard-Jones 포텐셜 추가
        for i in range(N):
            for j in range(i + 1, N):
                if self.adj[i][j] == 0:
                    dist = self.get_distance(i, j)
                    adj_atom_type = ''
                    if self.atom[i] == 'C' and self.atom[j] == 'H':
                        adj_atom_type = 'C-C'
                    elif self.atom[i] == 'H' and self.atom[j] == 'H':
                        adj_atom_type = 'H-H'
                    elif self.atom[i] == 'C' and self.atom[j] == 'C':
                        adj_atom_type = 'C-H'
                    else:
                        raise AtomTypeException #if atom are no H or C, raise error

                    r0 = self.params[adj_atom_type]['r0']
                    rv = self.params[adj_atom_type]['r_vanderwalls'] #van der waals radius
                    eps = self.params[adj_atom_type]['eps']
                    #calculate LJ potential
                    total_energy += 4 * eps * ((rv/dist)**12 - (rv/dist)**2)

        return total_energy

    def calculate_forces_numerical(self, delta=1e-3): #requier test
        # 수치 미분으로 힘(Gradient) 계산
        # F = - (E(x+d) - E(x-d)) / 2d
        forces = np.zeros_like(self.pos) #
        original_coords = self.pos.copy()

        for i in range(forces.shape[0]):
            for j in range(3):
                self.pos[i, j] += delta
                energy_1 = self.calculate_energy() #E(x+d)
                self.pos[i, j] -= 2 * delta
                energy_2 = self.calculate_energy() #E(x-d)
                self.pos = original_coords.copy()
                forces[i, j] = -(energy_1 - energy_2)/ 2 * delta
        return forces

    def optimize(self, steps=20000, learning_rate=30):
        step = 0
        energy_0 = 0
        energy_1 = 10
        while abs(energy_0 - energy_1) > 1e-4 and step < steps:
            energy_0 = self.calculate_energy()
            forces = self.calculate_forces_numerical()
            self.pos += forces * learning_rate # Gradient Descent
            energy_1 = self.calculate_energy()
            if step % 10 == 0:
                print(f"Step {step}, Energy: {energy_1}")
            step += 1


    plt.ion()
    def optimize_with_3D(self, steps=500, learning_rate=100):
        step = 0
        energy_0 = 0
        energy_1 = 10

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')


        while abs(energy_0 - energy_1) > 1e-3 and step < steps:
            energy_0 = self.calculate_energy()
            forces = self.calculate_forces_numerical()
            self.pos += forces * learning_rate # Gradient Descent
            energy_1 = self.calculate_energy()
            if step % 5 == 0:
                self.update_plot(ax, step, energy_1)
                plt.draw()
                plt.pause(0.0001)
                if step <10:
                    plt.pause(0.2)
            if energy_0 - energy_1 < 5 and learning_rate <500:
                learning_rate *= 1.02


            step += 1

        plt.ioff()
        plt.show(block=True)



    def update_plot(self, ax, step, energy):
        ax.cla()
        limit = 10.0


        colors = ['black' if a == 'C' else 'blue' for a in self.atom]
        sizes = [limit*20 if a == 'C' else limit*5 for a in self.atom]
        ax.scatter(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2], c=colors, s=sizes, edgecolor = 'black',depthshade=True)

        num_atoms = len(self.atom)
        rows, cols = np.nonzero(np.triu(self.adj-np.eye(num_atoms)))

        for r, c in zip(rows, cols):
            ax.plot([self.pos[r, 0], self.pos[c,0]], [self.pos[r, 1], self.pos[c,1]], [self.pos[r, 2], self.pos[c,2]], color = 'gray', linewidth = 2, alpha = 0.5)
        ax.set_title(f"Optimization Step: {step}\nEnergy: {energy:.4f} kcal/mol")

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        return
