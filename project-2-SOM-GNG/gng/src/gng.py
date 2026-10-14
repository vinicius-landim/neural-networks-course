import numpy as np
import cv2

class GrowingNeuralGas:
    def __init__(self, max_nodes=64, eta_b=10e-2, eta_n=10e-4, max_age=90, alpha=0.5, beta=0.005, lambda_step=50):
        self.max_nodes = max_nodes
        self.eta_b = eta_b
        self.eta_n = eta_n
        self.max_age = max_age
        self.alpha = alpha
        self.beta = beta
        self.lambda_step = lambda_step
        
        self.nodes = np.zeros((max_nodes, 3), dtype=np.float32)
        self.errors = np.zeros(max_nodes, dtype=np.float32)
        self.is_active = np.zeros(max_nodes, dtype=bool)
        
        # matrizes de adjacência
        self.edges = np.zeros((max_nodes, max_nodes), dtype=bool)
        self.ages = np.zeros((max_nodes, max_nodes), dtype=np.int32)
        
        self.num_nodes = 0
        
    def fit(self, pixels, epochs=1):
        #iniciar a rede N com dois neurônios c1 e c2
        idx = np.random.choice(len(pixels), 2, replace=False)
        self.nodes[0] = pixels[idx[0]]
        self.nodes[1] = pixels[idx[1]]
        
        self.is_active[0] = True
        self.is_active[1] = True
        self.num_nodes = 2
        
        #iniciar o conjunto de conexões C
        self.edges[0, 1] = True
        self.edges[1, 0] = True
        
        iteration = 0
        
        for epoch in range(epochs):
            rand_idx = np.random.permutation(len(pixels))
            
            for idx in rand_idx:
                #seleciona um padrão xi e o apresenta a rede
                p = pixels[idx]
                
                #filtra apenas os nós ativos via máscara booleana
                active_idx = np.where(self.is_active)[0]
                if len(active_idx) < 2:
                    continue
                    
                active_nodes = self.nodes[active_idx]
                
                #processo competitivo
                #determinar os dois neurônios mais próximos
                dists = np.linalg.norm(p - active_nodes, axis=1)
                sorted_local = np.argsort(dists)
                
                #mapear de volta para os índices globais
                s1 = active_idx[sorted_local[0]]
                s2 = active_idx[sorted_local[1]]
                
                #adicionar o quadrado da distância entre o neurônio vencedor e o padrão xi a uma variável de erro local Es1
                self.errors[s1] += dists[sorted_local[0]]**2
                
                #atualizar os pesos do neurônio vencedor s1 e seus vizinhos
                self.nodes[s1] += self.eta_b * (p - self.nodes[s1])
                
                #encontra vizinhos varrendo a linha de s1 na matriz de adjacência
                neighbors = np.where(self.edges[s1] & self.is_active)[0]
                for n in neighbors:
                    self.nodes[n] += self.eta_n * (p - self.nodes[n]) #atualiza os vizinhos
                    #ncrementar a idade de todas as conexões de s1
                    self.ages[s1, n] += 1
                    self.ages[n, s1] += 1
                    
                #se não existir uma conexão entre s1 e s2, então criá-la
                self.edges[s1, s2] = True
                self.edges[s2, s1] = True
                self.ages[s1, s2] = 0
                self.ages[s2, s1] = 0
                
                #remover as conexões com idade maior que amax
                old_edges = np.where((self.ages[s1] > self.max_age) & self.edges[s1])[0]
                for n in old_edges:
                    self.edges[s1, n] = False
                    self.edges[n, s1] = False
                
                #remover neurônios sem conexão
                for i in active_idx:
                    if not np.any(self.edges[i]):
                        self.is_active[i] = False
                        self.num_nodes -= 1
                
                #caso a condição seja satisfeita, inserir um novo neurônio a rede
                iteration += 1
                if iteration % self.lambda_step == 0 and self.num_nodes < self.max_nodes:
                    self._insert_node()
                    
                #diminuir o erro E de todas as unidades
                self.errors[self.is_active] -= self.beta * self.errors[self.is_active]

    def _insert_node(self):
        active_idx = np.where(self.is_active)[0]
        if len(active_idx) == 0: 
            return
        
        #determinar o neurônio q com o maior erro acumulado
        u = active_idx[np.argmax(self.errors[active_idx])]
        
        #dentre os vizinhos de q, determinar o neurônio f com maior erro acumulado
        neighbors = np.where(self.edges[u] & self.is_active)[0]
        if len(neighbors) == 0:
            return
        v = neighbors[np.argmax(self.errors[neighbors])]
        
        #encontra o primeiro índice vazio disponível no array
        free_slots = np.where(~self.is_active)[0]
        if len(free_slots) == 0:
            return
        w = free_slots[0]
        
        #adicionar a nova unidade r a rede interpolando seu vetor de pesos a partir de q e f
        self.nodes[w] = (self.nodes[u] + self.nodes[v]) / 2.0
        self.is_active[w] = True
        self.num_nodes += 1
        
        #inserir conexões entre r-q, r-f e remover conexão q-f
        self.edges[u, v] = False
        self.edges[v, u] = False
        
        self.edges[u, w] = True
        self.edges[w, u] = True
        self.edges[v, w] = True
        self.edges[w, v] = True
        
        self.ages[u, w] = self.ages[w, u] = 0
        self.ages[v, w] = self.ages[w, v] = 0
        
        #diminuir o erro E das unidades q e f
        self.errors[u] *= self.alpha
        self.errors[v] *= self.alpha
        
        #configurar o erro de r a partir dos erros de q e f
        self.errors[w] = self.errors[u]