
# coding: utf-8

# In[1]:

import numpy as np
import time
from operator import itemgetter


# In[2]:

POP_SIZE = 100
CULL_TOP = 15
CULL_BOT = 5
OFFSPRING_COUNT = POP_SIZE - CULL_TOP - CULL_BOT

GENE_SIZE = 16
GENOME_SIZE = 16

GENERATIONS = 1000


# In[3]:

EVAL_LENGTH = 1000
LOWCUT_IND = 249
HIGHCUT_IND = 449
EVAL_FREQS = [2**((step+1-EVAL_LENGTH)/(step+1))*2*np.pi for step in range(EVAL_LENGTH)]
IDEAL_RESP = [0]*LOWCUT_IND + [1]*(HIGHCUT_IND-LOWCUT_IND) + [0]*(EVAL_LENGTH-HIGHCUT_IND)


# In[4]:

class Gene:
    def __init__(self, dna=None):
        if dna is None:
            dna = np.random.randn(GENE_SIZE)
        self.dna = dna
    
    def mutate(self):
        targets = np.random.choice(2,GENE_SIZE)
        mutation = np.multiply(np.random.randn(GENE_SIZE),targets)
        self.dna += mutation


# In[5]:

class Organism:
    def __init__(self, genome=None):
        if genome is None:
            genome = [Gene() for i in range(GENOME_SIZE)]
        self.genome = genome
        self.phenotype = self.translate()
    
    def mutate(self):
        target_count = np.random.randint(16)
        targets = np.random.choice(16,target_count)
        for i in targets:
            self.genome[i].mutate()
        self.phenotype = self.translate()
        
    def translate(self):
        raw_seq = np.concatenate([gene.dna for gene in self.genome])
        return np.concatenate([raw_seq, np.flip(raw_seq,0)])
        


# In[8]:

class Generation:
    def __init__(self):
        self.population = [Organism() for i in range(POP_SIZE)]

    @staticmethod
    def z_transform(fir):
        return lambda w : np.sum([fir[n]*(np.cos(n*w)-np.sin(n*w)*1j) for n in range(len(fir))])

    @staticmethod
    def generate_mag_response(Hz, frequencies=EVAL_FREQS):
        for ind,freq in enumerate(frequencies):
            yield (ind, np.absolute(Hz(freq)))

    @staticmethod
    def total_sse(Hz, frequencies=EVAL_FREQS, ideal_resp=IDEAL_RESP):
        sse = 0       
        for freq in frequencies:
            sse += np.absolute(Hz(freq))**2
        return sse
        
    # looks at list of current best candidates, finds which one is worst and returns its fitness and index
    @staticmethod
    def status_quo(current_top):
        _, cost = zip(*current_top)
        threshold = max(cost)
        hotspot = cost.index(threshold)
        return hotspot, threshold
        
    # culls the population and returns a list of surviving candidates
    def cull(self):
    # top performers, guaranteed survival
        top = []
    # the rest, random subset will survive
        bottom = []

        for tag, candidate in enumerate(self.population):
            fir = candidate.phenotype
            fir_z = self.z_transform(fir)
    # before we have a full set of top candidates
            if tag < CULL_TOP:
                top.append((candidate, self.total_sse(fir_z)))
            else:
    # start replacing top candidates as they come
                hotspot, threshold = self.status_quo(top)
                sse = 0
                step = 0
                keep = True
                while step < EVAL_LENGTH:
                    step, mag = self.generate_mag_response(fir_z)
                    sse += (IDEAL_RESP[step]-mag)**2
    # worse than status quo
                    if sse > threshold:
                        keep = False
                        bottom.append(candidate)
                        break
                if keep:
                    bottom.append(top[hotspot][0])
                    top[hotspot] = (candidate, score)

        fittest, = zip(*top)
        lucky = np.random.choice(bottom,CULL_BOT,replace=False)

        self.population = fittest+lucky
        
    def cull_2(self):
        fitness_scores = [self.total_sse(self.z_transform(candidate.phenotype)) for candidate in self.population]
        ranked,  = zip(*sorted(zip(self.populatioin,fitness_scores), key=itemgetter(1)))
        fittest = ranked[0:CULL_TOP-1]
        lucky = np.random.choice(ranked[CULL_TOP:-1], CULL_BOT, replace=False)
        
        self.population = fittest+lucky
    
    @staticmethod
    def cross(genomes):       
        return [genomes[np.random.randint(2)][gene] for gene in range(GENOME_SIZE)]
    
    @staticmethod
    def mate(parents):
        new_genome = self.cross((parents[0].genome(),parents[1].genome()))
        return Organism(new_genome)

    def breed(self, offspring_count=OFFSPRING_COUNT):
        self.population = [self.mate(np.random.choice(self.population,2)) for count in range(offspring_count)]
  
    def mutate(self):
        target_count = np.random.randint(POP_SIZE)
        targets = np.random.choice(POP_SIZE,target_count)
        for i in targets:
            self.population[i].mutate()
            
    def cycle(self):
        self.cull_2()
        self.breed()
        self.mutate()
        


# In[ ]:

test_gen = Generation()
start = time.time()
print('One Cycle')
test_gen.cycle()
end = time.time()
print(end-start)


# In[ ]:




# In[ ]:



