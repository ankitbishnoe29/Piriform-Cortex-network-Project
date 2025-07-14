#!/usr/bin/env python
# coding: utf-8
from neuron import h
import random
import ast
from collections import Counter
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpi4py import MPI
comm = MPI. COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

h.load_file("stdrun.hoc")
from neuron.units import µm,mV,ms



with open('SpikeTimes_2hz_correlated.txt', 'r') as file:
    spike_times = file.read()
Spike_times_2hz = ast.literal_eval(spike_times)
with open('SpikeTimes_20hz_correlated.txt', 'r') as file:
    spike_times = file.read()
Spike_times_20hz = ast.literal_eval(spike_times)
with open('SpikeTimes_A_odour_2hz.txt', 'r') as file:
    spike_times = file.read()
Spike_times_A2hz = ast.literal_eval(spike_times)
with open('SpikeTimes_A_odour_20hz.txt', 'r') as file:
    spike_times = file.read()
Spike_times_A20hz = ast.literal_eval(spike_times)
with open('SpikeTimes_B_odour_2hz.txt', 'r') as file:
    spike_times = file.read()
Spike_times_B2hz = ast.literal_eval(spike_times)
with open('SpikeTimes_B_odour_20hz.txt', 'r') as file:
    spike_times = file.read()
Spike_times_B20hz = ast.literal_eval(spike_times)

Spike_times_list = [Spike_times_2hz,Spike_times_20hz,Spike_times_A2hz,Spike_times_A20hz,Spike_times_B2hz,Spike_times_B20hz]

Odours_name = ["2 hz correlated", "20 hz correlated", "A odour 2 Hz","A odour 20 Hz","B odour 2 Hz","B odour 20 Hz"]


class pyramidal:
    def __init__(self,gid):
        self._gid=gid
        self.morphology()
        self.biophysics()

        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)
        self.soma_v = h.Vector().record(self.soma(0.5)._ref_v)

    def morphology(self):
        self.soma=h.Section(name="soma",cell=self)
        self.all=self.soma.wholetree()
        self.soma.diam= 29*µm
        self.soma.L=29*µm
        
        
        self.dend1 = h.Section(name="dend1", cell=self) 
        self.dend1.diam = 2.57
        self.dend1.L = 100*µm
        self.dend2 = h.Section(name="dend2", cell=self) 
        self.dend2.diam = 1.77
        self.dend2.L = 100*µm
        self.dend3 = h.Section(name="dend3", cell=self) 
        self.dend3.diam = 1.2
        self.dend3.L = 100*µm
        self.dend4 = h.Section(name="dend4", cell=self) 
        self.dend4.diam = 0.8
        self.dend4.L = 100*µm
        self.dend5 = h.Section(name="dend5", cell=self) 
        self.dend5.diam = 0.5
        self.dend5.L = 100*µm
    
        self.dend1.connect(self.soma(1))
        self.dend2.connect(self.dend1(1)) 
        self.dend3.connect(self.dend2(1)) 
        self.dend4.connect(self.dend3(1)) 
        self.dend5.connect(self.dend4(1)) 
                
    def biophysics(self):
        
        self.soma.insert("nax")
        for sec in self.soma:
           sec.nax.gbar= 0.5 #gnabar           
        self.soma.insert("kdr")
        for sec in self.soma:
           sec.kdr.gkdrbar=0.06 #gkbar  
        self.soma.insert("leak")
        for sec in self.soma:
          sec.leak.g= 0.0003 # S/cm^2 
          sec.leak.e= -75*mV 

        self.dend1.insert("nax")
        for sec in self.dend1:
           sec.nax.gbar= 0.5  #gnabar       
        self.dend1.insert("kdr")
        for sec in self.dend1:
           sec.kdr.gkdrbar=0.06 #gkbar 
        self.dend1.insert("leak")
        for sec in self.dend1:
          sec.leak.g= 0.0003 # S/cm^2
          sec.leak.e= -75*mV  
            
        self.dend2.insert("nax")
        for sec in self.dend2:
           sec.nax.gbar=0.5 #gnabar         
        self.dend2.insert("kdr")
        for sec in self.dend2:
           sec.kdr.gkdrbar=0.06 #gkbar 
        self.dend2.insert("leak")
        for sec in self.dend2:
          sec.leak.g= 0.0003 # S/cm^2
          sec.leak.e= -75*mV 
            
        self.dend3.insert("nax")
        for sec in self.dend3:
           sec.nax.gbar= 0.5 #gnabar   
        self.dend3.insert("kdr")
        for sec in self.dend3:
           sec.kdr.gkdrbar=0.06 #gkbar  
        self.dend3.insert("leak")
        for sec in self.dend3:
          sec.leak.g= 0.0003 # S/cm^2
          sec.leak.e= -75*mV 
            
        self.dend4.insert("nax")
        for sec in self.dend4:
           sec.nax.gbar= 0.5 #gnabar         
        self.dend4.insert("kdr")
        for sec in self.dend4:
           sec.kdr.gkdrbar=0.06 #gkbar   
        self.dend4.insert("leak")
        for sec in self.dend4:
          sec.leak.g= 0.0003 # S/cm^2
          sec.leak.e= -75*mV  
            
        self.dend5.insert("nax")
        for sec in self.dend5:
           sec.nax.gbar= 0.5 #gnabar         
        self.dend5.insert("kdr")
        for sec in self.dend5:
           sec.kdr.gkdrbar=0.06  #gkbar 
        self.dend5.insert("leak")
        for sec in self.dend5:
          sec.leak.g= 0.0003 # S/cm^2
          sec.leak.e= -75*mV 
            
        for sec in self.all:
           sec.Ra=173 #ohm*cm
           sec.cm=1   #µFarad/cm^2    

    def __repr__(self):
       return "pyramidal[{}]".format(self._gid)



class FSI:
    def __init__(self,gid):
        self._gid=gid
        self.morphology()
        self.biophysics()

        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)
        self.soma_v = h.Vector().record(self.soma(0.5)._ref_v)

    def morphology(self):
        self.soma=h.Section(name="soma",cell=self)
        self.all=self.soma.wholetree()
        self.soma.diam= 68*µm
        self.soma.L=68*µm
        
                
    def biophysics(self):
        
        self.soma.insert("wangbuzaki")
        for sec in self.soma:
          sec.wangbuzaki.gna= 0.05 # S/cm^2
          sec.wangbuzaki.ena= 55*mV
          sec.wangbuzaki.gk= 0.01  # S/cm^2
          sec.wangbuzaki.ek= -90*mV
          sec.wangbuzaki.gl= 0.00014 # S/cm^2
          sec.wangbuzaki.el= -78*mV        
            
        for sec in self.all:
           sec.Ra=173 #ohm*cm
           sec.cm=1   #µFarad/cm^2    
        
    def __repr__(self):
        return "FSI[{}]".format(self._gid)

class FF:
    def __init__(self,gid):
        
        self._gid=gid
        self.morphology()
        self.biophysics()

        self._spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec=self.soma)
        self.spike_times = h.Vector()
        self._spike_detector.record(self.spike_times)
        self.soma_v = h.Vector().record(self.soma(0.5)._ref_v)

    def morphology(self):
        
        self.soma=h.Section(name="soma",cell=self)
        self.all=self.soma.wholetree()
        self.soma.diam = 85*µm
        self.soma.L = 85*µm       
                
    def biophysics(self):
        
        self.soma.insert("wangbuzaki")
        for sec in self.soma:
          sec.wangbuzaki.gna= 0.05 # S/cm^2
          sec.wangbuzaki.ena= 55*mV
          sec.wangbuzaki.gk= 0.015  # S/cm^2
          sec.wangbuzaki.ek= -90*mV
          sec.wangbuzaki.gl= 0.000048 # S/cm^2
          sec.wangbuzaki.el= -72*mV        
            
        for sec in self.all:
           sec.Ra=173 #ohm*cm
           sec.cm=1   #µFarad/cm^2    
        
    def __repr__(self):
        return "FF[{}]".format(self._gid)
            

# Creating the population of pyramidal and fast spiking interneurons:

pyrs=[pyramidal(i) for i in range(500)]
Fsis=[FSI(i) for i in range(50)]
Ffs = [FF(i) for i in range(50)]
  
t=h.Vector().record(h._ref_t)  

# Save current random states
numpy_state = np.random.get_state()
python_state = random.getstate()

    
#@@@@@@@@@@@@@@@@@@@ Creating noise in every cell of the network: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


noise_list=[]

for Fsi in Fsis:
    noise=h.InGauss(Fsi.soma(0.5))
    noise.mean=0
    noise.stdev=  5 
    noise.seed(random.randint(0, 1000000))
    noise_list.append(noise)

for pyr in pyrs:   
   soma, dend1, dend2, dend3, dend4, dend5 = (pyr.soma(0.5), pyr.dend1(0.5), pyr.dend2(0.5),pyr.dend3(0.5), pyr.dend4(0.5), pyr.dend5(0.5))
   for comp in (soma, dend1, dend2, dend3, dend4, dend5):
      noise = h.InGauss(comp)
      noise.mean = 0
      noise.stdev = 0.09 #0.092 is control
      noise.seed(random.randint(0, 1000000))
      noise_list.append(noise)

for Ff in Ffs:
    noiseF=h.InGauss(Ff.soma(0.5))
    noiseF.mean=0
    noiseF.stdev = 2.3
    noiseF.seed(random.randint(0, 1000000))
    noise_list.append(noiseF)


h.finitialize()
h.continuerun(6000*ms)
h.celsius=33
h.tstop=6000*ms 

# Set the seed number
random.seed(42)
np.random.seed(42)

# Creating synapses on the Pyramidal cells and Fast spiking Interneurons and FeedForward Interneurons:

SynsPyr = []
pyr_syn_idx = {}
syn_to_pyr_idx = {}

for pyr_idx, pyr in enumerate(pyrs):
    Syns = [syn for dend in [pyr.dend1, pyr.dend2, pyr.dend3, pyr.dend4, pyr.dend5] for syn in [h.Exp2Syn(dend(i)) for i in np.linspace(0, 1, 100)]]
    for syn in Syns:
        syn_idx = len(SynsPyr)
        pyr_syn_idx[syn] = syn_idx
        syn_to_pyr_idx[syn] = pyr_idx
        SynsPyr.append(syn)


SynsFsi=[]
fsi_syn_idx={}
syn_to_fsi_idx = {}

for fsi_idx, Fsi in enumerate(Fsis):
  for syN in [h.Exp2Syn(Fsi.soma(i)) for i in np.linspace(0, 1, 120)]:
      syn_idx = len(SynsFsi)
      fsi_syn_idx[syN] = syn_idx
      syn_to_fsi_idx[syN] = fsi_idx
      SynsFsi.append(syN)
      

SynsFf=[]
ff_syn_idx={}
syn_to_ff_idx = {}

for ff_idx, Ff in enumerate(Ffs):
  for syN in [h.Exp2Syn(Ff.soma(i)) for i in np.linspace(0, 1, 100)]:
      syn_idx = len(SynsFf)
      ff_syn_idx[syN] = syn_idx
      syn_to_ff_idx[syN] = ff_idx
      SynsFf.append(syN)



#@@@@@@@@@@@@@@@@@ Creating associative connections among Pyramidal and Fast spiking Interneurons:@@@@@@@@@@@@@@@@@


# ******** Pyr-Pyr excitatory ***********

synsPyr = []
netconsPyr = []

for idx, cell in enumerate(pyrs):
    
    all_synapses = [syn for syn, pyr_idx in syn_to_pyr_idx.items() if pyr_idx == idx]       # Select all syns which map to idx of given "cell".
    dend5_synapses =   all_synapses[int(len(all_synapses)*0.8):int(len(all_synapses) * 1)]
 
    available_sources = [presyn for presyn in pyrs if presyn != cell ]          # excluding the current cell
    num_sources = int(len(available_sources) * 0.2)                      # 0.2 corresponds to 20%
    source_indices = np.random.choice(range(len(available_sources)), size=num_sources, replace=False)
    sources = [available_sources[i] for i in source_indices]
    
    for source in sources:
        available_target = [syn for syn in all_synapses if syn not in dend5_synapses if syn not in synsPyr] 
        syn = SynsPyr[pyr_syn_idx[random.sample(available_target,1)[0]]]
        syn.tau1 = 0.24 * ms
        syn.tau2 = 3.6 * ms
        syn.e = 0 * mV
        nc = h.NetCon(source.soma(0.5)._ref_v, syn, sec=source.soma)
        nc.weight[0] =  0.00002 
        nc.delay = 1.54 * ms
        netconsPyr.append(nc)
        synsPyr.append(syn)

# ***** FSIN-FSIN inhibitory ******

synsFsi = []
netconsFsi = []
for idx, cell in enumerate(Fsis):
    
    available_sources = [presyn for presyn in Fsis if presyn != cell ]          # excluding the current cell
    num_sources = int(len(available_sources) * 0.1)                             # 0.1 corresponds to 10%
    source_indices = np.random.choice(range(len(available_sources)), size=num_sources, replace=False)
    sources = [available_sources[i] for i in source_indices]

    for source in sources:
        available_target = [syn for syn, fsi_idx in syn_to_fsi_idx.items() if fsi_idx == idx if syn not in synsFsi]
        syn = SynsFsi[fsi_syn_idx[random.sample(available_target,1)[0]]]
        syn.tau1 = 0.2 * ms
        syn.tau2 = 1.2 * ms
        syn.e = -80 * mV                  # Inhibitory Synpase 
        nc = h.NetCon(source.soma(0.5)._ref_v, syn, sec=source.soma)
        nc.weight[0] = 0.06 
        nc.delay = 1.54 * ms
        netconsFsi.append(nc)
        synsFsi.append(syn)

 # ***** FF-FF inhibitory ******

synsFf = []
netconsFf = []
for idx, cell in enumerate(Ffs):
    
    available_sources = [presyn for presyn in Ffs if presyn != cell ]          # excluding the current cell
    num_sources = int(len(available_sources) * 0.4)                           # 0.4 corresponds to 40%
    source_indices = np.random.choice(range(len(available_sources)), size=num_sources, replace=False)
    sources = [available_sources[i] for i in source_indices]

    for source in sources:
        available_target = [syn for syn, ff_idx in syn_to_ff_idx.items() if ff_idx == idx if syn not in synsFf]
        syn = SynsFf[ff_syn_idx[random.sample(available_target,1)[0]]]
        syn.tau1 = 0.2 * ms
        syn.tau2 = 1.2 * ms
        syn.e = -80 * mV                  # Inhibitory Synpase 
        nc = h.NetCon(source.soma(0.5)._ref_v, syn, sec=source.soma)
        nc.weight[0] = 0.0095
        nc.delay = 1.54 * ms
        netconsFf.append(nc)
        synsFf.append(syn)
        

# @@@@@@@@@@@ Creating reciprocal connections between PYR & FSI & FF @@@@@@@@@@@@@@@ :

# ****** FSIN-Pyr ***********
synsFb_Pyr = []
netconsFb_Pyr = []
for pyr in pyrs:
    
    somatic_synapses = [h.Exp2Syn(pyr.soma(i)) for i in np.linspace(0, 1, 40)] # Adjust according to population of FSI
    available_sources = [presyn for presyn in Fsis] 
    num_sources = int(len(available_sources) * 0.35)   # 35% of interneurons network
    source_indices = np.random.choice(range(len(available_sources)), size=num_sources, replace=False)
    sources = [available_sources[i] for i in source_indices]

    for source in sources:
        available_targets = [syn for syn in somatic_synapses if syn not in synsFb_Pyr]  # Avoid creating multiple synapses on same location
        syn =  available_targets[random.sample(range(len(available_targets)),k=1)[0]]
        syn.tau1 = 0.2 * ms
        syn.tau2 = 1.2 * ms
        syn.e = -80 * mV                                                   # Inhibitory connections
        nc = h.NetCon(source.soma(0.5)._ref_v, syn, sec=source.soma)
        nc.weight[0] = 0.007 
        nc.delay = 0.65 * ms
        netconsFb_Pyr.append(nc)
        synsFb_Pyr.append(syn)
        
# ******* Pyr - FSIN *********

synsPyr_Fb = []
netconsPyr_Fb = []
for idx,cell in enumerate(Fsis):
    
    available_target = [syn for syn, fsi_idx in syn_to_fsi_idx.items() if fsi_idx == idx if syn not in synsFsi]   # Some synapses already formed during fb-fb.
    available_sources = [presyn for presyn in pyrs] 
    num_sources = int(len(available_sources) * 0.18)                # 18 % of pyramidal network
    source_indices = np.random.choice(range(len(available_sources)), size=num_sources, replace=False)
    sources = [available_sources[i] for i in source_indices]

    for source in sources:
        available_target = [syn for syn in available_target if syn not in synsPyr_Fb]    # Avoid creating multiple synapses at same location.  
        syn = SynsFsi[fsi_syn_idx[random.sample(available_target,1)[0]]]
        syn.tau1 = 0.26 * ms
        syn.tau2 = 9 * ms
        syn.e = 0 * mV                                                   # Excitatory Synpase
        nc = h.NetCon(source.soma(0.5)._ref_v, syn, sec=source.soma)
        nc.weight[0] = 0.0001 
        nc.delay = 0.64 * ms
        netconsPyr_Fb.append(nc)
        synsPyr_Fb.append(syn)
        

# ******** FF to Pyr ***********

synsFf_Pyr = []
netconsFf_Pyr = []
for idx,cell in enumerate(pyrs):

    all_synapses = [syn for syn, pyr_idx in syn_to_pyr_idx.items() if pyr_idx == idx]       # Select all syns which map to idx of given "cell".
    dend4and5synapses =   all_synapses[int(len(all_synapses)*0.6):int(len(all_synapses) * 1)]
          
 
    available_sources = [presyn for presyn in Ffs]         
    num_sources = int(len(available_sources) * 0.3)               # 0.3 corresponds to 30% of FF network
    source_indices = np.random.choice(range(len(available_sources)), size=num_sources, replace=False)
    sources = [available_sources[i] for i in source_indices]

    for source in sources:
        available_target = [syn for syn in dend4and5synapses if syn not in synsPyr if syn not in synsFf_Pyr ] # Synapses only from dend4&dend5 excluding dend4 syns during pyr-pyr & avoid overlaping
        syn = SynsPyr[pyr_syn_idx[random.sample(available_target,1)[0]]]
        syn.tau1 = 0.2 * ms
        syn.tau2 = 1.2 * ms
        syn.e = -80 * mV              # Inhibitory connections
        nc = h.NetCon(source.soma(0.5)._ref_v, syn, sec=source.soma)
        nc.weight[0] = 0.007 
        nc.delay = 0.6 * ms  
        netconsFf_Pyr.append(nc)
        synsFf_Pyr.append(syn)




# Restore previous random states 
np.random.set_state(numpy_state)
random.setstate(python_state)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

#******* Stimulate the Network OB-Pyr excitatory:************
 

for idx,Spike_times in enumerate(Spike_times_list):
    odor_name = Odours_name[idx]

    Syn_Ob_pyr=[]
    netconsOb_Pyr=[]
    tVecList_pyr = []
    vecstimList_pyr = []
    for idx,cell in enumerate(pyrs):
         all_synapses = [syn for syn, pyr_idx in syn_to_pyr_idx.items() if pyr_idx == idx]       # Select all syns which map to idx of given "cell".
         dend5_synapses =   all_synapses[int(len(all_synapses)*0.8):int(len(all_synapses) * 1)]
         remaining_target = [syn for syn in dend5_synapses if syn not in synsFf_Pyr ]  # Remaining synapses on dend5 
         number_of_sources_pyr = 23  # Each pyr is connected to 20% of Mitral populatiom & Mi population is 22.5% of PYRs.
         available_sources_pyr = [h.VecStim() for src in range(number_of_sources_pyr)]

         t_indices = np.random.choice(range(len(Spike_times)), size=number_of_sources_pyr, replace=False)
         sources_t = [Spike_times[i] for i in t_indices]
         
         for T, vecstim in zip(sources_t,available_sources_pyr): 
            tVec = h.Vector(T)   
            vecstim.play(tVec)
            tVecList_pyr.append(tVec)
            vecstimList_pyr.append(vecstim)
             
            available_target = [syn for syn in remaining_target if syn not in Syn_Ob_pyr ] # Synapses only from dend4&dend5 excluding dend4 syns during pyr-pyr & avoid overlaping
            syn = SynsPyr[pyr_syn_idx[random.sample(available_target,1)[0]]]
            syn.tau1 = 0.2 * ms
            syn.tau2 = 1.2 * ms
            syn.e = 0 * mV              # Excitatory connections
            nc = h.NetCon(vecstim,syn)
            nc.weight[0] = 0.000037        
            nc.delay = 1.27 * ms  
            netconsOb_Pyr.append(nc)
            Syn_Ob_pyr.append(syn)


         
    #******* Stimulate the Network OB-FF excitatory:************
    
    Syn_Ob_ff=[]
    netconsOb_ff=[]
    tVecList_ff = []
    vecstimList_ff = []
    
    for idx,cell in enumerate(Ffs):
         FF_synapses = [syn for syn, ff_idx in syn_to_ff_idx.items() if ff_idx == idx]       # Select all syns which map to idx of given "cell".
         remaining_target = [syn for syn in FF_synapses if syn not in synsFf] 
         number_of_sources_ff = 46  # Each ff is connected to 40% of mitral cells.
         available_sources_ff = [h.VecStim() for src in range(number_of_sources_ff)]
        
         t_indices = np.random.choice(range(len(Spike_times)), size=number_of_sources_ff, replace=False)
         sources_t = [Spike_times[i] for i in t_indices] 
         
         
         for T, vecstim in zip(sources_t,available_sources_ff):
            tVec = h.Vector(T)   
            vecstim.play(tVec)
            tVecList_ff.append(tVec)
            vecstimList_ff.append(vecstim)
             
            available_target = [syn for syn in remaining_target if syn not in Syn_Ob_ff ] # Synapses only from dend4&dend5 excluding dend4 syns during pyr-pyr & avoid overlaping
            syn = SynsFf[ff_syn_idx[random.sample(available_target,1)[0]]]
            syn.tau1 = 0.2 * ms
            syn.tau2 = 1.2 * ms
            syn.e = 0 * mV              # Excitatory connections
            nc = h.NetCon(vecstim,syn)
            nc.weight[0] = 0.00015       
            nc.delay = 2.42 * ms  
            netconsOb_ff.append(nc)
            Syn_Ob_ff.append(syn)

     
  
        
    
    #########################################################################################################################
    
    
    h.finitialize()
    h.continuerun(6000*ms)
    h.celsius=33
    h.tstop=6000*ms 
        
    for i, cell in enumerate(Fsis):
        plt.vlines(list(cell.spike_times), i + 0.5, i + 1.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Cells")
    plt.title("Fsis Raster")
    plt.show()
    #plt.savefig('F.png', dpi=300, bbox_inches='tight')
    plt.close() 
    
    
    
    #########################################################################################################################
    
    
    base_path = "/home/ankitk23/venv/Control-2"

    
    common_saving_path = os.path.join(base_path, odor_name, f'trial {rank+1}')
    os.makedirs(common_saving_path, exist_ok=True)
    
    
    pyr_spike_data = []
    for i, pyr in enumerate(pyrs):
        neuron_spikes = pd.Series(pyr.spike_times)
        neuron_spikes.name = i+1  # Set the column name to the neuron index
        pyr_spike_data.append(neuron_spikes)
    
    pyr_spike_data = pd.DataFrame(pyr_spike_data)
    pyr_spike_data = pyr_spike_data.reset_index().rename(columns={'index': 'Cells'})
    pyr_file_name = f'pyr{rank+1}.csv'
    pyr_file_path = os.path.join(common_saving_path,pyr_file_name)
    pyr_spike_data.to_csv(pyr_file_path, index=False)
    
    
   

comm.Barrier()





    




