import random
import csv
import re
import math
import sys
import os 
import statistics as stat
import time as t

# initialize chromosomes with cardinality constraints
def cardinalInit(args):
    new_ind = []
    inv = list(range(args.start_id, args.start_id + args.end_id)) 
    # apply "intra-layer" stacking on all partitions
    if (random.random() < args.gold):
        new_ind = simpleStack(inv, args)
    # shuffle the partitions around with probability: 1 - gold
    else:
        new_ind = simpleStack(random.sample(inv, args.end_id), args)
    return new_ind

# stack adjacent memory partitions up to max_stack height
def simpleStack(inventory, args):
    stacks = []
    cur_stack = []
    h = 0
    cur_layer = args.lookupT[inventory[0]][0]
    while len(list(inventory)) > 0:
        new_layer = args.lookupT[inventory[0]][0]
        if (h < args.max_stack_height) and ((cur_layer == new_layer) or args.enableInter):
            cur_stack.append(inventory[0])
            inventory.remove(inventory[0])
            h += 1
        else:
            stacks.append(cur_stack)
            cur_stack = []
            h = 0
            cur_layer = args.lookupT[inventory[0]][0]
    if len(cur_stack) > 0:
        stacks.append(cur_stack)
    return stacks

# first fit heuristic with dynamic bin capacity
def firstFitDynamic(inventory, args):
    stacks = []
    cur_stack = []
    h = 0
    w = 0
    overshoot = 0
    new_overshoot = 0
    pow2_overshoot = 0
    new_pow2_overshoot = 0
    stack_h = 0
    cur_layer = args.lookupT[inventory[0]][0]
    while len(inventory) > 0:
        for item in list(inventory):
            new_layer = args.lookupT[item][0]
            
            if stack_h == 0:
                w = args.lookupT[item][1] * args.lookupT[item][4]
            # new bin height if the item were to be packed in the same stack
            new_h = h + args.lookupT[item][3]
            # adjust the height of the memory resource based on the BRAM "aspect" mode
            (dyn_mem_height, k) = aspectRatio(w, args)
            # re-adjust the height for the case that the current item will join the stack
            (dyn_mem_height_new, k) = aspectRatio(max(w, args.lookupT[item][1] * args.lookupT[item][4]), args)
            # compute how well the current stack fits in the memory resource
            if args.aloc_pow2:
                if h > 0:
                    pow2_overshoot = max(abs(h - 2**math.ceil(math.log(h) / math.log(2))), abs((h % dyn_mem_height) - dyn_mem_height))
                else:
                    pow2_overshoot = dyn_mem_height

            overshoot = abs((h % dyn_mem_height) - dyn_mem_height)
            if h == dyn_mem_height:
                overshoot = 0
            # and how well the stack fits after adding the item
            if args.aloc_pow2:
                if new_h > 0:
                    new_pow2_overshoot = max(abs(new_h - 2**math.ceil(math.log(new_h) / math.log(2))), abs((new_h % dyn_mem_height_new) - dyn_mem_height_new))
                else:
                    new_pow2_overshoot = dyn_mem_height
            new_overshoot = abs((new_h % dyn_mem_height_new) - dyn_mem_height_new)
            if new_h == dyn_mem_height_new:
                new_overshoot = 0
            # only add the item if it is the first in the bin or makes the stack fit better in the memory resource
            if (stack_h == 0) or (((stack_h < args.max_stack_height) 
            and (cur_layer == new_layer or args.enableInter)) 
            and (w == (args.lookupT[item][1] * args.lookupT[item][4]) or (random.random() <= args.p_adm))
            and (((random.random() <= args.mut_gen) or (new_overshoot < overshoot)) or (args.aloc_pow2 and (new_pow2_overshoot < pow2_overshoot) and overshoot == new_overshoot))):
                cur_stack.append(item)
                inventory.remove(item)
                stack_h += 1
                h += args.lookupT[item][3]
                w = max(w, args.lookupT[item][1] * args.lookupT[item][4])
            # finalize the current stack and place the item in a new bin otherwise
            else:
                stacks.append(cur_stack)
                cur_stack = []
                cur_stack.append(item)
                inventory.remove(item)
                stack_h = 1
                h = args.lookupT[item][3]
                w = args.lookupT[item][1] * args.lookupT[item][4]
                cur_layer = args.lookupT[item][0]
    # finalize any leftover stacks
    if len(cur_stack) > 0:
        stacks.append(cur_stack)
    return stacks

# adapt memory resource height based on the BRAM mode
def aspectRatio(partition_w, args):
    dyn_height = 0
    dyn_width = 0
    mem_height = args.mem_height
    mem_width = args.mem_width
    if args.primitive == 'bram':
        if (partition_w <= 1): # case "narrow" 1-bit x 16384 aspect ratio cost 
            dyn_height = 16 * mem_height
            dyn_width = mem_width / 16
        elif (partition_w <= 2): # case "narrow" 2-bit x 8192 aspect ratio cost
            dyn_height = 8 * mem_height
            dyn_width = mem_width / 8
        elif (partition_w <= 4): # case "narrow" 4-bit x 4096 aspect ratio cost
            dyn_height = 4 * mem_height
            dyn_width = mem_width / 4
        elif (partition_w <= 8): # case "narrow" 8-bit x 2048 aspect ratio cost
            dyn_height = 2 * mem_height
            dyn_width = mem_width / 2
        else: # regular BRAM cost (i.e. 16-bit x 1024)
            dyn_height = mem_height
            dyn_width = mem_width        
    else:
        dyn_height = mem_height
        dyn_width = mem_width

    return dyn_height, dyn_width

def mapEff(individual, args):
    imap = []
    for bin in individual:
        h = 0 # bin height
        w = 0 # bin width
        mem_cap = 0 # actual data stored in the bin [bits]
        for part in bin:
            item_height = args.lookupT[part][3]
            item_width = args.lookupT[part][1] * args.lookupT[part][4]
            mem_cap += item_width * item_height
            h += item_height
            w = max(item_width, w)
        dyn_mem_height, dyn_mem_width = aspectRatio(w, args)
        if args.aloc_pow2:
            bram_count = math.ceil(2**math.ceil(math.log(h)/math.log(2)) / dyn_mem_height) * math.ceil(w / dyn_mem_width)
        else:
            bram_count = math.ceil(h / dyn_mem_height) * math.ceil(w / dyn_mem_width)
        map_efficiency = mem_cap / (bram_count * args.mem_width * args.mem_height)
        imap.append((bin, map_efficiency))

    return imap

def getKey(item):
    return item[1]

def mutateValid(args, individual):
    # new_ind = individual
    inventory = []
    repacked = []
    # new_ind2 = random.sample(new_ind, len(new_ind))
    new_ind = sorted(mapEff(individual, args), key=getKey)
    new_ind2 = []
    ctr = 0
    for item in new_ind:
        if item[1] < 1.0:
            ctr += 1
        new_ind2.append(item[0])
    # rng = min(0.2*len(new_ind2), ctr)
    for reps in range(0, ctr): # TODO: make random and adapt rng to range(1, len(individual))
        # target = random.randint(0, ctr)
        for gene in list(new_ind2[0]):
            inventory.append(gene)
        new_ind2.remove(new_ind2[0])
    if random.random() <= 0.5:
        repacked = firstFitDynamic(random.sample(inventory, len(inventory)), args)
    else:
        repacked = firstFitDynamic(inventory, args)
    for new_bin in repacked:
        new_ind2.append(new_bin)
    return new_ind2

def perturbPack(args, solution):
    perturbed = []
    for col in solution:
        perturbed.append(col.copy())

    trgt = int(len(perturbed)/10)
    for moves in range(1, 4):
        ind1 = random.randint(0, len(perturbed) - 1)
        ind2 = random.randint(0, len(perturbed) - 1)
        
        while (ind1 == ind2):
            ind1 = random.randint(0, len(perturbed) - 1)
            ind2 = random.randint(0, len(perturbed) - 1)
        
        rnd = random.randint(0, len(perturbed[ind2]) - 1)
        if len(perturbed[ind1]) >= args.max_stack_height:
            perturbed.append(list([perturbed[ind2][rnd]]))
        else:
            perturbed[ind1].append(perturbed[ind2][rnd])

        perturbed[ind2].remove(perturbed[ind2][rnd])
        if len(list(perturbed[ind2])) < 1:
            perturbed.remove(perturbed[ind2])
    return perturbed

def cxOver(args, solution, solution2):
    perturbed = []
    for col in solution:
        perturbed.append(col.copy())

    ind1 = random.randint(int(len(solution2)/2), int(len(solution2))-1)
    ind2 = random.randint(int(len(solution2)/2), int(len(solution2))-1)
    
    while (ind1 == ind2):
        ind1 = random.randint(0, len(perturbed) - 1)
        ind2 = random.randint(0, len(perturbed) - 1)
    
    lo = min(ind1, ind2)
    hi = max(ind1, ind2)

    snippet = solution2[lo:hi].copy()
    inventory_raw = []
    for candidate in snippet:
        for item in candidate:
            ctr = 0
            for bin in list(perturbed):
                if item in list(bin):
                    cbin = bin.copy()
                    cbin.remove(item)
                    if len(list(cbin)) > 0:
                        inventory_raw.append(cbin)

                    perturbed.pop(ctr)
                    # print(perturbed)
                    break
                ctr += 1
        perturbed.append(candidate)

    inventory = []
    if len(inventory_raw) > 0:
        for lst in inventory_raw:
            for item in lst:
                inventory.append(item)
        repacked = firstFitDynamic(random.sample(inventory, len(inventory)), args)
        for bin in repacked:
            perturbed.append(bin)

    # print("org: " + str(mapEff(solution, args)) + "\n")
    # print("cxd: " + str(mapEff(perturbed, args)) + "\n")

    return perturbed

# Fitness evaluation function (decreases BRAM cost and balances throughput for now)
def getFitness(args, individual):
    bin_metrics = []
    mem_height = args.mem_height
    mem_width = args.mem_width
    for bin in individual:
        hi_stack = 0
        h = 0 # bin height
        w = 0 # bin width
        mem_cap = 0 # actual data stored in the bin [bits]
        bram_count = 0 # RAMB18 modules spanned by the bin
        stack_height = 0 # amount of depth-wise stacked partitions within the bin
        clayers = []
        for part in bin:
            # Keep track of the capacity stored in the bin
            clayers.append(args.lookupT[part][0])
            item_height = args.lookupT[part][3]
            item_width = args.lookupT[part][1] * args.lookupT[part][4]
            mem_cap += item_width * item_height
            # and update the dimensions of the bin
            h += item_height
            w = max(item_width, w)
            stack_height += 1
        # Compute the BRAM span of the bin
        dyn_mem_height, dyn_mem_width = aspectRatio(w, args)
        if args.aloc_pow2:
            bram_count = math.ceil(2**math.ceil(math.log(h)/math.log(2)) / dyn_mem_height) * math.ceil(w / dyn_mem_width)
        else:
            bram_count = math.ceil(h / dyn_mem_height) * math.ceil(w / dyn_mem_width)
        if stack_height == 1:
            if h <= int(mem_height / 2):
                bram_count = math.ceil((2 * h) / dyn_mem_height) * math.ceil(w / (2 * mem_width))
        if stack_height > 2:
            hi_stack = 1
        layer_count = len(set(clayers))
        map_efficiency = mem_cap / (bram_count * mem_width * mem_height)
        bin_metrics.append((bin, w, h, bram_count, map_efficiency, stack_height, layer_count, hi_stack))

    if args.debug > 1:    
        print("Bin Packing: " + str(individual) + " | " + str(bin_metrics) + "\n")

    # Fitness values to optimize for (total BRAM cost, max stack height)
    bram = sum([metric[3] for metric in bin_metrics])
    stack = sum([metric[5] for metric in bin_metrics]) / len(individual)
    comp = sum([metric[6] for metric in bin_metrics])
    latency = float(math.ceil(max([metric[5] for metric in bin_metrics]) / 2))
    lut_complexity = sum([metric[7] for metric in bin_metrics])
    # waste = sum([((1-metric[4])*metric[3])**2 for metric in bin_metrics])

    return bram + comp/1000 + 10*(stack / args.max_stack_height), bram, lut_complexity
import random
import csv
import re
import math
import sys
import os
import statistics as stat
import time as t
import numpy as np
import random
import pickle
import bitstring
from collections import Counter
from operator import itemgetter

from deap import base
from deap import creator
from deap import tools

import argparse

parser = argparse.ArgumentParser(description='Memory Packing using genetic algorithms')
parser.add_argument('--network', type=str, default=None, help='Name of neural network to parse', required=True)
parser.add_argument('--enableInter', action='store_true', default=False, help='Do inter-layer packing')
parser.add_argument('--aloc_pow2', action='store_true', default=False, help='Constrain stacks to power-of-2 depths')
parser.add_argument('--thresh', type=int, default=129, help='Minimum depth at which memory is mapped to BRAM')
parser.add_argument('--thresh_max', type=int, default=2**20, help='Maximum depth at which memory is mapped to BRAM')
parser.add_argument('--max_num_bins', type=int, default=100, help='Maximum number of bins in the non-optimized GA')
parser.add_argument('--max_stack_height', type=int, default=4, help='Maximum height of a stack')
parser.add_argument('--debug', type=int, default=0, help='Debug verbosity level')
parser.add_argument('--output', type=str, default=None, help='Name of output file', required=False)
parser.add_argument('--primitive', type=str, default='bram', help='specify memory resource primitive', required=False)
parser.add_argument('--rtime', type=float, default=0, help='specify the maximum runtime for optimization', required=False)
parser.add_argument('--firstfit', action='store_true', default=False, help='enable first-fit dynamic heuristic')

# Hardware specfic
parser.add_argument('--mem_width', type=int, default=16, help='Width of a memory block')
parser.add_argument('--mem_height', type=int, default=1024, help='Height of a memory block')

# GA parameters
parser.add_argument('--t_con', type=int, default=100, help='proportionality constant for the latency (fitness function)')
parser.add_argument('--mut', type=float, default=0.3, help='mutation probability')
parser.add_argument('--mut_gen', type=float, default=0.001, help='probability of mutating a gene in chromosome')
parser.add_argument('--p_adm', type=float, default=0.1, help='probability of admitting memories with mismatching widths')
parser.add_argument('--cross', type=float, default=0.0, help='crossover probability')
parser.add_argument('--gold', type=float, default=0.125, help='probability of creating a "golden" unit')
parser.add_argument('--top', type=int, default=5, help='top-x fittest selection size')
parser.add_argument('--pop_count', type=int, default=600, help='Size of the population per iteration in the GA')
parser.add_argument('--generations', type=int, default=1000, help='Generation limit after which the GA is stopped')

# Select GA optimizer
parser.add_argument('--opt', action='store_true', default=False, help='Use the optimized GA')

# PRNG seed
parser.add_argument('--seed', type=int, default=42, help='PRNG seed')

# Flag a dry run (no output products generation)
parser.add_argument('--dryrun', action='store_true', default=False, help='Run packing without output product generation')

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

## input parameters
#file = "data/" + args.network + "-memories.csv"

network_spec = pickle.load(open(args.network, 'rb'))

#TODO: move these definitions somewhere else
bitwidth = {}
bitwidth['BINARY'] = 1
bitwidth['UINT2'] = 2
bitwidth['UINT3'] = 3
bitwidth['UINT4'] = 4
bitwidth['UINT8'] = 8
bitwidth['UINT16'] = 16
bitwidth['UINT32'] = 32
bitwidth['BIPOLAR'] = 1
bitwidth['INT2'] = 2
bitwidth['INT3'] = 3
bitwidth['INT4'] = 4
bitwidth['INT8'] = 8
bitwidth['INT16'] = 16
bitwidth['INT32'] = 32
bitwidth['FLOAT32'] = 32

opt = (-1.0, -1.0, -1.0) # optimization weights
args.start_id = 0
line_count = 0
layers = 0
partitions = 0
args.lookupT = []
mapped_layers = []
fittest = []
resStr = "Network spec: " + str(args.network) + "\nThreshold: " + str(args.thresh) + "\nMax Stack Height: " + str(args.max_stack_height) + "\n\n"
benchlog = 'RAMB18,Stacks,Time,Seed,Algo\n'
algo = 'GA'

# network parser

for layer in network_spec:
    label = layer
    simd = network_spec[layer]['Attributes']['SIMD']
    pe = network_spec[layer]['Attributes']['PE']
    wmem = network_spec[layer]['Attributes']['WMEM']
    wprec = bitwidth[network_spec[layer]['Attributes']['Precision']]
    # Only record layers that map reasonably efficiently to BRAM 
    if wmem >= args.thresh and wmem <= args.thresh_max:
        mapped_layers.append(layers)
        for partition in range (0, pe):
            # Look-up table to search for memory partition dimensions
            args.lookupT.append((layers, simd, pe, wmem, wprec, label))
            partitions += 1
    line_count += 1
    layers += 1

print(f'Processed {line_count} lines, and found {layers} layers of which {len(mapped_layers)} are mapped to BRAM, spanning a total of {partitions} partitions')
print(f'Mapped layers: {sorted(set([l[5] for l in args.lookupT]))}\n')
resStr += "Mapped layers: " + str(sorted(set([l[5] for l in args.lookupT]))) + "\n\n"

if len(mapped_layers) == 0:
    print("No suitable layers to process. Try reducing your threshold")
    sys.exit()

args.end_id = partitions

if args.opt:
    from ga_packing_opt import cardinalInit, getFitness, mutateValid, perturbPack, cxOver
else:
    from ga_packing import cardinalInit, getFitness, mutateValid

# Create individual and fitness parameter
creator.create("MultiModeFit", base.Fitness, weights=opt)
creator.create("Individual", list, fitness=creator.MultiModeFit)

# Register the population related tools to the toolbox
toolbox = base.Toolbox()
toolbox.register("initGenes", cardinalInit, args)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.initGenes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Register the Genetic Operators to the toolbox
toolbox.register("evaluate", getFitness, args)
toolbox.register("mate", cxOver, args)
if args.firstfit:
    toolbox.register("mutate", mutateValid, args)
else:
    toolbox.register("mutate", perturbPack, args)
    algo = 'GAnoff'
toolbox.register("select", tools.selTournament, tournsize=args.top)

def optimize():
    global resStr
    global benchlog

    start = t.time()
    pop = toolbox.population(n=args.pop_count)

     # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, (bram_c, stack_h, lut_c) in zip(pop, fitnesses):
        ind.fitness.values = (bram_c, stack_h, lut_c)

    # Crossover & Mutation probabilities
    CXPB, MUTPB = args.cross, args.mut
    # Fitness value that will be used as stopping criterium
    bram_count = [ind.fitness.values[0] for ind in pop]
    max_stack_height = [ind.fitness.values[1] for ind in pop]

    g = 0
    
    while g < args.generations or (t.time() - start) < args.rtime:
        # Generate the new generation
        if args.max_stack_height == 1:
            g = g + args.generations
        else:
            g = g + 1

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Perform crossover
        ctr1 = 0
        ctr2 = 1 
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                offspring[ctr1][:] = toolbox.mate(child1, child2)
                offspring[ctr2][:] = toolbox.mate(child2, child1)
                del child1.fitness.values
                del child2.fitness.values
            ctr1 += 2
            ctr2 += 2

        # Perform mutation
        ctr = 0
        for mutant in offspring:
            if random.random() < MUTPB:
                offspring[ctr][:] = toolbox.mutate(mutant)
                del mutant.fitness.values
            ctr += 1
        
        # Update the fitness values of the mutants and offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, (fit0, fit1, fit2) in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit0, fit1, fit2)
        
        # Repeat the cycle of life
        pop[:] = offspring

        qor = [(int(ind.fitness.values[1]), int(ind.fitness.values[2])) for ind in pop]
        qor.sort(key=itemgetter(0))
        # bram_count = [ind.fitness.values[1] for ind in pop]
        min_bram_count = (qor[0])[0]
        hi_stacks = (qor[0])[1]
        max_stack_height = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(bram_count) / length
        sum2 = sum(x*x for x in bram_count)
        std = abs(sum2 / length - mean**2)**0.5

        # Track the chromosome of the fittest individual (in terms of BRAM count)
        fittest = [indiv for indiv in pop if indiv.fitness.values[0] == min(max_stack_height)][0]

        resStr += "Min BRAM count (gen " + str(g) + "): " + str(min_bram_count) + " " + str(hi_stacks) + " hi stacks @ " + str(t.time() - start) + "\n"
        benchlog += str(int(min_bram_count)) + "," + str(hi_stacks) + "," + str(t.time() - start) + "," + str(args.seed) + "," + algo + "\n" 
        
        if args.debug > 0:
            print("-- Generation %i --" % g)
            # print(" Min BRAM count: %s" % min(bram_count))
            # print(" Max BRAM count: %s" % max(bram_count))
            print(" Avg BRAM count: %s" % mean)
            print(" Std %s\n" % std)

            print(" Max stack height of the fittest: %s\n\n" % min((max_stack_height)*partitions))
            print(" fittest: %s\n\n" % min(max_stack_height))

            # print(" Fittest individual: %s\n" % fittest)

    end = t.time()

    resStr += "\n\tRuntime: " + str(int(end-start)) + " seconds"
    print("\n\tRuntime: " + str(end-start))

    return fittest

def report(fittest):
    global resStr
    global benchlog
    
    resStr += "------\n\nBest packing solution: \n\n"
    print(" Best packing solution: ")

    count = 0
    
    result = {}
    
    if not args.opt:
        bins = set(fittest)
        for cbin in bins:
            stack = []
            result['bin'+str(count)] = []
            for gene in range(0, len(fittest)):
                if fittest[gene] == cbin:
                    stack.append((("<" + str(args.lookupT[gene + args.start_id][1]) + ", " + str(args.lookupT[gene + args.start_id][3]) + ">") , args.lookupT[gene + args.start_id][5]))
                    result['bin'+str(count)].append(args.lookupT[gene + args.start_id][5])
            # sol.append(stack)
            resStr += "\tbin[" + str(count) + "]: {" + str(stack) + "}\n"
            print(" \tbin[" + str(count) + "]:  {%s}" % stack)
            count += 1
    else:
        for cbin in fittest:
            stack = []
            for gene in cbin:
                stack.append((("<" + str(args.lookupT[gene][1] * args.lookupT[gene][4]) + ", " + str(args.lookupT[gene][3]) + ">") , args.lookupT[gene][5]))
            
            resStr += "\tbin[" + str(count) + "]: {" + str(stack) + "}\n"
            print(" \tbin[" + str(count) + "]:  {%s}" % stack)
            result['bin'+str(count)] = [args.lookupT[i + args.start_id][5] for i in cbin]
            count += 1

    if args.output != None:
        resFile = open(args.output + ".txt", "w+")
        resFile.write(resStr)
        if args.dryrun:
            if args.firstfit:
                benchFile = open("ga.csv", "w+")
                benchFile.write(benchlog)
            else:
                benchFile = open("ganoff.csv", "w+")
                benchFile.write(benchlog)
    return result

def weights2hex(data, dtype, owidth):
    #data is a numpy array of dimensions [WMEM][SIMD] and dtype is a string
    hex = []
    for line in data:
        lineval = bitstring.BitArray(length=0)
        for val in line:
            #do casting and value conversion for the few cases where that's required (signed ints, bipolar, float) to uint32
            if dtype == 'BIPOLAR' and val == -1:
                val = int(0)
            elif dtype == 'FLOAT32':
                val = bitstring.BitArray(float=val,length=bitwidth[dtype]).uint
            elif dtype == 'INT2' or dtype == 'INT3' or dtype == 'INT4' or dtype == 'INT8' or dtype == 'INT16' or dtype == 'INT32':
                val = bitstring.BitArray(int=int(val),length=bitwidth[dtype]).uint
            else:
                val = int(val)
            #pack the data into the line value
            lineval.append(bitstring.BitArray(uint=val,length=bitwidth[dtype]))
        #extend to the desired output width (a minimum of 4 bits)
        lineval.prepend(bitstring.BitArray(length=(max(4,owidth)-lineval.len)))
        #represent as hex
        hex.append(lineval.hex)
        
    return hex

def gen_artefacts(solution, net):
    # generate:
    #    - readmemh files, full height and segmented in 1k deep chunks
    #    - header for HLS streamer
    #keep track of PEs for each layer in a dictionary with same keys as net
    pe_index = dict.fromkeys(net.keys(),0)
    #keep track of source bins, and offset in bins, for each PE in each layer (to generate stream combiners)
    combiner_spec = {key: [] for key in net.keys()}
    bin_index = 0
    hex_data = []
    #define some essential stuff in the BD assembly script
    with open('bd_assembly.tcl', 'a+') as f:
        f.write("if {[info exists computefreqmhz]} { puts \"Using existing compute clock frequency setting of $computefreqmhz MHz\" } else { set computefreqmhz 100 }\n")
        f.write("if {[info exists memfreqmhz]} { puts \"Using existing memory clock frequency setting of $memfreqmhz MHz\" } else { set memfreqmhz 100 }\n")
        f.write("create_bd_port -dir I -type clk compute_clk\n")
        f.write("set_property CONFIG.FREQ_HZ [expr {$computefreqmhz*1000000}] [get_bd_ports compute_clk]\n")
        f.write("create_bd_port -dir I -type clk memory_clk\n")
        f.write("set_property CONFIG.FREQ_HZ [expr {$memfreqmhz*1000000}] [get_bd_ports memory_clk]\n")
        f.write("create_bd_port -dir I -type rst memory_aresetn\n")
        f.write("create_bd_port -dir I -type rst compute_aresetn\n")

    #parse bins
    for bin in solution:
        #parse bin and: identify width (max of all widths), identify depth (sum of all depths), 
        bin_width = 0
        bin_depth = 0
        bin_data = []
        #parse bin and identify width (max of all widths) and depth (sum of all depths)
        for layer in solution[bin]:
            layer_width = net[layer]['Attributes']['SIMD'] * bitwidth[net[layer]['Attributes']['Precision']]
            bin_width = max(bin_width, layer_width)
            bin_depth += net[layer]['Attributes']['WMEM']
        #parse again, to get data
        layer_index = 0
        for layer in solution[bin]:
            #pack PE data into hex list
            bin_data += weights2hex(net[layer]['Data'][pe_index[layer]], net[layer]['Attributes']['Precision'], bin_width)
            pe_index[layer] += 1
            combiner_spec[layer].append((bin_index,layer_index))
            layer_index += 1

        #create dedicated folder for bin artefacts
        directory = "./bin"+str(bin_index)
        if not os.path.exists(directory):
            os.makedirs(directory)

        #write these to files, one per bin
        with open(directory+'/memory.dat', 'w') as f:
            for item in bin_data:
                f.write("%s\n" % item)
    
        #write data to files, in 1k blocks
        block_idx = 0
        for line_idx in range(len(bin_data)):
            if (line_idx % 1024) == 0:
                f = open(directory+'/memblock_'+str(block_idx)+'.dat', 'w')
            f.write("%s\n" % bin_data[line_idx])
            if (line_idx % 1024) == 1023:
                f.close()
                block_idx += 1
                
        #generate HLS headers
        with open(directory+'/streamer.h', 'w') as f:
            #write defines
            #instantiate streamers
            f.write("#define NSTREAMS "+str(len(solution[bin]))+"\n")
            f.write("#define MEM_DEPTH "+str(bin_depth)+"\n")
            f.write("#define MEM_WIDTH "+str(bin_width)+"\n")
            offset = 0
            for layer_idx in range(len(solution[bin])):
                    layer_width = net[solution[bin][layer_idx]]['Attributes']['SIMD'] * bitwidth[net[solution[bin][layer_idx]]['Attributes']['Precision']]
                    layer_depth = net[solution[bin][layer_idx]]['Attributes']['WMEM']
                    f.write("#define STRM"+str(layer_idx)+"_WIDTH "+str(layer_width)+"\n")
                    f.write("#define STRM"+str(layer_idx)+"_DEPTH "+str(layer_depth)+"\n")
                    f.write("#define STRM"+str(layer_idx)+"_OFFSET "+str(offset)+"\n")
                    offset += layer_depth
            #write memory contents
            f.write('\nconst ap_uint<MEM_WIDTH> weights[MEM_DEPTH] = {\n')
            for item in bin_data:
                item = '"0x'+item+'",'
                f.write("%s\n" % item)
            f.write('};')
            
        hex_data.append(bin_data)
        bin_index += 1
        #hex_data now contains hex representations for each bin, in the correct width
        
        #generate BD assembly script - instantiate and configure streamer corresponding to the bin
        with open('bd_assembly.tcl', 'a+') as f:
            #instantiate streamers
            f.write("create_bd_cell -type ip -vlnv xilinx.com:user:memstream:1.0 "+bin+"\n")
            f.write("set_property -dict [list CONFIG.NSTREAMS "+str(len(solution[bin]))+"] [get_bd_cells "+bin+"]\n")
            f.write("set_property -dict [list CONFIG.MEM_DEPTH "+str(bin_depth)+"] [get_bd_cells "+bin+"]\n")
            f.write("set_property -dict [list CONFIG.MEM_WIDTH "+str(bin_width)+"] [get_bd_cells "+bin+"]\n")
            f.write("set_property -dict [list CONFIG.MEM_INIT " + os.getcwd() + "/" + bin+"/] [get_bd_cells "+bin+"]\n")
            if len(solution[bin]) > 2:
                f.write("connect_bd_net [get_bd_ports memory_clk] [get_bd_pins "+bin+"/aclk]\n")
                f.write("connect_bd_net [get_bd_ports memory_aresetn] [get_bd_pins "+bin+"/aresetn]\n")
            else:
                f.write("connect_bd_net [get_bd_ports compute_clk] [get_bd_pins "+bin+"/aclk]\n")
                f.write("connect_bd_net [get_bd_ports compute_aresetn] [get_bd_pins "+bin+"/aresetn]\n")
            offset = 0
            for layer_idx in range(len(solution[bin])):
                    layer_width = net[solution[bin][layer_idx]]['Attributes']['SIMD'] * bitwidth[net[solution[bin][layer_idx]]['Attributes']['Precision']]
                    layer_depth = net[solution[bin][layer_idx]]['Attributes']['WMEM']
                    f.write("set_property -dict [list CONFIG.STRM"+str(layer_idx)+"_WIDTH "+str(layer_width)+"] [get_bd_cells "+bin+"]\n")
                    f.write("set_property -dict [list CONFIG.STRM"+str(layer_idx)+"_DEPTH "+str(layer_depth)+"] [get_bd_cells "+bin+"]\n")
                    f.write("set_property -dict [list CONFIG.STRM"+str(layer_idx)+"_OFFSET "+str(offset)+"] [get_bd_cells "+bin+"]\n")
                    offset += layer_depth
        
    #generate BD assembly script - instantiate stream combiners and connect everything
    print(combiner_spec)
    with open('bd_assembly.tcl', 'a') as f:
        #instantiate and configure combiners
        for layer in combiner_spec:
            #if the spec is empty then the layer was discarded from packing, just ignore
            if len(combiner_spec[layer]) == 0:
                continue
            #if there is no combiner needed, make corresponding stream external and continue to next layer
            if len(combiner_spec[layer]) == 1:
                f.write("create_bd_intf_port -mode master -vlnv xilinx.com:interface:axis_rtl:1.0 "+layer+"\n")
                #iff the bin has depth>2 add CDC logic
                if len(solution['bin'+str(combiner_spec[layer][0][0])]) > 2:
                    #insert cdc and make external, then continue with next layer
                    cdc_name = "cdc_bin"+str(combiner_spec[layer][0][0])+"_"+str(combiner_spec[layer][0][1])
                    f.write("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_clock_converter:1.1 "+cdc_name+"\n")
                    f.write("connect_bd_intf_net [get_bd_intf_pins bin"+str(combiner_spec[layer][0][0])+"/m_axis_"+str(combiner_spec[layer][0][1])+"] [get_bd_intf_pins "+cdc_name+"/S_AXIS]\n")
                    f.write("connect_bd_net [get_bd_ports memory_clk] [get_bd_pins "+cdc_name+"/s_axis_aclk]\n")
                    f.write("connect_bd_net [get_bd_ports compute_clk] [get_bd_pins "+cdc_name+"/m_axis_aclk]\n")
                    f.write("connect_bd_net [get_bd_ports memory_aresetn] [get_bd_pins "+cdc_name+"/s_axis_aresetn]\n")
                    f.write("connect_bd_net [get_bd_ports compute_aresetn] [get_bd_pins "+cdc_name+"/m_axis_aresetn]\n")
                    f.write("connect_bd_intf_net [get_bd_intf_ports "+layer+"]  [get_bd_intf_pins "+cdc_name+"/M_AXIS]\n")
                else:
                    f.write("connect_bd_intf_net [get_bd_intf_ports "+layer+"] [get_bd_intf_pins bin"+str(combiner_spec[layer][0][0])+"/m_axis_"+str(combiner_spec[layer][0][1])+"]\n")
                continue
            #instantiate (possibly multiple) axi combiner(s)
            #each combiner has up to 16 inputs
            ncombiners = math.ceil(len(combiner_spec[layer])/16)
            ninputs = 16 if ncombiners > 1 else len(combiner_spec[layer])
            for i in range(ncombiners):
                f.write("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_combiner:1.1 "+layer+"_"+str(i)+"\n")
                f.write("set_property -dict [list CONFIG.NUM_SI "+str(ninputs)+"] [get_bd_cells "+layer+"_"+str(i)+"]\n")
                f.write("connect_bd_net [get_bd_ports compute_clk] [get_bd_pins "+layer+"_"+str(i)+"/aclk]\n")
                f.write("connect_bd_net [get_bd_ports compute_aresetn] [get_bd_pins "+layer+"_"+str(i)+"/aresetn]\n")
            #if multiple combiners, instantiate a combiner combiner (this takes us up to 256 streams for a total maximum output size of 512 bytes)
            if ncombiners > 1:
                f.write("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_combiner:1.1 "+layer+"_out\n")
                f.write("set_property -dict [list CONFIG.NUM_SI "+str(ncombiners)+"] [get_bd_cells "+layer+"_out]\n")
                f.write("connect_bd_net [get_bd_ports compute_clk] [get_bd_pins "+layer+"_out/aclk]\n")
                f.write("connect_bd_net [get_bd_ports compute_aresetn] [get_bd_pins "+layer+"_out/aresetn]\n")
                f.write("create_bd_intf_port -mode master -vlnv xilinx.com:interface:axis_rtl:1.0 "+layer+"\n")
                f.write("connect_bd_intf_net [get_bd_intf_ports "+layer+"] [get_bd_intf_pins "+layer+"_out/M_AXIS]\n")
                for i in range(ncombiners):
                    f.write("connect_bd_intf_net [get_bd_intf_pins "+layer+"_"+str(i)+"/M_AXIS] [get_bd_intf_pins "+layer+"_out/S"+("%02d"%i)+"_AXIS]\n")
            else:
                f.write("create_bd_intf_port -mode master -vlnv xilinx.com:interface:axis_rtl:1.0 "+layer+"\n")
                f.write("connect_bd_intf_net [get_bd_intf_ports "+layer+"] [get_bd_intf_pins "+layer+"_0/M_AXIS]\n")
            #connect everything through FIFOs
            for connection_idx in range(len(combiner_spec[layer])):
                #iff the bin has depth>2 add CDC logic
                if len(solution['bin'+str(combiner_spec[layer][connection_idx][0])]) > 2:
                    cdc_name = "cdc_bin"+str(combiner_spec[layer][connection_idx][0])+"_"+str(combiner_spec[layer][connection_idx][1])
                    f.write("create_bd_cell -type ip -vlnv xilinx.com:ip:axis_clock_converter:1.1 "+cdc_name+"\n")
                    f.write("connect_bd_intf_net [get_bd_intf_pins bin"+str(combiner_spec[layer][connection_idx][0])+"/m_axis_"+str(combiner_spec[layer][connection_idx][1])+"] [get_bd_intf_pins "+cdc_name+"/S_AXIS]\n")
                    f.write("connect_bd_net [get_bd_ports memory_clk] [get_bd_pins "+cdc_name+"/s_axis_aclk]\n")
                    f.write("connect_bd_net [get_bd_ports compute_clk] [get_bd_pins "+cdc_name+"/m_axis_aclk]\n")
                    f.write("connect_bd_net [get_bd_ports memory_aresetn] [get_bd_pins "+cdc_name+"/s_axis_aresetn]\n")
                    f.write("connect_bd_net [get_bd_ports compute_aresetn] [get_bd_pins "+cdc_name+"/m_axis_aresetn]\n")
                    f.write("connect_bd_intf_net [get_bd_intf_pins "+cdc_name+"/M_AXIS] [get_bd_intf_pins "+layer+"_"+str(int(connection_idx/ninputs))+"/S"+("%02d"%int(connection_idx%ninputs))+"_AXIS]\n")
                else:
                    f.write("connect_bd_intf_net [get_bd_intf_pins bin"+str(combiner_spec[layer][connection_idx][0])+"/m_axis_"+str(combiner_spec[layer][connection_idx][1])+"] [get_bd_intf_pins "+layer+"_"+str(int(connection_idx/ninputs))+"/S"+("%02d"%int(connection_idx%ninputs))+"_AXIS]\n")
        #export the BD as an IP
        f.write("ipx::package_project -root_dir [get_property DIRECTORY [current_project]]/[current_bd_design] -vendor xilinx.com -library user -taxonomy /UserIP -module [current_bd_design] -import_files\n")

if __name__ == '__main__':
    if args.dryrun:
        report(optimize())
    else:
        gen_artefacts(report(optimize()), network_spec)
