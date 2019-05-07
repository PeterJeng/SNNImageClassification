import os
import numpy as np
from PIL import Image
from brian2 import *
from matplotlib import pyplot


def brian_example():
    taum = 10 * ms
    taupre = 20 * ms
    taupost = taupre
    Ee = 0 * mV
    vt = -54 * mV
    vr = -60 * mV
    El = -74 * mV
    taue = 5 * ms
    F = 15 * Hz
    gmax = .01
    dApre = .01
    dApost = -dApre * taupre / taupost * 1.05
    dApost *= gmax
    dApre *= gmax

    eqs = '''

    dv/dt  = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms)                : volt
    dgi/dt = -gi/(10*ms)               : volt
    '''

    P = NeuronGroup(5, eqs, threshold='v>-50*mV', reset='v=-60*mV')
    P2 = NeuronGroup(5, eqs, threshold='v>-50*mV', reset='v=-60*mV')
    P.v = -60 * mV
    P2.v=-60*mV
    Ce = Synapses(P, P2,
                 '''w : 1
                    dApre/dt = -Apre / taupre : 1 (event-driven)
                    dApost/dt = -Apost / taupost : 1 (event-driven)''',
                 on_pre='ge+=1.62*mV'
                 )

    Ce.connect()
    #Ci = Synapses(Pi, P, on_pre='gi-=9*mV')
    #Ci.connect(p=0.02)
    Ce.w='rand()*0.01'
    M = SpikeMonitor(P2)
    SM=StateMonitor(Ce,'w',record=True)
    run(20 * ms)
    maxNeuron=np.argmax(M.count)
    maxCount=np.max(M.count)
    for i in range(5):
        print SM[Ce[i,:]].w

    for i in range(5):
        print "Neuron ",i
        for j in range(5):
            print SM[Ce[i,j]].w[0][len(SM[Ce[i,j]].w)]
 

if __name__ == '__main__':
    # SETTING UP NETWORK
    image_width = 100
    image_height = 66
    N = image_height * image_width
    total_dog_species = 5

    taum = 10 * ms
    taupre = 20 * ms
    taupost = taupre
    Ee = 0 * mV
    vt = -54 * mV
    vr = -60 * mV
    El = -74 * mV
    taue = 5 * ms
    F = 15 * Hz
    gmax = .01
    dApre = .01
    dApost = -dApre * taupre / taupost * 1.05
    dApost *= gmax
    dApre *= gmax

    eqs_neurons = '''
    dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
    dge/dt = -ge / taue : 1
    '''

    eqs = '''
    dv/dt  = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms)                : volt
    dgi/dt = -gi/(10*ms)               : volt
    '''
    # Input neurons - one for each pixel in the image
    input_neurons = PoissonGroup(N, rates=(5 * Hz))

    # Training neurons - correct dog image has higher rate of 15Hz, other neurons have 5Hz
    training_neurons = PoissonGroup(total_dog_species, rates=(5 * Hz))

    # Output neurons - one for each dog species
    output_neurons = NeuronGroup(total_dog_species, eqs, threshold='v>vt', reset='v = vr', method='exact')

    S = Synapses(input_neurons, output_neurons,
                 '''w : 1
                    dApre/dt = -Apre / taupre : 1 (event-driven)
                    dApost/dt = -Apost / taupost : 1 (event-driven)''',
                 on_pre='ge+=1.62*mV'
                 )
    S.connect()
    S2 = Synapses(training_neurons, output_neurons,
                 '''w : 1'''
                 )
    # Only connect training neuron to corresponding output neuron
    for x in range(total_dog_species):
        S2.connect(i=x, j=x)

    S.w = 'rand() * gmax'
    S2.w = '1'

    #Monitors for synapse weights
    #monitorS=StateMonitor(S,'w',record=True)
    #monitorS2=StateMonitor(S2,'w',record=True)
    # TRAINING
    print "training"
    # Traverse images in image_resized folder
    root_dir = 'images/'
    counter = 0
    # Create list where dogList[i] = dog name for output neuron i
    dog_list = []
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        dir_name_split = dir_name.split('-')
        if len(dir_name_split) > 1:
            dog_name = dir_name_split[1]
            dog_list.append(dog_name)
            dog_num = len(dog_list) - 1
            counter += 1
        if counter <= total_dog_species:
            for file_name in file_list:
                img = Image.open(dir_name + "/" + file_name)
                img = img.resize((image_width, image_height), Image.ANTIALIAS)
                img = img.convert('L')
                data = list(img.getdata())
                # Get value of pixel, divide it by 10 for input value
                for i in range(len(data)):
                    data[i] = (data[i] / 10) * Hz

                # Training inputs. All inputs give 5Hz except for correct output neuron, which gives 15Hz
                # dog_num corresponds to the correct output neuron
                training_inputs = [5 * Hz] * total_dog_species
                training_inputs[dog_num] = 30 * Hz

                input_neurons.rates = data
                training_neurons.rates = training_inputs

                print("Training with image: " + file_name)

                run(50 * ms)
                img.close()
                #print(output_neurons.spikes)
	#Save trained synaptic weights
	# f=open('S.csv','w+')
    # for i in range(N):
        # for j in range(4000):
            # f.write(monitorS[S[i,j]].w[0][len(monitorS[S[i,j]].w)])
            # f.write(',')
        # f.write('\n')
    # f.close()
    # f2=open('S2.csv','w+')
    # for i in range(total_dog_species):
        # for j in range(total_dog_species):
            # f2.write(monitorS2[S2[i,j]].w[0][len(monitorS2[S2[i,j]].w)])
            # f2.write(',')
        # f2.write('\n')
    # f2.close()
	
    # TESTING
    print "testing"
    # Test with training data
    S2.w = '0'
    outMonitor=SpikeMonitor(output_neurons)
    store('after_learning')

    tests_correct = 0.0
    tests_total = 0.0

    counter = 0
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        dir_name_split = dir_name.split('-')
        if len(dir_name_split) > 1:
            dog_name = dir_name_split[1]
            dog_list.append(dog_name)
            dog_num = len(dog_list) - 1
            counter += 1
        if counter <= total_dog_species:
            for file_name in file_list:
                restore('after_learning')
                img = Image.open(dir_name + "/" + file_name)
                img = img.resize((image_width, image_height), Image.ANTIALIAS)
                img = img.convert('L')
                data = list(img.getdata())
                for i in range(len(data)):
                    data[i] = (data[i] / 10) * Hz

                tests_total += 1.0

                input_neurons.rates = data
                run(25 * ms)
                img.close()
                # Of the output_neurons spiked, check if correct dog species spikes
                #spikes = output_neurons.spikes
                print("Testing with image: " + file_name + " : dog_num = " + str(dog_num))
                #Find output neuron with highest spike frequency
                maxNeuron=np.argmax(outMonitor.count)
                #maxCount=np.max(outMonitor.count)
                if maxNeuron==counter-1:
                    tests_correct+=1.0
                print "Guess:",maxNeuron," is:",counter-1 
                #print(spikes)
                '''
                i = 0
                while i < len(spikes) and spikes[i] <= dog_num:
                    # Correct output neuron spiked
                    if spikes[i] == dog_num:
                        tests_correct += 1.0
                        i = len(spikes)
                    else:
                        i += 1
               '''
    print("Using testing data, the network correctly identified: " + str(tests_correct / tests_total) +
          "% of the images")
