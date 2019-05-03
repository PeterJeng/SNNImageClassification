import os
from PIL import Image
from brian2 import *
from matplotlib import pyplot


if __name__ == '__main__':
    # SETTING UP NETWORK
    image_width = 100
    image_height = 66
    N = image_height * image_width
    total_dog_species = 10

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

    # Input neurons - one for each pixel in the image
    input_neurons = PoissonGroup(N, rates=(5 * Hz))

    # Training neurons - correct dog image has higher rate of 15Hz, other neurons have 5Hz
    training_neurons = PoissonGroup(total_dog_species, rates=(5 * Hz))

    # Output neurons - one for each dog species
    output_neurons = NeuronGroup(total_dog_species, eqs_neurons, threshold='v>vt', reset='v = vr', method='exact')

    S = Synapses(input_neurons, output_neurons,
                 '''w : 1
                    dApre/dt = -Apre / taupre : 1 (event-driven)
                    dApost/dt = -Apost / taupost : 1 (event-driven)''',
                 on_pre='''ge += w
                        Apre += dApre
                        w = clip(w + Apost, 0, gmax)''',
                 on_post='''Apost += dApost
                         w = clip(w + Apre, 0, gmax)''',
                 )
    S.connect()
    S2 = Synapses(training_neurons, output_neurons,
                 '''w : 1
                    dApre/dt = -Apre / taupre : 1 (event-driven)
                    dApost/dt = -Apost / taupost : 1 (event-driven)''',
                 on_pre='''ge += w
                            Apre += dApre
                            w = clip(w + Apost, 0, gmax)''',
                 on_post='''Apost += dApost
                             w = clip(w + Apre, 0, gmax)''',
                 )
    # Only connect training neuron to corresponding output neuron
    for x in range(total_dog_species):
        S2.connect(i=x, j=x)

    S.w = 'rand() * gmax'
    S2.w = 'rand() * gmax'


    # TRAINING
    # Traverse images in image_resized folder
    root_dir = 'images/'
    # Create list where dogList[i] = dog name for output neuron i
    dog_list = []
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        dir_name_split = dir_name.split('-')
        if len(dir_name_split) > 1:
            dog_name = dir_name_split[1]
            dog_list.append(dog_name)
            dog_num = len(dog_list) - 1
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
            training_inputs[dog_num] = 15 * Hz

            input_neurons.rates = data
            training_neurons.rates = training_inputs

            print("Training with image: " + file_name)

            run(10 * ms)

            print(output_neurons.spikes)

    # TESTING
    # Test with training data
    store('after_learning')

    tests_correct = 0.0
    tests_total = 0.0

    for dir_name, subdir_list, file_list in os.walk(root_dir):
        dir_name_split = dir_name.split('-')
        if len(dir_name_split) > 1:
            dog_name = dir_name_split[1]
            dog_list.append(dog_name)
            dog_num = len(dogList) - 1
        for file_name in file_list:
            restore('after_learning')
            img = Image.open(dir_name + "/" + file_name)
            img = img.resize((image_width, image_height), Image.ANTIALIAS)
            img = img.convert('L')
            data = list(img.getdata())

            tests_total += 1

            input_neurons.rates = data

            run(100 * ms)

            # Of the output_neurons spiked, check if correct dog species spikes
            spikes = output_neurons.spikes
            i = 0
            while i < len(spikes) and spikes[i] <= dog_num:
                # Correct output neuron spiked
                if spikes[i] == dog_num:
                    tests_correct += 1
                    i = len(spikes)
                else:
                    i += 1

    print("Using testing data, the network correctly identified: " + str(tests_correct / tests_total) +
          "% of the images")


def brian_example():
    '''
    eqs = '''

    '''
    dv/dt  = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms)                : volt
    dgi/dt = -gi/(10*ms)               : volt
    '''

    '''
    P = NeuronGroup(6600, eqs, threshold='v>-50*mV', reset='v=-60*mV')
    P.v = -60 * mV
    Pe = P[:3200]
    Pi = P[3200:]
    Ce = Synapses(Pe, P, on_pre='ge+=1.62*mV')
    Ce.connect(p=0.02)
    Ci = Synapses(Pi, P, on_pre='gi-=9*mV')
    Ci.connect(p=0.02)
    M = SpikeMonitor(P)
    run(1 * second)
    plot(M.t / ms, M.i, '.')
    show()

    '''