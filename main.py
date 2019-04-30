import os
from PIL import Image
from brian2 import *
from matplotlib import pyplot

class LifNeuron:
    def __init__(self, weights):
        self.weights = weights
        self.avg_rate = 0
        self.num_occur = 0

    def run(self, current):
        resistance = 0.7
        capacitor = 1
        time_constant = resistance * capacitor
        time_diff = 0.15
        threshold_voltage = 1
        end_time = 5

        # Increase steady input current after each test
        current_constant = 0.15
        cur_voltage = 0
        cur_time = 0
        num_spikes = 0
        while cur_time < end_time:
            cur_time += time_diff

            cur_voltage_diff = (((-1 * cur_voltage) + (resistance * current)) / time_constant) * time_diff
            cur_voltage = cur_voltage + cur_voltage_diff

            if cur_voltage >= threshold_voltage:
                num_spikes += 1
                current += current_constant
                cur_voltage = 0

        cur_rate = num_spikes * (1000 / cur_time)
        self.avg_rate = ((self.avg_rate * self.num_occur) + cur_rate) / (self.num_occur + 1)
        self.num_occur += 1
        return cur_rate

    def learn(self, weight_num, vi, vj, i_avg, j_avg):
        weight_coeff = 0.000001
        min_weight = 0.25
        dif_weight = weight_coeff * (vi - i_avg) * (vj - j_avg)
        '''
        if weight_num == 1:
            max_weight = 1.0 - self.w2 - self.w3
            self.w1 += dif_weight
            if self.w1 < min_weight:
                self.w1 = min_weight
            elif self.w1 > max_weight:
                self.w1 = max_weight
        elif weight_num == 2:
            max_weight = 1.0 - self.w1 - self.w3
            self.w2 += dif_weight
            if self.w2 < min_weight:
                self.w2 = min_weight
            elif self.w2 > max_weight:
                self.w2 = max_weight
        elif weight_num == 3:
            max_weight = 1.0 - self.w1 - self.w2
            self.w3 += dif_weight
            if self.w3 < min_weight:
                self.w3 = min_weight
            elif self.w3 > max_weight:
                self.w3 = max_weight
        '''


class NeuralNetwork:
    def __init__(self):
        self.layer1 = []

def brian_example():
    eqs = '''
    dv/dt  = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms)                : volt
    dgi/dt = -gi/(10*ms)               : volt
    '''
    P = NeuronGroup(4000, eqs, threshold='v>-50*mV', reset='v=-60*mV')
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

if __name__ == '__main__':
    # image_width = 500
    # image_height = 333
    # # Set up neural network
    # network = NeuralNetwork()
    # for row in range(0, image_height):
    #     network.layer1.append([])
    #     for col in range(0, image_width):
    #         network.layer1[row].append(LifNeuron(0))
    #
    # # Traverse images in image_resized folder
    # rootDir = 'images/'
    # for dirName, subdirList, fileList in os.walk(rootDir):
    #     for fileN in fileList:
    #         # Get dog type from folder name
    #         dirNameSplit = dirName.split('-')
    #         dogName = dirNameSplit[1]
    #
    #         # Convert image into black and white. Rescale to 500x333 image
    #         newDirName = dirName.replace('\\', '/')
    #         img = Image.open(newDirName + '/' + fileN)
    #         img = img.resize((image_width, image_height), Image.ANTIALIAS)
    #         img = img.convert('L')
    #         WIDTH, HEIGHT = img.size
    #         data = list(img.getdata())
    #         data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]
    #
    #         # For each pixel,
    #         for row in data:
    #             for value in row:
    #                 # TODO run through network
    #                 print(value)


    brian_example()

    # TODO After training, test with user images (or run through images again and check accuracy)
