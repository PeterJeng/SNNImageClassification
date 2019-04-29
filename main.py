import os
from PIL import Image

if __name__ == '__main__':
    rootDir = 'images/'
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fileN in fileList:
            dirNameSplit = dirName.split('-')
            dogType = dirNameSplit[1]


class LifNeuron:
    def __init__(self, w1, w2, w3):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.avg_rate = 0
        self.num_occur = 0

    def run(self, current1, current2, current3):
        resistance = 0.7
        capacitor = 1
        time_constant = resistance * capacitor
        time_diff = 0.15
        threshold_voltage = 1
        end_time = 5

        # Increase steady input current after each test
        current = (current1 * self.w1) + (current2 * self.w2) + (current3 * self.w3)
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


if __name__ == '__main__':
    # TODO Create initial neural network

    # Traverse images in image_resized folder
    rootDir = 'images/'
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fileN in fileList:
            # Get dog type from folder name
            dirNameSplit = dirName.split('-')
            dogName = dirNameSplit[1]

            # Get greyscale value of each pixel in image
            newDirName = dirName.replace('\\', '/')
            img = Image.open(newDirName + '/' + fileN)
            WIDTH, HEIGHT = img.size
            data = list(img.getdata())
            data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]

            # For each pixel,
            for row in data:
                for value in row:
                    # TODO run through network
                    print(value)
