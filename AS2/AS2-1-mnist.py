import struct
import numpy as np
from copy import copy

train_images_file = "data/train-images-idx3-ubyte"
train_labels_file = "data/train-labels-idx1-ubyte"
test_images_file = "data/t10k-images-idx3-ubyte"
test_labels_file = "data/t10k-labels-idx1-ubyte"
magic_image = 2051
magic_label = 2049

def loadfile(filename):
    data = []
    with open(filename, 'rb') as f:
        magic, n = struct.unpack(">II", f.read(8)) # >: big endian, I: unsigned int 32, 8 bytes = 64 bits
        
        nrows, ncols = (1, 1)
        if magic == magic_image:
            nrows, ncols = struct.unpack(">II", f.read(8))
        
        data = np.fromfile(f, dtype=np.uint8)
        if magic == magic_image:
            data = data.reshape((n, nrows*ncols))
        else:
            data = data.reshape((n, 1))
    f.close()
    return data

def preprocessing(images, labels, _bin):
    # bin
    if _bin == True: 
        images = binning(images,)
        
    # concatenate labels to images
    data = np.c_[images, labels]
    return data

def binning(data):
    binned_data = np.zeros(data.shape, dtype=np.uint8)
    
    it = np.nditer([data, binned_data],
                   ['external_loop', 'buffered'],
                   [['readonly'], ['writeonly']])

    with it:
        for instance, instance_new in it:
            instance_new[...] = instance >> 3

    return binned_data

def print_images(imaginations, mode):
    if mode == 0:
        for i, img in enumerate(imaginations): # img: (784, 32), value: count
            print(str(i) + ":")
            image = np.zeros((784, ), dtype=np.uint8)
            for j, pixel in enumerate(img):# j: 784 pixels
                # for k, value_count in enumerate(pixel): # k: 32 values, value_count: count
                white = np.sum(pixel[:16])
                black = np.sum(pixel[16:])
                if black > white: image[j] = 1
                else: image[j] = 0  
            print_image(image)

    else:
        for i, img in enumerate(imaginations):
            print(str(i) + ":")
            print_image(np.where(img > 127, 1, 0))

def print_image(image):
    for row in image.reshape((28,28)).tolist():
        for pixel in row:
            print(pixel, end=" ")
        print("")
    print("")

def calc_posterior(mode, instance, t, lookups):

    if mode == 0:
        likelihood, prior, likelihood_prior = lookups
        posterior = prior[t] # log prior
        for x_idx, x_val in enumerate(instance[:-1]):
            posterior += likelihood_prior[t, x_idx, x_val] # log likelihood
    elif mode == 1:
        means, variances, prior = lookups
        posterior = prior[t]
        for x_idx, x_val in enumerate(instance[:-1]):
            posterior += gaussian_likelihood(x_val, means[t, x_idx], variances[t, x_idx])
    return posterior

def gaussian_likelihood(x, mean, variance):
    '''
    2*pi = 6.283185307179586
    0.5*log(2*pi) = 0.9189385332046727
    sqrt(2*pi) = 2.5066282746310002,
    e = 2.718281828459045
    '''
    return -np.log(variance)/2-0.9189385332046727-0.5*((x-mean)**2/variance)
    
def build_lookup_discrete(data):
    print("Build lookup table (discrete)...")
    likelihood        = np.zeros((10, 784, 32)) # P(D|theta), shape = (y_value, x_index, x_value)
    prior             = np.zeros((10,)) # P(theta)
    likelihood_prior  = np.zeros((10, 784, 32)) 

    for instance in data: # 60000
        t = instance[-1]
        prior[t] += 1
        for x_idx, x_val in enumerate(instance[:-1]):
            likelihood[t, x_idx, x_val] += 1

    # likelihood times prior
    for t in range(10):
        for x_val in range(32):
            for x_idx in range(784):
                likelihood_prior[t, x_idx, x_val] = np.log(likelihood[t, x_idx, x_val]+1) - np.log(prior[t]+32)
        prior[t] = np.log(prior[t]+1) - np.log(60000+10)

    return likelihood, prior, likelihood_prior

def build_lookup_continuous(data):
    print("Build lookup table (continuous)...")
    prior      = np.zeros((10,))
    means      = np.zeros((10, 784))
    variances  = np.zeros((10, 784)) 

    for instance in data: # 60000
        t = instance[-1]
        prior[t] += 1
        for x_idx, x_val in enumerate(instance[:-1]):
            means[t, x_idx] += x_val
            variances[t, x_idx] += x_val**2

    for t in range(10):
        means[t, :] = np.divide(means[t, :], prior[t])
        variances[t, :] = np.divide(variances[t, :], prior[t]) - np.power(means[t, :], 2)
        prior[t] = np.log(prior[t]+1) - np.log(60000+10)
    variances += 15

    return means, variances, prior

def predict(mode, instance, lookups, _print, idx):
    
    scores = np.zeros((10,))
    for t in range(10):
        posterior = calc_posterior(mode, instance, t, lookups)
        scores[t] = posterior
    
    scores /= np.nansum(scores)
    prediction = np.argmin(scores)
    
    if _print:
        # print results
        print("Posterior (in log scale):")
        for i, score in enumerate(scores):
            print(str(i) + ": " + str(score))
        print("Prediction: " + str(prediction) + ", Ans: " + str(instance[-1]) + "\n")

    return prediction

def main():
    # Training
    train_images = loadfile(train_images_file)
    train_labels = loadfile(train_labels_file)
    train_data = [preprocessing(copy(train_images), train_labels, _bin=True),
                  preprocessing(copy(train_images), train_labels, _bin=False)]
    print("Training...")
    
    lookups = [build_lookup_discrete(train_data[0]),
               build_lookup_continuous(train_data[1])]

    # Testing
    test_images = loadfile(test_images_file)
    test_labels = loadfile(test_labels_file)
    test_data = [preprocessing(copy(test_images), test_labels, _bin=True),
                 preprocessing(copy(test_images), test_labels, _bin=False)]

    n = len(test_labels)

    while True:
        mode = int(input("Discrete or continuous (0 or 1): "))

        if mode not in (0, 1): continue

        error, _print = 0, True
        for i, instance in enumerate(test_data[mode]):
            if i == 5: _print = False 
            if instance[-1] == predict(mode, instance, lookups[mode], _print, i): continue
            error += 1
        print("...")
        print("Imagination of numbers in Bayesian classifier:\n")
        print_images(lookups[mode][0], mode)
        print("Error rate:", error/n)

if __name__ == '__main__':
    main()