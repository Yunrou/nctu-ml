import os, sys, getopt

from kernel_kmeans import KernelKMeans
from spectral_clustering import SpectralClustering
from util import imread, kernel

def argparser(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:k:path:method:init:gamma_s:gamma_c", ["infile=", "k=",
                                                            "method=", "init=", "gamma_s=", "gamma_c="])
    except getopt.GetoptError:
        print('AS6.py -i <infile> ...')
        sys.exit(2)

    infile, k, path = '', 2, ''
    method, init, gamma_s, gamma_c = '', '', '', ''
    for opt, arg in opts:
        if opt == '-i': infile = arg
        elif opt == '-k': k = int(arg)
        elif opt == '--method': method = arg
        elif opt == '--init': init = arg
        elif opt == '--gamma_s': gamma_s = float(arg)
        elif opt == '--gamma_c': gamma_c = float(arg)
    return infile, k, method, init, gamma_s, gamma_c
  
def main(argv):
    
    infile, k, method, init, gamma_s, gamma_c = argparser(argv)
    
    # Load image and flatten it into data
    data = imread(infile)
    
    outfile = os.path.join('media', infile.split('.')[0], '_'.join([method, str(k), init, 
                                                                    ''.join(str(gamma_s).split('.')), 
                                                                    ''.join(str(gamma_c).split('.'))]))
    G = kernel(data, gamma_s, gamma_c)
    if method == 'NSC': # spectral clustering
        prediction = SpectralClustering(G, k, 'normalized', init, outfile)
    elif method == 'UNSC':
        prediction = SpectralClustering(G, k, 'unnormalized', init, outfile)
    elif method == 'kkmeans':
        prediction = KernelKMeans(G, k, init, outfile)

if __name__ == '__main__':
    main(sys.argv[1:])