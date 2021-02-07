import matplotlib.pyplot as plt 

import numpy as np

def plot_unmasked_and_masked( unmasked , masked ):
    fig, axs = plt.subplots(1, 2, figsize=(12,8))

    axs[0].imshow(unmasked )
    axs[0].axis('off')
    axs[0].set_title('unmasked face')
    
    #plot image and add the mask
    axs[1].imshow(masked)
    axs[1].axis('off')   
    axs[1].set_title('masked')

    # set suptitle
    plt.suptitle('unmasked vs masked')
    plt.show()
    
def correct_name(x) : 
    return x[:-1]