import matplotlib.pyplot as plt

def check_mask_layers(img, layer = 0):
    '''
    Check condition:
    The output should have two layers. A 1 in layer 0 should indicate a background pixel, while a 1
    in layer 1 should indicate a foreground pixel.
    
    Args:
        img (torch.Tensor): 3D tensor of image, 2 channels and 2D matrix for each channel.
        layer (int): layer to visualize
    '''

    img_layer = img[layer]

    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()

    plt.imshow(img, cmap="gray")
    plt.colorbar()
    plt.title("Mask Visualization")
    plt.show()

