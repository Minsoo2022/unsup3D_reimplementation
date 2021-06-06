def add_images(board, images, iteration) :
    #cliping
    for key in images.keys():
        image = images[key][:8]
        if len(images[key].shape) == 3:
            image = image.unsqueeze(1)
        if 'depth' in key:
            image = (image - 0.9) / (1.1 - 0.9)
        if image.shape[1] == 3:
            image = (image / 2 + 0.5).clamp(0,1)
        board.add_images(key, image, iteration)

def add_scalars(board, scalars, iteration) :
    keys = scalars.keys()
    for key in keys :
        board.add_scalar(key, scalars[key].mean(), iteration)

def add_losses(board, losses, iteration) :
    keys = losses.keys()
    for key in keys :
        board.add_scalar(key, losses[key], iteration)