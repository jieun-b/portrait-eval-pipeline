import torchvision.transforms as T


def get_standard_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]),
    ])


def get_tensor_transform():
    return T.ToTensor()
