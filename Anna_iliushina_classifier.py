import torch
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision


class OpenEyesClassificator:
    def __init__(self):
        """инициализируется и загружается модель"""

        self.sigmoid = torch.nn.Sigmoid()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_model_weights = "resnet50_best_weights.pth"
        self.model = self.custom_resnet50()
        self.model.load_state_dict(torch.load(best_model_weights, weights_only=True))
        self.model.to(self.device)

    def custom_resnet50(self):

        model = torchvision.models.resnet50(pretrained=True, progress=True)
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=1000, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=1000, out_features=1, bias=True),
        )
        return model

    def predict(self, inpIm):
        """inpIm - полный путь к изображению глаза, который возвращает
        is_open_score - float score классификации от 0.0 до 1.0
        (где 1 - открыт, 0 - закрыт)"""

        self.model.eval()
        # загрузить изображение
        im = Image.open(inpIm)
        transform = v2.Compose([v2.ToTensor()])
        tensor = transform(im)
        # продублировать чб изображение 3 раза для имитации RGB
        rgb_batch = np.repeat(tensor[..., np.newaxis], 3, 0)
        rgb_batch = torch.squeeze(rgb_batch, 3)
        # нормализовать
        input = torch.tensor([np.array(rgb_batch) / 255])

        input_image = input.float().to(self.device)
        is_open_score = round(self.sigmoid(self.model(input_image)).item(), 5)

        return is_open_score


# путь до изборажения
inpIm = "eyes_images/unclear/000050.jpg"
clf = OpenEyesClassificator()
is_open_score = clf.predict(inpIm)

# визуализация изображения с предсказанием
plt.imshow(Image.open(inpIm), cmap="gray")
plt.title("predict: {}".format(is_open_score))
plt.axis("off")
plt.show()
