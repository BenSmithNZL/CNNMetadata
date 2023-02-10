import torch
import torchvision


class BaselineModel(torch.nn.Module):

    def __init__(self, NUM_OF_CLASSES):
        super().__init__()

        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.base_model = torchvision.models.efficientnet_b0(weights=weights)

        for param in self.base_model.features.parameters():
            param.requires_grad = False

        self.base_model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280,
                            out_features=NUM_OF_CLASSES,
                            bias=True))

    def forward(self, image):
        self.output = self.base_model(image)
        return(self.output)


class MetadataModel(torch.nn.Module):

    def __init__(self, NUM_OF_CLASSES, METADATA_LENGTH):
        super().__init__()

        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.image_network = torchvision.models.efficientnet_b0(weights=weights)

        for param in self.image_network.features.parameters():
            param.requires_grad = False

        self.image_network = torch.nn.Sequential(
            *list(self.image_network.children())[:-1])

        self.connected_network = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280 + METADATA_LENGTH,
                            out_features=NUM_OF_CLASSES,
                            bias=True))

    def forward(self, image, metadata):
        self.image_output_1 = self.image_network(image)
        self.image_output_2 = self.image_output_1.reshape(self.image_output_1.size()[0], 1280)
        self.concatenated = torch.cat([self.image_output_2, metadata], 1)
        self.final_output = self.connected_network(self.concatenated)
        return(self.final_output)
