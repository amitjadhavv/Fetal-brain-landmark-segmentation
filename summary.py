from torchinfo import summary
from models.AttentionVNet import AttentionVNet
from configs.config import Config
model = AttentionVNet(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
summary(model, input_size=(1, 1, 32, 32, 32))
print(model)