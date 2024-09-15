from .unet_parts import *
from liquidnet.main import LiquidNet

class ModifiedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_units, bilinear=False):
        super(ModifiedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Adjusted size based on observed output
        channels_after_down = 1024  # 1024 channels
        height_after_down = 18
        width_after_down = 25

        # Adjust flattened size based on actual size
        flattened_size = channels_after_down * height_after_down * width_after_down

        # Define linear layers with the correct input and output sizes
        self.liquidnet_input_layer = nn.Linear(flattened_size, num_units)
        self.liquid_net = LiquidNet(num_units=num_units)
        self.liquidnet_output_layer = nn.Linear(num_units, flattened_size)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Reshape x5 for LiquidNet
        batch_size, channels, height, width = x5.size()
        x5_flat = x5.reshape(batch_size, -1)

        # Ensure dimensions match for linear layer
        x5_lq_in = self.liquidnet_input_layer(x5_flat)

        # Initialize hidden_state for LiquidNet with zeros
        hidden_state = torch.zeros(batch_size, self.liquid_net.state_size).to(x.device)

        # Pass through LiquidNet
        x5_flat, _ = self.liquid_net(x5_lq_in, hidden_state)

        # Adjust output dimensions back to original
        x5_out = self.liquidnet_output_layer(x5_flat)
        x5 = x5_out.reshape(batch_size, channels, height, width)  # Reshape back to original

        # Pass through U-Net decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
