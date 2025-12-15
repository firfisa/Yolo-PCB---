"""
Attention mechanisms for PCB defect detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel attention module of CBAM."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Initialize channel attention module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel compression
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through channel attention."""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module of CBAM."""
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize spatial attention module.
        
        Args:
            kernel_size: Kernel size for convolution
        """
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spatial attention."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(attention_input))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        """
        Initialize CBAM module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CBAM."""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Initialize SE block.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel compression
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SE block."""
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECA(nn.Module):
    """Efficient Channel Attention (ECA) module."""
    
    def __init__(self, in_channels: int, gamma: int = 2, b: int = 1):
        """
        Initialize ECA module.
        
        Args:
            in_channels: Number of input channels
            gamma: Gamma parameter for kernel size calculation
            b: Beta parameter for kernel size calculation
        """
        super().__init__()
        kernel_size = int(abs((torch.log2(torch.tensor(in_channels, dtype=torch.float32)) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ECA."""
        b, c, h, w = x.size()
        
        # Global average pooling
        y = self.avg_pool(x)
        
        # 1D convolution
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Sigmoid activation
        y = torch.sigmoid(y)
        
        return x * y.expand_as(x)


class CoordAttention(nn.Module):
    """Coordinate Attention module."""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 32):
        """
        Initialize Coordinate Attention module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel compression
        """
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, in_channels // reduction_ratio)
        
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Coordinate Attention."""
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_w * a_h
        
        return out


class AttentionBlock(nn.Module):
    """Configurable attention block supporting multiple attention mechanisms."""
    
    def __init__(self, in_channels: int, attention_type: str = "cbam", **kwargs):
        """
        Initialize attention block.
        
        Args:
            in_channels: Number of input channels
            attention_type: Type of attention mechanism ('cbam', 'se', 'eca', 'coord')
            **kwargs: Additional arguments for specific attention modules
        """
        super().__init__()
        
        if attention_type.lower() == "cbam":
            self.attention = CBAM(in_channels, **kwargs)
        elif attention_type.lower() == "se":
            self.attention = SEBlock(in_channels, **kwargs)
        elif attention_type.lower() == "eca":
            self.attention = ECA(in_channels, **kwargs)
        elif attention_type.lower() == "coord":
            self.attention = CoordAttention(in_channels, **kwargs)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention block."""
        return self.attention(x)