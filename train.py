import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import datapracess
import VAE
import textEncode
import UNet
import matplotlib.pyplot as plt
import numpy as np
# 参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
image_size = 64

epochs = 100

latent_dim = 256
text_dim = 256
beta_start = 0.0001
beta_end = 0.02
timesteps = 1000

# 数据集
dataset = datapracess.TextToImageDataset(data_dir="data", image_size=image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型
vae = VAE.VAE(latent_dim=latent_dim).to(device)
text_encoder = textEncode.TextEncoder(vocab_size=dataset.vocab_size).to(device)
unet = UNet.UNet(text_dim=text_dim).to(device)

# 优化器
optimizer_vae = optim.Adam(vae.parameters(), lr=1e-3)
optimizer_unet = optim.Adam(unet.parameters(), lr=1e-4)
optimizer_text = optim.Adam(text_encoder.parameters(), lr=1e-4)


# VAE损失函数
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss


# 扩散过程参数
betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, 0).to(device)

# 训练循环
for epoch in range(epochs):
    for i, batch in enumerate(dataloader):
        real_images = batch["image"].to(device)
        text = batch["text"].to(device)

        # VAE训练
        optimizer_vae.zero_grad()
        recon, mu, logvar = vae(real_images)
        loss_vae = vae_loss(recon, real_images, mu, logvar)
        loss_vae.backward()
        optimizer_vae.step()

        # 扩散模型训练
        # 采样噪声步长
        t = torch.randint(0, timesteps, (batch_size,), device=device)
        noise = torch.randn_like(real_images)
        # print(alphas_cumprod[t].sqrt().view(-1, 1, 1, 1).shape)
        # print(real_images.shape)
        noisy_images = (
                alphas_cumprod[t].sqrt().view(-1, 1, 1, 1) * real_images
                + (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1) * noise
        )

        # 文本编码
        text_feat = text_encoder(text)

        # UNet预测噪声
        optimizer_unet.zero_grad()
        # print(noisy_images.shape)
        # print(t.unsqueeze(1).float().shape)
        # print(text_feat.shape)
        pred_noise = unet(noisy_images, t.unsqueeze(1).float(), text_feat)
        loss_unet = F.mse_loss(pred_noise, noise)
        loss_unet.backward()
        optimizer_unet.step()

        # 打印进度
        if i % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} "
                  f"VAE Loss: {loss_vae.item():.4f}, UNet Loss: {loss_unet.item():.4f}")

    # 每个epoch保存模型
    if (epoch + 1) % 5 == 0:
        torch.save({
            'vae': vae.state_dict(),
            'text_encoder': text_encoder.state_dict(),
            'unet': unet.state_dict()
        }, f"model_epoch_{epoch}.pth")
# 加载模型
checkpoint = torch.load("model_epoch_99.pth")
vae.load_state_dict(checkpoint['vae'])
text_encoder.load_state_dict(checkpoint['text_encoder'])
unet.load_state_dict(checkpoint['unet'])


# 生成图像
def generate_image(text_prompt):
    text_seq = dataset.text_to_sequence(text_prompt).unsqueeze(0).to(device)
    text_feat = text_encoder(text_seq)

    # 扩散采样（简化版）
    x = torch.randn(1, 3, image_size, image_size).to(device)
    for t in reversed(range(timesteps)):
        t_tensor = torch.tensor([t]).to(device)
        pred_noise = unet(x, t_tensor.float(), text_feat)
        x = (x - betas[t] / (1 - alphas_cumprod[t]) * pred_noise) / alphas[t].sqrt()

    x = vae.decoder(x)
    return x.detach().cpu().numpy()[0]


# 示例生成
image = generate_image("一只熊猫在竹林里吃竹子")
plt.imshow(np.transpose(image, (1, 2, 0)))
plt.show()