# DeGAN: a ProGAN to generate DeGods

The objective of the project was to generate new DeGods NFTs using a GAN. First, I have developed a DCGAN but this GAN architecture is not suited to generate high-resolution images. So, I have developed this ProGAN model to generate 256x256 pixels images. The training took 2 days using a GPU on Google Colab.    

## Results

Real DeGods NFT images:
<p float="left">
  <img src="https://raw.githubusercontent.com/louisreberga/degan/main/images/DeGods_0.jpg" width="200" />
  <img src="https://raw.githubusercontent.com/louisreberga/degan/main/images/DeGods_1.jpg" width="200" />
  <img src="https://raw.githubusercontent.com/louisreberga/degan/main/images/DeGods_2.jpg" width="200" />
  <img src="https://raw.githubusercontent.com/louisreberga/degan/main/images/DeGods_3.jpg" width="200" />
</p>

Generated DeGods NFT:
<p float="left">
  <img src="https://raw.githubusercontent.com/louisreberga/degan/main/images/DeGAN_0.jpg" width="200" />
  <img src="https://raw.githubusercontent.com/louisreberga/degan/main/images/DeGAN_1.jpg" width="200" />
  <img src="https://raw.githubusercontent.com/louisreberga/degan/main/images/DeGAN_2.jpg" width="200" />
  <img src="https://raw.githubusercontent.com/louisreberga/degan/main/images/DeGAN_3.jpg" width="200" />
</p>

## References
1 - Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen (2017). Progressive Growing of GANs for Improved Quality, Stability, and Variation \
2 - Sarah Wolf (2018) ProGAN: How NVIDIA Generated Images of Unprecedented Quality
