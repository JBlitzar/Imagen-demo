import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
import torchvision
import os
os.system(f"caffeinate -is -w {os.getpid()} &")
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

device = "mps" if torch.backends.mps.is_available() else "cpu"



img_size = 64
batch_size=16
epochs = 10
transforms = v2.Compose([
    #v2.PILToTensor(),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(img_size),
    v2.CenterCrop(size=(img_size, img_size))
    
])
dataset = torchvision.datasets.CocoCaptions(root=os.path.expanduser("~/torch_datasets/coco/train2017"), annFile= os.path.expanduser("~/torch_datasets/coco/annotations/captions_train2017.json"), transform=transforms)
# print('Number of samples: ', len(dataset))

# cap = dataset[4]
# img, label = cap

# print("Image Size: ", img.size())
# print(label)
# print(torch.max(img))
# print(torch.min(img))
# print(torch.mean(img))

dataloader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# unet for imagen

unet1 = Unet(
    dim = 32,
    cond_dim = 256,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True)
)

unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)

# imagen, which contains the unets above (base unet and super resoluting ones)

imagen = Imagen(
    unets = (unet1),
    image_sizes = (img_size),
    timesteps = 1000,
    cond_drop_prob = 0.1
).to(device)

trainer = ImagenTrainer(imagen)


# feed images into imagen, training each unet in the cascade

for i in trange(epochs):
    for step, (images, texts) in enumerate(pbar := tqdm(dataloader, dynamic_ncols=True)):
        texts = list(texts[0])
        #print(texts)
        
        


        loss = trainer(images, texts = texts, unet_number = 1)
        #loss.backward()

        

        pbar.set_description(f"Loss: {'%.4f' % loss}")
        if step % 500 == 499:
            
           
            print(f"Loss: {loss}")


            with open(f"ckpt/latest.pt", "wb+") as f:
                torch.save(imagen.state_dict(),f)



# do the above for many many many many steps
# now you can sample an image based on the text embeddings from the cascading ddpm

images = imagen.sample(texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
], cond_scale = 3.)

images.shape # (3, 3, 256, 256)