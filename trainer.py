from datetime import datetime
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import util
from glow import Glow

class GlowTrainer(object):
    def __init__(self):
        pass

    def train(self, params):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size = params["batch_size"]
        learning_rate = params["learning_rate"]
        max_epoch = params["max_epoch"]
        interval = params["interval"]
        image_shape = params["image_shape"]
        dataset_name = params["dataset_name"]

        max_grad_clip = params["max_grad_clip"]
        max_grad_norm = params["max_grad_norm"]

        train_dataset = util.ImageDataset(params)
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=batch_size,
                                      num_workers=4,
                                      shuffle=True,
                                      drop_last=True)

        dt_now = datetime.now()
        dt_seq = dt_now.strftime("%y%m%d_%H%M")
        result_dir = os.path.join("./result", f"{dt_seq}_{dataset_name}")
        weight_dir = os.path.join(result_dir, "weights")
        sample_dir = os.path.join(result_dir, "sample")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(weight_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        glow = Glow(params).to(device)

        optimizer = Adam(
            glow.parameters(),
            lr=learning_rate
            )

        initialized = False
        for epoch in range(max_epoch):
            for i, batch in enumerate(train_dataloader):
                batch = batch.to(device)
                if not initialized:
                    glow.initialize_actnorm(batch)
                    initialized = True
                z, nll = glow.inference(batch)

                loss_generative = torch.mean(nll)

                optimizer.zero_grad()
                loss_generative.backward()
                torch.nn.utils.clip_grad_value_(glow.parameters(), max_grad_clip)
                torch.nn.utils.clip_grad_norm_(glow.parameters(), max_grad_norm)
                optimizer.step()

                print(f"epoch {epoch} {i}/{len(train_dataloader)}, loss: {loss_generative.item():.4f}")

            if epoch % interval == 0:
                torch.save(glow.state_dict(), f"{weight_dir}/{epoch}_glow.pth")
                torch.save(optimizer.state_dict(), f"{weight_dir}/{epoch}_opt.pth")
                filename = f"{epoch}_glow.png"
                with torch.no_grad():
                    img = glow.generate(z, eps_std=0.5)
                    util.save_samples(img, sample_dir, filename, image_shape, num_tiles=4)


            

