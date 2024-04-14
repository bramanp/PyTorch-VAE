from dataset import VAEDataset
from pytorch_lightning.utilities.seed import seed_everything

data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

data.setup()

# Dapatkan dataloader untuk dataset validasi
val_loader = data.val_dataloader()

# Specify the directory in your Google Drive to save the validation data
save_dir = "/content/drive/MyDrive/validation_data"

# Iterate through the validation data and save images
for i, (images, _) in enumerate(val_loader):
    for j in range(len(images)):
        image = images[j]
        save_path = os.path.join(save_dir, f"image_{i * val_loader.batch_size + j}.jpg")
        torchvision.utils.save_image(image, save_path)

print("Validation data saved successfully.")