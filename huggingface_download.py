from huggingface_hub import snapshot_download

def download_all():
    print("Downloading: sd-vae-ft-mse")
    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse", 
        local_dir="./pretrained_model/sd-vae-ft-mse"
    )
    
    print("Downloading: animatediff-motion-adapter-v1-5-2")
    snapshot_download(
        repo_id="guoyww/animatediff-motion-adapter-v1-5-2", 
        local_dir="./pretrained_model/animatediff"
    )

    print("Downloading: sd-image-variations-diffusers")
    snapshot_download(
        repo_id="lambdalabs/sd-image-variations-diffusers", 
        local_dir="./pretrained_model", 
        allow_patterns=["image_encoder/*"]
    )
    snapshot_download(
        repo_id="lambdalabs/sd-image-variations-diffusers", 
        local_dir="./pretrained_model/sd-image-variations-diffusers", 
    )

    print("Downloading: stable-diffusion-v1-5")
    snapshot_download(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        local_dir="./pretrained_model/stable-diffusion-v1-5",
        allow_patterns=[
            "feature_extractor/preprocessor_config.json", 
            "model_index.json", 
            "unet/config.json", 
            "unet/diffusion_pytorch_model.safetensors",
            "v1-inference.yaml"
        ],
    )

    print("Downloading: FollowYourEmoji")
    snapshot_download(
        repo_id="YueMafighting/FollowYourEmoji", 
        local_dir="./pretrained_model", 
        allow_patterns=["ckpts/*"]
    )
    
    print("Downloading: LivePortrait")
    snapshot_download(
        repo_id="KwaiVGI/LivePortrait",
        local_dir="./pretrained_model/liveportrait",
        ignore_patterns=["*.git*", "README.md", "docs"]
    )

    print("All downloads completed.")

if __name__ == "__main__":
    download_all()
