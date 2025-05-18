from diffusers import StableDiffusionPipeline
import torch

model_id = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token="your_huggingface_token_here"
).to("cpu")  

prompt = "a fantasy landscape with mountains and a river during sunset"

image = pipe(prompt).images[0]

image.save("generated_image.png")

print("✅ Image generated and saved as generated_image.png")