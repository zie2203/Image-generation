from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoPipelineForText2Image
from fastapi.responses import FileResponse
import io
from PIL import Image

# Define the FastAPI app
app = FastAPI()

# Load the model and set it up
pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipeline_text2image = pipeline_text2image.to("cuda")

# Define the input data model
class TextPrompt(BaseModel):
    prompt: str
    guidance_scale: float = 7.5  # Default value; can be overridden
    num_inference_steps: int = 50  # Default value; can be overridden

@app.post("/generate-image")
async def generate_image(prompt: TextPrompt):
    try:
        # Generate the image
        image = pipeline_text2image(prompt=prompt.prompt, guidance_scale=prompt.guidance_scale, num_inference_steps=prompt.num_inference_steps).images[0]
        
        # Save the image to a byte stream
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Return the image as a response
        return FileResponse(img_byte_arr, media_type="image/png", filename="generated_image.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
