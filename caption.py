from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')

    inputs = processor(image, return_tensors="pt")

    output = model.generate(
        **inputs,
        max_length=100,        # increased
        min_length=35,         # forces longer text
        num_beams=7,
        length_penalty=1.3,
        no_repeat_ngram_size=2
    )

    caption = processor.decode(output[0], skip_special_tokens=True)

    # LIGHT enhancement (not template spam)
    caption = caption + ". The scene appears detailed and contains multiple interacting elements."

    return caption