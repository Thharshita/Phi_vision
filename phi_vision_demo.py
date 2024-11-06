from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import torch

model_id = "microsoft/Phi-3-vision-128k-instruct" 

model = AutoModelForCausalLM.from_pretrained(model_id, 
                                            device_map="cuda",
                                            trust_remote_code=True, 
                                            _attn_implementation='eager',
                                            torch_dtype="auto") # use _attn_implementation='eager' to disable flash attention

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

# messages = [ 
#     {"role": "user", "content": "<|image_1|>\nWhat is shown in this image?"}, 
#     {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."}, 
#     {"role": "user", "content": "Provide insightful questions to spark discussion."} 
# ] 

messages = [
    {"role": "user", "content": "<|image_1|>\nOCR the text of the image. Extract the text of the following fields and put it in a JSON format: \
'नगर / Municipality', 'कौग्नोम / Surname', 'नाम / NAME', 'जन्म का स्थान और तिथि / DOB', \
'लिंग / SEX', 'ऊंचाई / HEIGHT', 'नागरिकता / NATIONALITY', 'जारी / ISSUING', 'समाप्ति / EXPIRY'. Read the code at the top right and put it in the JSON field 'CODE'"}
]

# url = "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" 
# image = Image.open(requests.get(url, stream=True).raw)   #raw gives file like oject


image_ = r"D:\Phi_3_vision\Images\r9.jpg"  #raw gives file like oject
image= Image.open(image_) #raw gives file like oject
"""
Open the Image: It reads the image data from the provided file path or file-like object.

Identify the Format: It automatically detects the image format (such as JPEG, PNG, GIF, etc.) based on the data.

Create an Image Object: It returns an instance of the Image class that represents the opened image. This object contains methods and attributes to manipulate the image, such as resizing, cropping, rotating, or converting to different formats.

Lazy Loading: The image data is loaded lazily, meaning that the actual image data is not fully loaded into memory until you perform an operation that requires it (like displaying or processing the image).

: It returns an image object, which represents the image data in memory.
"""

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 

generation_args = { 
    "max_new_tokens": 500, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

print(response) 