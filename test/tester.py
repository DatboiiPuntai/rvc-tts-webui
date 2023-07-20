from gradio_client import Client
import shutil
import os

client = Client("http://127.0.0.1:7860/")
result = client.predict(
				"Amelia",	# str (Option from: ['Amelia', 'Gura']) in 'RVC Model' Dropdown component
				"Howdy, there, it's ya boi Amelia. I love horse cocks!",	# str  in 'Input Text' Textbox component
				"neurosama",	# str (Option from: ['neurosama', 'snake']) in 'Tortoise voices' Dropdown component
				0,	# int | float  in 'Transpose' Number component
				"rmvpe",	# str  in 'Pitch extraction method 
				1,	# int | float (numeric value between 0 and 1) in 'Index rate' Slider component
				0.33,	# int | float (numeric value between 0 and 0.5) in 'Protect' Slider component
				"ultra_fast",	# str (Option from: ['ultra_fast', 'standard', 'high_quality', 'single_sample']) in 'Presets' Dropdown component
				api_name="/tortoise_tts"
)

save_path = 'test/output.wav'
shutil.copy(result[-1], save_path)