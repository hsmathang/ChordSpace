import json
import base64

notebook_path = 'd:\\Documents\\Tesis MSC\\TesisModulos\\TesisModulos\\rugosidad_model'
output_image_path = 'sethares_curva_notebook.png'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_content = json.load(f)

    image_data = None
    # Search backwards, as plots are often at the end of notebooks
    for cell in reversed(nb_content['cells']):
        if 'outputs' in cell:
            for output in reversed(cell['outputs']):
                if 'data' in output and 'image/png' in output['data']:
                    image_data = output['data']['image/png']
                    break
            if image_data:
                break

    if image_data:
        with open(output_image_path, 'wb') as f:
            f.write(base64.b64decode(image_data))
        print(f"Image extracted from notebook and saved as {output_image_path}")
    else:
        print("No PNG image found in the notebook outputs.")

except Exception as e:
    print(f"An error occurred: {e}")
