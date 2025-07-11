from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), 'RGB')
img.save('sample.jpg')
print('sample.jpg created')
