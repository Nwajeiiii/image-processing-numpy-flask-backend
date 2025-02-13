from flask import Flask, request, send_file, jsonify
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Allow frontend access (CORS)
from flask_cors import CORS
CORS(app)

def process_image(image, option):
    """
    Processes the image based on the selected option.
    Options:
    1 - Grayscale Conversion
    2 - Edge Detection
    3 - Blurring
    4 - Brightness Adjustment
    5 - Inversion
    """

    # Convert image to NumPy array
    img_array = np.array(image)

    if option == "grayscale":
        # Convert to grayscale using weighted sum of RGB channels
        processed_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        processed_array = processed_array.astype(np.uint8)  # Ensure valid pixel values

    elif option == "edge_detection":
        # Convert to grayscale first
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        # Apply edge detection using gradient magnitude
        edges = np.abs(np.gradient(gray, axis=0)) + np.abs(np.gradient(gray, axis=1))
        processed_array = (edges / edges.max()) * 255  # Normalize
        processed_array = processed_array.astype(np.uint8)

    elif option == "blur":
        # Apply simple 3x3 averaging filter (basic blurring)
        kernel = np.ones((3, 3)) / 9.0
        processed_array = np.copy(img_array[:, :, :3])  # Copy RGB values

        for i in range(3):  # Loop through RGB channels
            processed_array[:, :, i] = np.convolve(img_array[:, :, i].flatten(), kernel.flatten(), mode='same').reshape(img_array.shape[:2])

    elif option == "brightness":
        # Increase brightness by adding 50 to pixel values, ensuring they stay within [0, 255]
        processed_array = np.clip(img_array[:, :, :3] + 50, 0, 255).astype(np.uint8)

    elif option == "invert":
        # Invert colors (negative effect)
        processed_array = 255 - img_array[:, :, :3]

    else:
        return None  # Invalid option

    # Convert NumPy array back to PIL image
    processed_image = Image.fromarray(processed_array)

    return processed_image

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handles the image upload and applies the selected processing option.
    Expects:
    - 'file': Image file
    - 'option': Processing option (grayscale, edge_detection, blur, brightness, invert)
    """

    if 'file' not in request.files or 'option' not in request.form:
        return jsonify({'error': 'File and processing option are required'}), 400

    file = request.files['file']
    option = request.form['option']

    image = Image.open(file.stream)  # Open image file

    processed_image = process_image(image, option)
    if processed_image is None:
        return jsonify({'error': 'Invalid processing option'}), 400

    # Convert image to bytes and send response
    img_io = io.BytesIO()
    processed_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
