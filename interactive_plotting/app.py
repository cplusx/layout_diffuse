import os
import time
from flask import Flask, render_template, request, jsonify, make_response, send_file
from PIL import Image

app = Flask(__name__)

rectangles = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image/<image_name>')
def get_image(image_name):
    print('INFO: image_name =', image_name)
    image_path = os.path.join('tmp', image_name)
    resize_image(image_path, target_height=384)
    return send_file(image_path, mimetype='image/jpg')

@app.route('/get_sd_images', methods=['GET', 'POST'])
def get_sd_images():
    if request.method == 'POST':
        data = request.get_json()
        rectangles = data.get('rectangles', [])
        # do something with the rectangles data
        this_hash = save_rectangles(rectangles)
        try:
            # wait for the wait_for_image coroutine to complete and return its result
            image_path = wait_for_image(this_hash, timeout=200)
            return jsonify({'image_path': image_path})
        except TimeoutError:
            return make_response(jsonify({'error': 'Timeout waiting for image'}), 404)
        except Exception as e:
            print(e)
            return make_response(jsonify({'error': 'Error in wait_for_image coroutine'}), 500)

    return make_response(jsonify({'message': 'Method not allowed'}), 405)

def wait_for_image(hash_value, timeout=60):
    # set up a file system watcher to monitor for the image file
    watched_folder = 'tmp'
    watched_file = hash_value + '.jpg'
    timeout = time.time() + timeout
    while time.time() < timeout:
        for file_name in os.listdir(watched_folder):
            if file_name == watched_file or file_name == 'hamburger_pic.jpeg':
                save_path = os.path.join(watched_folder, watched_file)
                return watched_file
        time.sleep(1)

    raise TimeoutError('Timeout waiting for image')

def save_rectangles(rectangles, save_dir='tmp'):
    # save rectangles to a file, the input is a list of dictionary
    # e.g., [{'x1': 107, 'y1': 271, 'x2': 386, 'y2': 407, 'color': '#2605af', 'class': 'car'}]

    # make dir if not existing
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # convert pixel values to relative size
    for rectangle in rectangles:
        x1, y1, x2, y2 = rectangle['x1'], rectangle['y1'], rectangle['x2'], rectangle['y2']
        w, h = 512, 512  # canvas size is 512x512 pixels
        rectangle['x'] = x1 / w
        rectangle['y'] = y1 / h
        rectangle['w'] = (x2 - x1) / w
        rectangle['h'] = (y2 - y1) / h
        rectangle.pop('x1', None)
        rectangle.pop('y1', None)
        rectangle.pop('x2', None)
        rectangle.pop('y2', None)

    # compute a hash for saving file name
    hash_name = str(hash(str(rectangles)))[1:11]
    save_name = hash_name + '.txt'
    save_path = os.path.join(save_dir, save_name)

    # save rectangles to file in specified format
    with open(save_path, 'w') as f:
        for rectangle in rectangles:
            x, y, w, h = rectangle['x'], rectangle['y'], rectangle['w'], rectangle['h']
            class_id = rectangle['class']
            f.write(f"{x},{y},{w},{h},{class_id}\n")

    return hash_name

def resize_image(image_path, target_height=256):
    img = Image.open(image_path)
    height_percent = target_height / float(img.size[1])
    width_size = int((float(img.size[0]) * float(height_percent)))
    img = img.resize((width_size, target_height), Image.ANTIALIAS)
    img.save(image_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')