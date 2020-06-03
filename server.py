import os
from flask import Flask, jsonify, flash, request, redirect, url_for, render_template, json, current_app as app
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/form', methods=['POST', 'GET'])
def form():
	if request.method == 'POST':
		if 'file' not in request.files:
			return render_template("my_form.html")
		file = request.files['file']
		if file.filename == '':
			return render_template("index.html")
		else:
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.static_folder, "data/user_image"))
			return redirect('/response')
	return render_template('my_form.html')

@app.route('/response', methods=['POST', 'GET'])
def response():
    os.system('python3 ./neuronal_network/main.py')
    json_url = os.path.join("static/data", "example.json")
    data = json.load(open(json_url))
    return render_template('response.html', data=data)

@app.route('/json', methods=['POST', 'GET'])
def send_json():
	json_url = os.path.join("static/data", "example.json")
	data = json.load(open(json_url))
	return (data)

if __name__ == '__main__':
    app.run()
