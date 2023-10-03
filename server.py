from flask import Flask, jsonify, request
import os
import subprocess

app = Flask(__name__)


@app.route('/list-files')
def list_files():
    screenshot_files = [filename for filename in os.listdir(
        '.') if filename.startswith('screenshot')]
    return jsonify({"files": screenshot_files, "count": len(screenshot_files)})


@app.route('/open-capture')
def open_capture():
    subprocess.run("python main.py", shell=True)


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if 'email' in data and 'password' in data:
        return jsonify({
            "is_login_successful": data['email'] == "menuka@gmail.com" and data['password'] == "password"
        })


if __name__ == '__main__':
    app.run(debug=True)
