from flask import Flask, render_template, redirect, jsonify, Response
from face_recognition import AdvancedFaceRecognition

app = Flask(__name__)

fr = AdvancedFaceRecognition()



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(fr.run(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True,port=8079)