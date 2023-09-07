import video_confidence
from flask import Flask,request,jsonify
import boto3
from dotenv import load_dotenv

app= Flask(__name__)

@app.route('/confidence',methods=['POST'])
def get_confidence():
    data= request.get_json()
    file_name= data['file_name']

    score= video_confidence.VideoConfidence(file_name)
    return jsonify(score)

if __name__=='__main__':
    app.run()