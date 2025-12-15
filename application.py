from flask import Flask,request,render_template,jsonify
import pandas as pd
import numpy as np
import pickle

std = pickle.load(open("models/StandardScaler.pkl","rb"))
model = pickle.load(open("models/lin_model.pkl","rb"))

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/prediction",methods = ["GET","POST"])
def prediction():
    if request.method == "POST":
        features = [float(request.form["Temperature"]),float(request.form["RH"]),float(request.form['Ws']),float(request.form["Rain"]),
                    float(request.form["FFMC"]),float(request.form["DMC"]),float(request.form['ISI']),
                    float(request.form["Classes"]),float(request.form["Region"])]
       
        scaled = std.transform([features])#for 2d conversion i used square bracket
        prediction = model.predict(scaled)[0]
        return render_template("prediction.html",prediction = prediction)
    else:
        return render_template("prediction.html")



if __name__ == "__main__":
    app.run(debug=True)