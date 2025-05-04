from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("app.html")

# 작동 안됨
@app.route("/contactthanks")
def contactthanks():
    return render_template("contactthanks.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/shortcodes")
def shortcodes():
    return render_template("shortcodes.html")

@app.route("/subscribe")
def subscribe():
    return render_template("subscribe.html")

@app.route("/video")
def video():
    return render_template("video.html")    

@app.route("/download")
def download():
    return render_template("download.html")

@app.route("/index")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
