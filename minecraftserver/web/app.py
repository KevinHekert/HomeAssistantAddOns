from flask import Flask, render_template_string

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
  <head>
    <title>Minecraft Bedrock UI</title>
  </head>
  <body>
    <h1>Minecraft Bedrock Server</h1>
    <p>Hier komt je status/config UI etc.</p>
  </body>
</html>
"""

@app.route("/")
def index():
    return HTML

if __name__ == "__main__":
    # Belangrijk voor Ingress: 0.0.0.0 en vaste poort
    app.run(host="0.0.0.0", port=8789)
