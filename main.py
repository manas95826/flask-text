from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model('model')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            res = model.predict(np.array([text]))
            if res[0][0] < 0.5:
                result = f"Human Generated 🙃! Probability of it being human: {(1 - res[0][0]) * 100:.2f}%"
            else:
                result = f"Written by an AI 🤖! Probability of it being AI: {(res[0][0]) * 100:.2f}%"
            return render_template('index.html', result=result)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
