from pathlib import Path
from flask import Flask, request, render_template

from prediction import make_prediction_on_video, create_object_detection_model

# Define a flask app
app = Flask(__name__)
app.config['upload_folder'] = 'uploads'
app.config['model_output_folder'] = 'model_output'


@app.route('/', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = Path(__file__).resolve().parents[0]
        file_path = basepath/app.config['upload_folder']/f.filename
        f.save(file_path)

        # run prediction
        model = create_object_detection_model(model_directory=basepath/'ml_utils/model_data')
        make_prediction_on_video(file_path, model,
                                 output_directory_path=Path(basepath/f"static/{app.config['model_output_folder']}"))

        filename = f'prediction_{f.filename}'

        return render_template('prediction_done.html', filename=filename)

    return render_template('index.html')


@app.route('/display/<filename>')
def display_video(filename):
    return render_template('display.html', filename=Path(app.config['model_output_folder'])/filename)


if __name__ == '__main__':
    app.run(debug=True)
