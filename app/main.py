from fastapi import FastAPI, File, UploadFile
import pandas as pd
from io import StringIO
import joblib
from fastapi.responses import StreamingResponse
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/inference")
async def upload_csv(file: UploadFile = File(...)):
    # Read uploaded file content
    content = await file.read()
    # Decode bytes to string and load into a Pandas DataFrame
    df = pd.read_csv(StringIO(content.decode("utf-8")))

    # Get the absolute path to the model
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'trained_model.joblib')
 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model file is in the correct location.")
    
    model = joblib.load(model_path)
    
    predictions = model.predict(df)

    df_predictions = pd.DataFrame(df)
    df_predictions['Prediction'] = predictions

    columns = df_predictions.columns.tolist()

    columns.remove('Prediction')
    columns.insert(0, 'Prediction')

    df_predictions = df_predictions[columns]

     # Convert DataFrame to CSV
    csv_buffer = StringIO()
    df_predictions.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Return CSV as a response
    return StreamingResponse(
        csv_buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=inference_results.csv"}
    )
