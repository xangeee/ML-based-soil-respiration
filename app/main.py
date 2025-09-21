from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
from io import StringIO
import joblib
from fastapi.responses import StreamingResponse
import os
from typing import Tuple, Any
import logging
import numpy as np

# Configure logging with file storage
import logging.handlers
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure logging with both file and console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers
logger.handlers.clear()

# Create formatters
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File handler with rotation (10MB max, keep 5 files)
file_handler = logging.handlers.RotatingFileHandler(
    filename=os.path.join(logs_dir, 'api.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log startup message
logger.info("API logging initialized - logs stored in logs/api.log")

app = FastAPI(
    title="ML-based Soil Respiration API",
    description="API for soil respiration and nee predictions using Machine Learning models",
    version="1.0.0"
)

def load_model_and_scaler(model_name: str) -> Tuple[Any, Any]:
    """
    Loads the model and scaler corresponding to the specified name.
    
    Args:
        model_name: Model name ('rs' or 'nee')
        
    Returns:
        Tuple with (model, scaler)
        
    Raises:
        FileNotFoundError: If model or scaler files are not found
    """
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_path = os.path.join(models_dir, f'{model_name}.joblib')
    scaler_path = os.path.join(models_dir, f'{model_name}_xscaler.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler {model_name} not found at {scaler_path}")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Log model type for debugging
        model_type = str(type(model))
        logger.info(f"Model {model_name} loaded successfully. Type: {model_type}")
        logger.info(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model {model_name}")

def process_predictions(df: pd.DataFrame, predictions: Any, prediction_column: str = "Prediction") -> pd.DataFrame:
    """
    Processes predictions and adds them to the original DataFrame.
    
    Args:
        df: Original DataFrame with input data
        predictions: Model predictions array
        prediction_column: Name of the prediction column
        
    Returns:
        DataFrame with predictions added
    """
    df_predictions = df.copy()
    df_predictions[prediction_column] = predictions
    
    # Reorder columns to put prediction at the beginning
    columns = df_predictions.columns.tolist()
    columns.remove(prediction_column)
    columns.insert(0, prediction_column)
    
    return df_predictions[columns]

def create_csv_response(df: pd.DataFrame, filename: str) -> StreamingResponse:
    """
    Creates a CSV response from a DataFrame.
    
    Args:
        df: DataFrame to convert to CSV
        filename: CSV filename
        
    Returns:
        StreamingResponse with the CSV
    """
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return StreamingResponse(
        csv_buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

async def process_inference_request(file: UploadFile, model_name: str, output_filename: str) -> StreamingResponse:
    """
    Processes a generic inference request.
    
    Args:
        file: Uploaded CSV file
        model_name: Model name to use ('rs','nee_all','nee_cro',etc)
        output_filename: Output filename
        
    Returns:
        StreamingResponse with results
    """
    try:
        # Read and validate file
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Load data
        df = pd.read_csv(StringIO(content.decode("utf-8")))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file contains no data")
        
        # Load model and scaler
        model, scaler = load_model_and_scaler(model_name)
        scaled_data = scaler.transform(df.values)
        
        # Try multiple prediction approaches
        predictions = None
        prediction_methods = []
        
        # Try regular numpy array prediction first
        try:
            predictions = model.predict(scaled_data)
            prediction_methods.append("numpy_array")
            logger.info(f"Successfully predicted using numpy array for {model_name}")
        except Exception as e1:
            logger.warning(f"Numpy array prediction failed: {str(e1)}")
            
            # Try XGBoost DMatrix
            try:
                import xgboost as xgb
                dmatrix = xgb.DMatrix(scaled_data)
                predictions = model.predict(dmatrix)
                prediction_methods.append("xgboost_dmatrix")
                logger.info(f"Successfully predicted using XGBoost DMatrix for {model_name}")
            except Exception as e2:
                logger.warning(f"XGBoost DMatrix prediction failed: {str(e2)}")
                
                # Try XGBoost with different prediction methods
                try:
                    import xgboost as xgb
                    dmatrix = xgb.DMatrix(scaled_data)
                    
                    # Try different XGBoost prediction methods
                    if hasattr(model, 'predict_proba'):
                        predictions = model.predict_proba(dmatrix)
                        prediction_methods.append("xgboost_predict_proba")
                    elif hasattr(model, 'predict'):
                        predictions = model.predict(dmatrix)
                        prediction_methods.append("xgboost_predict")
                    else:
                        raise Exception("No suitable prediction method found")
                    
                    logger.info(f"Successfully predicted using XGBoost alternative method for {model_name}")
                except Exception as e3:
                    logger.error(f"All XGBoost prediction methods failed: {str(e3)}")
                    
                    # Try with pandas DataFrame
                    try:
                        df_scaled = pd.DataFrame(scaled_data, columns=df.columns)
                        predictions = model.predict(df_scaled)
                        prediction_methods.append("pandas_dataframe")
                        logger.info(f"Successfully predicted using pandas DataFrame for {model_name}")
                    except Exception as e4:
                        logger.error(f"Pandas DataFrame prediction failed: {str(e4)}")
                        raise HTTPException(status_code=500, detail=f"All prediction methods failed. Last error: {str(e4)}")
        
        if predictions is None:
            raise HTTPException(status_code=500, detail="No prediction method succeeded")
        
        logger.info(f"Used prediction methods: {prediction_methods}")
        
        # Process results
        df_predictions = process_predictions(df, predictions)
        
        # Create CSV response
        return create_csv_response(df_predictions, output_filename)
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Error parsing CSV file")
    except Exception as e:
        logger.error(f"Error in inference processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
def read_root():
    """Root endpoint that returns basic API information."""
    return {
        "message": "ML-based Soil Respiration API",
        "version": "1.0.0",
        "endpoints": {
            "rs": "/inference/rs",
            "nee": "/inference/nee/all",
            "health": "/health"
        },
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    """API health check endpoint."""
    return {"status": "healthy", "message": "API is running correctly"}

# Removed public logs endpoint for security reasons
# Logs should only be accessed directly from the server filesystem
# or through a properly authenticated admin interface

@app.post("/inference/rs")
async def rs_inference(file: UploadFile = File(...)):
    """
    Endpoint for soil respiration (RS) predictions.
    
    Args:
        file: CSV file with input data
        
    Returns:
        CSV file with RS predictions. Unit: g C m-2 day-1
    """
    return await process_inference_request(file, "rs", "rs_inference_results.csv")

@app.post("/inference/nee/all")
async def nee_inference(file: UploadFile = File(...)):
    """
    Endpoint for net ecosystem exchange (NEE) predictions(for all landcover types).
    
    Args:
        file: CSV file with input data
        
    Returns:
        CSV file with NEE predictions. Unit: μmol CO2 m-2 s-1
    """
    return await process_inference_request(file, "nee_all", "nee_inference_results.csv")


@app.post("/inference/nee/cropland")
async def nee_inference(file: UploadFile = File(...)):
    """
    Endpoint for net ecosystem exchange (NEE) predictions(cropland).
    
    Args:
        file: CSV file with input data
        
    Returns:
        CSV file with NEE predictions. Unit: μmol CO2 m-2 s-1
    """
    return await process_inference_request(file, "nee_cro", "nee_inference_results.csv")