"""
FastAPI aplikacja do predykcji spalonych kalorii z plików GPX
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json
import logging
import shutil
from datetime import datetime
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Backend bez GUI
import matplotlib.pyplot as plt
import seaborn as sns

from src.predict_from_gpx import CaloriePredictor
from src.gpx_parser import GPXParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cycling Calories Predictor", version="1.0")

# Tworzenie katalogów
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Inicjalizacja predyktora
predictor = CaloriePredictor(model_path="data/ml_models/random_forest.pkl")

try:
    predictor.load_model()
    logger.info("Model załadowany pomyślnie")
except Exception as e:
    logger.error(f"Błąd ładowania modelu: {e}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Strona główna z formularzem uploadu"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_gpx(
    file: UploadFile = File(...),
    weight: float = Form(75)
):
    """
    Upload pliku GPX i uruchomienie predykcji

    Args:
        file: Plik GPX
        weight: Waga sportowca w kg

    Returns:
        JSON z wynikami predykcji i danymi do wizualizacji
    """
    try:
        # Sprawdź rozszerzenie pliku
        if not file.filename.endswith('.gpx'):
            raise HTTPException(status_code=400, detail="Plik musi być w formacie GPX")

        # Zapisz plik
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Plik zapisany: {file_path}")

        # Uruchom predykcję
        result = predictor.predict_calories(str(file_path), athlete_weight=weight)

        # Przygotuj dane do wizualizacji
        df = result['points_df']
        route_summary = result['route_summary']

        # Dane dla mapy Leaflet (lat, lon, speed, elevation)
        route_points = []
        for _, row in df.iterrows():
            point = {
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'elevation': float(row.get('elevation', 0)),
                'speed': float(row.get('speed_kmh', 0)),
                'distance': float(row.get('cumulative_distance', 0)) / 1000  # km
            }
            route_points.append(point)

        # Generuj wykresy
        charts = generate_charts(result)

        # Przygotuj odpowiedź
        response_data = {
            'success': True,
            'predicted_calories': float(result['predicted_calories']),
            'route_summary': {
                'distance_km': float(route_summary['total_distance_km']),
                'elevation_gain': float(route_summary['total_elevation_gain']),
                'elevation_loss': float(route_summary['total_elevation_loss']),
                'avg_speed': float(route_summary['avg_speed_kmh']),
                'max_speed': float(route_summary['max_speed_kmh']),
                'avg_grade': float(route_summary['avg_grade']),
                'max_grade': float(route_summary['max_grade']),
                'min_grade': float(route_summary['min_grade']),
                'total_time_min': float(route_summary['total_time_min']),
                'elevation_per_km': float(route_summary['elevation_per_km']),
            },
            'route_points': route_points,
            'charts': charts,
            'athlete_weight': float(weight),
            'filename': file.filename
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Błąd podczas przetwarzania: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Błąd przetwarzania pliku: {str(e)}")


def generate_charts(result: dict) -> dict:
    """
    Generuje wykresy jako base64 dla frontendu

    Args:
        result: Wynik predykcji

    Returns:
        Słownik z wykresami w formacie base64
    """
    df = result['points_df']
    charts = {}

    sns.set_style("whitegrid")

    # 1. Profil wysokościowy
    if 'elevation' in df.columns and len(df) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        distance_km = df['cumulative_distance'] / 1000
        ax.plot(distance_km, df['elevation'], linewidth=2, color='saddlebrown')
        ax.fill_between(distance_km, df['elevation'], alpha=0.3, color='tan')
        ax.set_xlabel('Dystans [km]', fontsize=11)
        ax.set_ylabel('Wysokość [m n.p.m.]', fontsize=11)
        ax.set_title('Profil wysokościowy trasy', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        charts['elevation_profile'] = fig_to_base64(fig)
        plt.close(fig)

    # 2. Wykres prędkości
    if 'speed_kmh' in df.columns and len(df[df['speed_kmh'] > 0]) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        df_speed = df[df['speed_kmh'] > 0].copy()
        distance_km = df_speed['cumulative_distance'] / 1000
        ax.plot(distance_km, df_speed['speed_kmh'], linewidth=2, color='blue', alpha=0.7)
        ax.axhline(y=result['route_summary']['avg_speed_kmh'], color='red',
                  linestyle='--', linewidth=2, label=f"Średnia: {result['route_summary']['avg_speed_kmh']:.1f} km/h")
        ax.set_xlabel('Dystans [km]', fontsize=11)
        ax.set_ylabel('Prędkość [km/h]', fontsize=11)
        ax.set_title('Prędkość w funkcji dystansu', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        charts['speed_profile'] = fig_to_base64(fig)
        plt.close(fig)

    # 3. Histogram nachylenia
    if 'grade' in df.columns and len(df[df['grade'] != 0]) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        df_grade = df[df['grade'] != 0].copy()
        ax.hist(df_grade['grade'], bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.axvline(x=result['route_summary']['avg_grade'], color='red',
                  linestyle='--', linewidth=2, label=f"Średnia: {result['route_summary']['avg_grade']:.2f}%")
        ax.set_xlabel('Nachylenie [%]', fontsize=11)
        ax.set_ylabel('Liczba punktów', fontsize=11)
        ax.set_title('Rozkład nachylenia', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        charts['grade_histogram'] = fig_to_base64(fig)
        plt.close(fig)

    return charts


def fig_to_base64(fig) -> str:
    """
    Konwertuje wykres matplotlib do base64

    Args:
        fig: Figura matplotlib

    Returns:
        String base64 z obrazem PNG
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return f"data:image/png;base64,{image_base64}"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
