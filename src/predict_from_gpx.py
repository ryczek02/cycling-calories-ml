"""
Moduł do predykcji spalonych kalorii i prędkości z pliku GPX
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging

from src.gpx_parser import GPXParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


class CaloriePredictor:
    """Klasa do predykcji spalonych kalorii z pliku GPX"""

    def __init__(self, model_path: str = "data/ml_models/random_forest.pkl"):
        """
        Inicjalizacja predyktora

        Args:
            model_path: Ścieżka do zapisanego modelu
        """
        self.model_path = Path(model_path)
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.output_dir = Path("data/predictions")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Wczytuje wytrenowany model"""
        logger.info(f"Wczytywanie modelu z: {self.model_path}")

        self.model_data = joblib.load(self.model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_names = self.model_data['feature_names']

        logger.info(f"✓ Model wczytany")
        logger.info(f"  Metryki: MAE={self.model_data['metrics']['test_mae']:.2f} kcal, "
                   f"R²={self.model_data['metrics']['test_r2']:.4f}")

    def prepare_features_from_route(self, route_summary: dict) -> pd.DataFrame:
        """
        Przygotowuje cechy z podsumowania trasy do predykcji

        Args:
            route_summary: Słownik z cechami trasy

        Returns:
            DataFrame z cechami gotowymi do predykcji
        """
        # Podstawowe cechy
        features = {
            'distance_km': route_summary.get('total_distance_km', 0),
            'moving_time_min': route_summary.get('total_time_min', 0),
            'total_elevation_gain': route_summary.get('total_elevation_gain', 0),
            'average_speed_kmh': route_summary.get('avg_speed_kmh', 20),
            'max_speed_kmh': route_summary.get('max_speed_kmh', 40),
            'avg_grade': route_summary.get('avg_grade', 0),
            'elevation_per_km': route_summary.get('elevation_per_km', 0),
        }

        # Dodatkowe cechy (domyślne wartości jeśli brak)
        features['stopped_time_min'] = 0
        features['moving_ratio'] = 1.0

        # Cechy czasowe (zakładamy dzisiejszą datę, południe)
        from datetime import datetime
        now = datetime.now()
        features['month'] = now.month
        features['day_of_week'] = now.weekday()
        features['hour'] = 12  # Domyślnie południe
        features['is_weekend'] = 1 if now.weekday() >= 5 else 0

        # Wypełnij brakujące cechy zerami
        df = pd.DataFrame([features])

        # Dodaj brakujące kolumny jeśli potrzebne
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        # Uporządkuj kolumny zgodnie z modelem
        df = df[self.feature_names]

        return df

    def predict_calories(self, gpx_file: str, athlete_weight: float = 75) -> dict:
        """
        Przewiduje spalone kalorie dla trasy z pliku GPX

        Args:
            gpx_file: Ścieżka do pliku GPX
            athlete_weight: Waga sportowca w kg

        Returns:
            Słownik z predykcją i szczegółami trasy
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"PREDYKCJA DLA: {gpx_file}")
        logger.info(f"{'='*70}\n")

        # Parsuj GPX
        parser = GPXParser(gpx_file)
        df = parser.parse()
        df = parser.calculate_features(df, athlete_weight=athlete_weight)

        # Ekstrahuj podsumowanie trasy
        route_summary = parser.extract_route_summary(df, athlete_weight=athlete_weight)
        route_summary['athlete_weight'] = athlete_weight

        # Przygotuj cechy do predykcji
        X = self.prepare_features_from_route(route_summary)

        # Skaluj jeśli model tego wymaga
        if self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns
            )

        # Predykcja
        predicted_calories = self.model.predict(X)[0]

        logger.info(f"\n{'='*70}")
        logger.info(f"WYNIKI PREDYKCJI")
        logger.info(f"{'='*70}")
        logger.info(f"Dystans: {route_summary['total_distance_km']:.2f} km")
        logger.info(f"Przewyższenie: {route_summary['total_elevation_gain']:.0f} m")
        logger.info(f"Średnia prędkość: {route_summary['avg_speed_kmh']:.1f} km/h")
        logger.info(f"Czas trwania: {route_summary['total_time_min']:.0f} min")
        logger.info(f"\n🔥 PRZEWIDYWANE SPALENIE: {predicted_calories:.0f} kcal")
        logger.info(f"{'='*70}\n")

        result = {
            'gpx_file': gpx_file,
            'predicted_calories': predicted_calories,
            'route_summary': route_summary,
            'points_df': df,
            'athlete_weight': athlete_weight,
        }

        return result

    def visualize_route_with_speed(self, result: dict, output_file: str = None):
        """
        Wizualizuje trasę z kolorami reprezentującymi prędkość

        Args:
            result: Wynik predykcji z predict_calories()
            output_file: Ścieżka do zapisu wykresu
        """
        logger.info("Tworzę wizualizację trasy z prędkościami...")

        df = result['points_df']

        # Filtruj punkty z prędkością
        df_with_speed = df[df['speed_kmh'] > 0].copy()

        if len(df_with_speed) == 0:
            logger.warning("Brak danych o prędkości w pliku GPX")
            df_with_speed = df.copy()
            df_with_speed['speed_kmh'] = result['route_summary']['avg_speed_kmh']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Mapa z prędkościami
        ax = axes[0, 0]
        scatter = ax.scatter(df_with_speed['lon'], df_with_speed['lat'],
                           c=df_with_speed['speed_kmh'], cmap='RdYlGn',
                           s=20, alpha=0.7)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Prędkość [km/h]', fontsize=11)
        ax.set_xlabel('Długość geograficzna', fontsize=11)
        ax.set_ylabel('Szerokość geograficzna', fontsize=11)
        ax.set_title('Mapa trasy z prędkościami', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Profil wysokościowy
        ax = axes[0, 1]
        if 'elevation' in df.columns:
            distance_km = df['cumulative_distance'] / 1000
            ax.plot(distance_km, df['elevation'], linewidth=2, color='saddlebrown')
            ax.fill_between(distance_km, df['elevation'], alpha=0.3, color='tan')
            ax.set_xlabel('Dystans [km]', fontsize=11)
            ax.set_ylabel('Wysokość [m n.p.m.]', fontsize=11)
            ax.set_title('Profil wysokościowy trasy', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # 3. Prędkość w funkcji dystansu
        ax = axes[1, 0]
        distance_km = df_with_speed['cumulative_distance'] / 1000
        # ax.plot(distance_km, df_with_speed['speed_smooth'], linewidth=2, color='blue', label='Prędkość')
        ax.axhline(y=result['route_summary']['avg_speed_kmh'], color='red',
                  linestyle='--', label=f"Średnia: {result['route_summary']['avg_speed_kmh']:.1f} km/h")
        ax.set_xlabel('Dystans [km]', fontsize=11)
        ax.set_ylabel('Prędkość [km/h]', fontsize=11)
        ax.set_title('Prędkość w funkcji dystansu', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Podsumowanie
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = f"""
PODSUMOWANIE TRASY

Dystans: {result['route_summary']['total_distance_km']:.2f} km
Przewyższenie: {result['route_summary']['total_elevation_gain']:.0f} m
Czas: {result['route_summary']['total_time_min']:.0f} min

Prędkość średnia: {result['route_summary']['avg_speed_kmh']:.1f} km/h
Prędkość maksymalna: {result['route_summary']['max_speed_kmh']:.1f} km/h

Nachylenie średnie: {result['route_summary']['avg_grade']:.2f}%
Nachylenie max: {result['route_summary']['max_grade']:.2f}%

Waga sportowca: {result['athlete_weight']:.0f} kg

═══════════════════════════════════

🔥 PRZEWIDYWANE SPALENIE:
   {result['predicted_calories']:.0f} kcal

═══════════════════════════════════
        """

        ax.text(0.1, 0.5, summary_text, fontsize=13, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Analiza trasy GPX: {Path(result["gpx_file"]).name}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if output_file is None:
            gpx_name = Path(result['gpx_file']).stem
            output_file = self.output_dir / f"prediction_{gpx_name}.png"

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Wizualizacja zapisana: {output_file}")

    def create_detailed_report(self, result: dict, output_file: str = None):
        """
        Tworzy szczegółowy raport tekstowy z predykcji

        Args:
            result: Wynik predykcji
            output_file: Ścieżka do zapisu raportu
        """
        if output_file is None:
            gpx_name = Path(result['gpx_file']).stem
            output_file = self.output_dir / f"report_{gpx_name}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RAPORT PREDYKCJI SPALONYCH KALORII\n")
            f.write("="*70 + "\n\n")

            f.write(f"Plik GPX: {result['gpx_file']}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Waga sportowca: {result['athlete_weight']:.0f} kg\n\n")

            f.write("CHARAKTERYSTYKA TRASY\n")
            f.write("-"*70 + "\n")
            r = result['route_summary']
            f.write(f"Dystans całkowity:        {r['total_distance_km']:.2f} km\n")
            f.write(f"Przewyższenie (↑):        {r['total_elevation_gain']:.0f} m\n")
            f.write(f"Zjazd (↓):                {r['total_elevation_loss']:.0f} m\n")
            f.write(f"Czas trwania:             {r['total_time_min']:.0f} min ({r['total_time_min']/60:.1f} h)\n\n")

            f.write("PRĘDKOŚĆ\n")
            f.write("-"*70 + "\n")
            f.write(f"Średnia prędkość:         {r['avg_speed_kmh']:.1f} km/h\n")
            f.write(f"Maksymalna prędkość:      {r['max_speed_kmh']:.1f} km/h\n\n")

            f.write("NACHYLENIE\n")
            f.write("-"*70 + "\n")
            f.write(f"Średnie nachylenie:       {r['avg_grade']:.2f}%\n")
            f.write(f"Maksymalne nachylenie:    {r['max_grade']:.2f}%\n")
            f.write(f"Minimalne nachylenie:     {r['min_grade']:.2f}%\n")
            f.write(f"Przewyższenie na km:      {r['elevation_per_km']:.1f} m/km\n\n")

            f.write("="*70 + "\n")
            f.write("WYNIK PREDYKCJI\n")
            f.write("="*70 + "\n")
            f.write(f"🔥 Przewidywane spalenie: {result['predicted_calories']:.0f} kcal\n")
            f.write("="*70 + "\n\n")

            f.write("DOKŁADNOŚĆ MODELU\n")
            f.write("-"*70 + "\n")
            f.write(f"MAE (błąd średni):        {self.model_data['metrics']['test_mae']:.2f} kcal\n")
            f.write(f"RMSE:                     {self.model_data['metrics']['test_rmse']:.2f} kcal\n")
            f.write(f"R² (dopasowanie):         {self.model_data['metrics']['test_r2']:.4f}\n")
            f.write(f"MAPE (błąd %):            {self.model_data['metrics']['test_mape']:.2f}%\n\n")

        logger.info(f"✓ Raport zapisany: {output_file}")


def main():
    """Główna funkcja do predykcji z wiersza poleceń"""
    import argparse

    parser = argparse.ArgumentParser(description="Predykcja spalonych kalorii z pliku GPX")
    parser.add_argument('gpx_file', help='Ścieżka do pliku GPX')
    parser.add_argument('--weight', type=float, default=75, help='Waga sportowca w kg (domyślnie 75)')
    parser.add_argument('--model', default='data/ml_models/random_forest.pkl',
                       help='Ścieżka do modelu (domyślnie Random Forest)')

    args = parser.parse_args()

    try:
        # Utwórz predyktor
        predictor = CaloriePredictor(model_path=args.model)
        predictor.load_model()

        # Przewiduj kalorie
        result = predictor.predict_calories(args.gpx_file, athlete_weight=args.weight)

        # Wizualizuj trasę
        # predictor.visualize_route_with_speed(result)

        # Utwórz raport
        predictor.create_detailed_report(result)

        logger.info(f"\n✓✓✓ PREDYKCJA ZAKOŃCZONA! ✓✓✓")
        logger.info(f"Pliki zapisane w: {predictor.output_dir}")

    except FileNotFoundError as e:
        logger.error(f"Nie znaleziono pliku: {e}")
    except Exception as e:
        logger.error(f"Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
