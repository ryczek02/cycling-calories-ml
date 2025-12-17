"""
Moduł do parsowania i analizy plików GPX
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GPXParser:
    """Parser plików GPX z ekstrakcją cech do predykcji"""

    def __init__(self, gpx_file: str):
        """
        Inicjalizacja parsera GPX

        Args:
            gpx_file: Ścieżka do pliku GPX
        """
        self.gpx_file = gpx_file
        self.points = []
        self.namespace = {'gpx': 'http://www.topografix.com/GPX/1/1'}

    def parse(self) -> pd.DataFrame:
        """
        Parsuje plik GPX i zwraca DataFrame z punktami

        Returns:
            DataFrame z punktami GPS i obliczonymi cechami
        """
        logger.info(f"Parsowanie pliku GPX: {self.gpx_file}")

        tree = ET.parse(self.gpx_file)
        root = tree.getroot()

        # Znajdź wszystkie punkty trasy (trkpt)
        points = []

        for trkpt in root.findall('.//gpx:trkpt', self.namespace):
            point = {
                'lat': float(trkpt.get('lat')),
                'lon': float(trkpt.get('lon')),
            }

            # Wysokość
            ele = trkpt.find('gpx:ele', self.namespace)
            if ele is not None:
                point['elevation'] = float(ele.text)

            # Czas
            time = trkpt.find('gpx:time', self.namespace)
            if time is not None:
                point['time'] = datetime.fromisoformat(time.text.replace('Z', '+00:00'))

            points.append(point)

        if not points:
            # Spróbuj bez namespace
            for trkpt in root.findall('.//trkpt'):
                point = {
                    'lat': float(trkpt.get('lat')),
                    'lon': float(trkpt.get('lon')),
                }

                ele = trkpt.find('ele')
                if ele is not None:
                    point['elevation'] = float(ele.text)

                time_elem = trkpt.find('time')
                if time_elem is not None:
                    point['time'] = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))

                points.append(point)

        self.points = points
        df = pd.DataFrame(points)

        logger.info(f"Znaleziono {len(df)} punktów GPS")

        return df

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Oblicza dystans między dwoma punktami GPS (wzór Haversine)

        Args:
            lat1, lon1: Współrzędne punktu 1
            lat2, lon2: Współrzędne punktu 2

        Returns:
            Dystans w metrach
        """
        R = 6371000  # Promień Ziemi w metrach

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        distance = R * c
        return distance

    def calculate_features(self, df: pd.DataFrame, athlete_weight: float = 75) -> pd.DataFrame:
        """
        Oblicza cechy z danych GPS

        Args:
            df: DataFrame z punktami GPS
            athlete_weight: Waga sportowca w kg (domyślnie 75)

        Returns:
            DataFrame z obliczonymi cechami
        """
        logger.info("Obliczanie cech z danych GPS...")

        # Dystans między punktami
        distances = []
        for i in range(len(df)):
            if i == 0:
                distances.append(0)
            else:
                dist = self.calculate_distance(
                    df.iloc[i-1]['lat'], df.iloc[i-1]['lon'],
                    df.iloc[i]['lat'], df.iloc[i]['lon']
                )
                distances.append(dist)

        df['distance_segment'] = distances
        df['cumulative_distance'] = df['distance_segment'].cumsum()

        # Różnica wysokości
        if 'elevation' in df.columns:
            df['elevation_diff'] = df['elevation'].diff().fillna(0)

            # Nachylenie (%)
            df['grade'] = np.where(
                df['distance_segment'] > 0,
                (df['elevation_diff'] / df['distance_segment']) * 100,
                0
            )

            # Przewyższenie pozytywne i negatywne
            df['elevation_gain'] = df['elevation_diff'].apply(lambda x: max(0, x))
            df['elevation_loss'] = df['elevation_diff'].apply(lambda x: min(0, x))

        # Różnica czasu i prędkość
        if 'time' in df.columns:
            df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)

            # Prędkość (m/s i km/h)
            df['speed_ms'] = np.where(
                df['time_diff'] > 0,
                df['distance_segment'] / df['time_diff'],
                0
            )
            df['speed_kmh'] = df['speed_ms'] * 3.6

        # Wygładzone wartości (rolling average)
        window = 5
        if len(df) >= window:
            df['speed_smooth'] = df['speed_kmh'].rolling(window=window, center=True).mean()
            if 'grade' in df.columns:
                df['grade_smooth'] = df['grade'].rolling(window=window, center=True).mean()
        else:
            df['speed_smooth'] = df.get('speed_kmh', 0)
            df['grade_smooth'] = df.get('grade', 0)

        # Wypełnij NaN
        df = df.fillna(0)

        logger.info("✓ Obliczono cechy")

        return df

    def extract_segment_features(self, df: pd.DataFrame, segment_length: float = 1000) -> pd.DataFrame:
        """
        Dzieli trasę na segmenty i oblicza agregowane cechy

        Args:
            df: DataFrame z punktami GPS i cechami
            segment_length: Długość segmentu w metrach

        Returns:
            DataFrame z cechami dla każdego segmentu
        """
        logger.info(f"Dzielenie trasy na segmenty po {segment_length}m...")

        # Przypisz segment do każdego punktu
        df['segment'] = (df['cumulative_distance'] // segment_length).astype(int)

        # Agreguj cechy dla każdego segmentu
        segment_features = df.groupby('segment').agg({
            'distance_segment': 'sum',
            'elevation_gain': 'sum',
            'elevation_loss': 'sum',
            'grade': 'mean',
            'grade_smooth': 'mean',
            'speed_kmh': 'mean',
            'speed_smooth': 'mean',
            'time_diff': 'sum',
        }).reset_index()

        segment_features.columns = [
            'segment',
            'distance_m',
            'elevation_gain_m',
            'elevation_loss_m',
            'avg_grade',
            'avg_grade_smooth',
            'avg_speed_kmh',
            'avg_speed_smooth',
            'duration_s'
        ]

        # Oblicz dodatkowe cechy
        segment_features['distance_km'] = segment_features['distance_m'] / 1000
        segment_features['duration_min'] = segment_features['duration_s'] / 60

        logger.info(f"Utworzono {len(segment_features)} segmentów")

        return segment_features

    def extract_route_summary(self, df: pd.DataFrame, athlete_weight: float = 75) -> dict:
        """
        Ekstrahuje podsumowanie całej trasy

        Args:
            df: DataFrame z punktami GPS
            athlete_weight: Waga sportowca w kg

        Returns:
            Słownik z cechami całej trasy
        """
        summary = {
            'total_distance_km': df['cumulative_distance'].max() / 1000,
            'total_elevation_gain': df['elevation_gain'].sum() if 'elevation_gain' in df.columns else 0,
            'total_elevation_loss': abs(df['elevation_loss'].sum()) if 'elevation_loss' in df.columns else 0,
            'avg_grade': df['grade'].mean() if 'grade' in df.columns else 0,
            'max_grade': df['grade'].max() if 'grade' in df.columns else 0,
            'min_grade': df['grade'].min() if 'grade' in df.columns else 0,
            'avg_speed_kmh': df['speed_kmh'].mean() if 'speed_kmh' in df.columns else 20,
            'max_speed_kmh': df['speed_kmh'].max() if 'speed_kmh' in df.columns else 40,
            'total_time_min': df['time_diff'].sum() / 60 if 'time_diff' in df.columns else 0,
            'athlete_weight': athlete_weight,
        }

        # Oblicz odległość na km
        summary['elevation_per_km'] = (
            summary['total_elevation_gain'] / summary['total_distance_km']
            if summary['total_distance_km'] > 0 else 0
        )

        # Estymacja czasu (jeśli brak danych czasowych)
        if summary['total_time_min'] == 0 and summary['avg_speed_kmh'] > 0:
            summary['total_time_min'] = (summary['total_distance_km'] / summary['avg_speed_kmh']) * 60

        # Dodaj cechy używane w ML
        summary['moving_time_min'] = summary['total_time_min']
        summary['distance_km'] = summary['total_distance_km']
        summary['average_speed_kmh'] = summary['avg_speed_kmh']
        summary['max_speed_kmh'] = summary['max_speed_kmh']

        logger.info(f"Podsumowanie trasy:")
        logger.info(f"  Dystans: {summary['total_distance_km']:.2f} km")
        logger.info(f"  Przewyższenie: {summary['total_elevation_gain']:.0f} m")
        logger.info(f"  Średnia prędkość: {summary['avg_speed_kmh']:.1f} km/h")

        return summary


def main():
    """Testowa funkcja parsowania GPX"""
    import sys

    if len(sys.argv) < 2:
        logger.error("Użycie: python gpx_parser.py <plik.gpx>")
        sys.exit(1)

    gpx_file = sys.argv[1]

    parser = GPXParser(gpx_file)
    df = parser.parse()
    df = parser.calculate_features(df)

    summary = parser.extract_route_summary(df)

    print("\n" + "="*60)
    print("PODSUMOWANIE TRASY")
    print("="*60)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
