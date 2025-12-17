"""
Moduł do przetwarzania danych z Strava
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StravaDataProcessor:
    """Klasa do przetwarzania surowych danych ze Stravy"""

    def __init__(self, input_file: str = "data/raw/strava_detailed_activities.json"):
        """
        Inicjalizacja procesora danych

        Args:
            input_file: Ścieżka do pliku z surowymi danymi
        """
        self.input_file = input_file
        self.activities = None
        self.processed_df = None

    def load_raw_data(self) -> List[Dict]:
        """Wczytuje surowe dane z pliku JSON"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.activities = json.load(f)

        logger.info(f"Wczytano {len(self.activities)} aktywności z {self.input_file}")
        return self.activities

    def extract_basic_features(self, activity: Dict) -> Dict:
        """
        Ekstrahuje podstawowe cechy z aktywności

        Args:
            activity: Słownik z danymi aktywności

        Returns:
            Słownik z wyekstrahowanymi cechami
        """
        features = {
            'activity_id': activity.get('id'),
            'name': activity.get('name'),
            'date': activity.get('start_date'),
            'distance': activity.get('distance', 0),  # metry
            'moving_time': activity.get('moving_time', 0),  # sekundy
            'elapsed_time': activity.get('elapsed_time', 0),  # sekundy
            'total_elevation_gain': activity.get('total_elevation_gain', 0),  # metry
            'calories': activity.get('calories', 0),  # kcal (zmienna docelowa)
            'average_speed': activity.get('average_speed', 0),  # m/s
            'max_speed': activity.get('max_speed', 0),  # m/s
            'average_heartrate': activity.get('average_heartrate', None),
            'max_heartrate': activity.get('max_heartrate', None),
            'average_cadence': activity.get('average_cadence', None),
            'average_watts': activity.get('average_watts', None),
            'max_watts': activity.get('max_watts', None),
            'kilojoules': activity.get('kilojoules', None),
        }

        # Dodaj dane o sportowcu jeśli dostępne
        athlete = activity.get('athlete', {})
        features['athlete_weight'] = athlete.get('weight', None)  # kg

        return features

    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Oblicza pochodne cechy z danych podstawowych

        Args:
            df: DataFrame z podstawowymi cechami

        Returns:
            DataFrame z dodatkowymi cechami
        """
        # Konwersje jednostek
        df['distance_km'] = df['distance'] / 1000
        df['moving_time_min'] = df['moving_time'] / 60
        df['average_speed_kmh'] = df['average_speed'] * 3.6
        df['max_speed_kmh'] = df['max_speed'] * 3.6

        # Nachylenie średnie
        df['avg_grade'] = np.where(
            df['distance'] > 0,
            (df['total_elevation_gain'] / df['distance']) * 100,
            0
        )

        # Intensywność (kalorie na minutę)
        df['calorie_burn_rate'] = np.where(
            df['moving_time_min'] > 0,
            df['calories'] / df['moving_time_min'],
            0
        )

        # Kalorie na km
        df['calories_per_km'] = np.where(
            df['distance_km'] > 0,
            df['calories'] / df['distance_km'],
            0
        )

        # Poziom wysiłku (elevation gain na km)
        df['elevation_per_km'] = np.where(
            df['distance_km'] > 0,
            df['total_elevation_gain'] / df['distance_km'],
            0
        )

        # Współczynnik efektywności (dystans na kcal)
        df['efficiency'] = np.where(
            df['calories'] > 0,
            df['distance_km'] / df['calories'],
            0
        )

        # Czas stania (elapsed - moving)
        df['stopped_time_min'] = (df['elapsed_time'] - df['moving_time']) / 60

        # Procent czasu w ruchu
        df['moving_ratio'] = np.where(
            df['elapsed_time'] > 0,
            df['moving_time'] / df['elapsed_time'],
            0
        )

        # Kategorie prędkości
        df['speed_category'] = pd.cut(
            df['average_speed_kmh'],
            bins=[0, 15, 20, 25, 30, 100],
            labels=['bardzo_wolna', 'wolna', 'srednia', 'szybka', 'bardzo_szybka']
        )

        # Kategorie nachylenia
        df['grade_category'] = pd.cut(
            df['avg_grade'],
            bins=[-100, -2, -0.5, 0.5, 2, 100],
            labels=['zjazd_stromy', 'zjazd', 'plaski', 'podjazd', 'podjazd_stromy']
        )

        # Kategorie dystansu
        df['distance_category'] = pd.cut(
            df['distance_km'],
            bins=[0, 20, 50, 80, 120, 1000],
            labels=['krotki', 'sredni', 'dlugi', 'bardzo_dlugi', 'ekstremalny']
        )

        return df

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ekstrahuje cechy czasowe z daty aktywności

        Args:
            df: DataFrame z kolumną 'date'

        Returns:
            DataFrame z cechami czasowymi
        """
        df['date'] = pd.to_datetime(df['date'])

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['hour'] = df['date'].dt.hour
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Sezon
        df['season'] = df['month'].map({
            12: 'zima', 1: 'zima', 2: 'zima',
            3: 'wiosna', 4: 'wiosna', 5: 'wiosna',
            6: 'lato', 7: 'lato', 8: 'lato',
            9: 'jesien', 10: 'jesien', 11: 'jesien'
        })

        # Pora dnia
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['noc', 'rano', 'poludnie', 'wieczor'],
            include_lowest=True
        )

        return df

    def process_streams_data(self, activity: Dict) -> Dict:
        """
        Przetwarza dane streams (szczegółowe dane z czujników)

        Args:
            activity: Słownik z danymi aktywności zawierający 'streams'

        Returns:
            Słownik ze statystykami ze streams
        """
        streams = activity.get('streams', {})
        stream_features = {}

        # Analiza nachylenia
        if 'grade_smooth' in streams:
            grades = streams['grade_smooth'].get('data', [])
            if grades:
                stream_features['grade_std'] = np.std(grades)
                stream_features['grade_max'] = np.max(grades)
                stream_features['grade_min'] = np.min(grades)
                stream_features['positive_grade_pct'] = (np.array(grades) > 0).mean() * 100

        # Analiza prędkości
        if 'velocity_smooth' in streams:
            velocities = streams['velocity_smooth'].get('data', [])
            if velocities:
                velocities_kmh = np.array(velocities) * 3.6
                stream_features['velocity_std'] = np.std(velocities_kmh)
                stream_features['velocity_median'] = np.median(velocities_kmh)

        # Analiza tętna
        if 'heartrate' in streams:
            hr = streams['heartrate'].get('data', [])
            if hr:
                stream_features['hr_std'] = np.std(hr)
                stream_features['hr_median'] = np.median(hr)
                stream_features['hr_range'] = np.max(hr) - np.min(hr)

        # Analiza mocy
        if 'watts' in streams:
            watts = streams['watts'].get('data', [])
            if watts:
                stream_features['watts_std'] = np.std(watts)
                stream_features['watts_median'] = np.median(watts)
                stream_features['watts_max'] = np.max(watts)

        # Analiza wysokości
        if 'altitude' in streams:
            altitude = streams['altitude'].get('data', [])
            if altitude:
                stream_features['altitude_min'] = np.min(altitude)
                stream_features['altitude_max'] = np.max(altitude)
                stream_features['altitude_range'] = np.max(altitude) - np.min(altitude)

        return stream_features

    def process_all_activities(self) -> pd.DataFrame:
        """
        Przetwarza wszystkie aktywności i zwraca DataFrame

        Returns:
            DataFrame z przetworzonymi danymi
        """
        if self.activities is None:
            self.load_raw_data()

        logger.info("Przetwarzam aktywności...")

        processed_activities = []

        for activity in self.activities:
            # Ekstrahuj podstawowe cechy
            features = self.extract_basic_features(activity)

            # Dodaj cechy ze streams
            stream_features = self.process_streams_data(activity)
            features.update(stream_features)

            processed_activities.append(features)

        # Utwórz DataFrame
        df = pd.DataFrame(processed_activities)

        # Oblicz pochodne cechy
        df = self.calculate_derived_features(df)

        # Ekstrahuj cechy czasowe
        df = self.extract_time_features(df)

        logger.info(f"Przetworzono {len(df)} aktywności")
        logger.info(f"Liczba cech: {len(df.columns)}")

        self.processed_df = df
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Czyści dane - usuwa outliers, uzupełnia braki

        Args:
            df: DataFrame do wyczyszczenia

        Returns:
            Wyczyszczony DataFrame
        """
        initial_count = len(df)

        # Usuń aktywności bez spalonych kalorii (zmienna docelowa)
        df = df[df['calories'] > 0]

        # Usuń aktywności bardzo krótkie (< 5 minut)
        df = df[df['moving_time_min'] >= 5]

        # Usuń aktywności bardzo krótkie dystansowo (< 2 km)
        df = df[df['distance_km'] >= 2]

        # Usuń nierealistyczne prędkości (> 70 km/h)
        df = df[df['average_speed_kmh'] <= 70]

        # Usuń outliers w kaloriach (używając IQR)
        Q1 = df['calories'].quantile(0.05)
        Q3 = df['calories'].quantile(0.95)
        IQR = Q3 - Q1
        df = df[(df['calories'] >= Q1 - 1.5 * IQR) & (df['calories'] <= Q3 + 1.5 * IQR)]

        logger.info(f"Usunięto {initial_count - len(df)} aktywności podczas czyszczenia")
        logger.info(f"Pozostało {len(df)} aktywności")

        return df

    def save_processed_data(self, df: pd.DataFrame,
                           filename: str = "data/processed/processed_activities.csv"):
        """
        Zapisuje przetworzone dane do pliku CSV

        Args:
            df: DataFrame do zapisania
            filename: Nazwa pliku wyjściowego
        """
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Zapisano przetworzone dane do {filename}")

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Zwraca podsumowanie danych

        Args:
            df: DataFrame do podsumowania

        Returns:
            Słownik z statystykami
        """
        summary = {
            'total_activities': len(df),
            'total_distance_km': df['distance_km'].sum(),
            'total_calories': df['calories'].sum(),
            'total_time_hours': df['moving_time_min'].sum() / 60,
            'total_elevation_m': df['total_elevation_gain'].sum(),
            'avg_distance_km': df['distance_km'].mean(),
            'avg_calories': df['calories'].mean(),
            'avg_speed_kmh': df['average_speed_kmh'].mean(),
            'date_range': f"{df['date'].min()} do {df['date'].max()}"
        }

        logger.info("\n" + "="*50)
        logger.info("PODSUMOWANIE DANYCH")
        logger.info("="*50)
        for key, value in summary.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.2f}")
            else:
                logger.info(f"{key}: {value}")
        logger.info("="*50 + "\n")

        return summary


def main():
    """Główna funkcja przetwarzania danych"""
    try:
        processor = StravaDataProcessor()

        # Wczytaj surowe dane
        processor.load_raw_data()

        # Przetwórz wszystkie aktywności
        df = processor.process_all_activities()

        # Wyczyść dane
        df = processor.clean_data(df)

        # Zapisz przetworzone dane
        processor.save_processed_data(df)

        # Pokaż podsumowanie
        processor.get_data_summary(df)

        logger.info("✓ Przetwarzanie danych zakończone!")

    except FileNotFoundError:
        logger.error("Nie znaleziono pliku z surowymi danymi. Uruchom najpierw strava_client.py")
    except Exception as e:
        logger.error(f"Wystąpił błąd: {e}")


if __name__ == "__main__":
    main()
