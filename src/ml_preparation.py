"""
Moduł do przygotowania danych do uczenia maszynowego
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLDataPreparator:
    """Klasa do przygotowania danych do uczenia maszynowego"""

    def __init__(self, data_file: str = "data/processed/processed_activities.csv"):
        """
        Inicjalizacja preparatora danych ML

        Args:
            data_file: Ścieżka do pliku z przetworzonymi danymi
        """
        self.data_file = data_file
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.output_dir = Path("data/ml_ready")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Wczytuje przetworzone dane"""
        self.df = pd.read_csv(self.data_file)
        logger.info(f"Wczytano {len(self.df)} aktywności z {self.data_file}")
        return self.df

    def select_features(self) -> pd.DataFrame:
        """
        Wybiera cechy do modelu ML

        Returns:
            DataFrame z wybranymi cechami
        """
        # Cechy do wykorzystania w modelu
        feature_columns = [
            # Podstawowe cechy treningowe
            'distance_km',
            'moving_time_min',
            'total_elevation_gain',
            'average_speed_kmh',
            'max_speed_kmh',
            'avg_grade',
            'elevation_per_km',
            'stopped_time_min',
            'moving_ratio',

            # Cechy czasowe
            'month',
            'day_of_week',
            'hour',
            'is_weekend',

            # Cechy ze streams (jeśli dostępne)
            'grade_std',
            'grade_max',
            'grade_min',
            'velocity_std',
            'hr_std',
            'altitude_range',
        ]

        # Cechy kategoryczne do zakodowania
        categorical_features = [
            'season',
            'time_of_day',
        ]

        # Filtruj tylko istniejące kolumny
        available_features = [col for col in feature_columns if col in self.df.columns]
        available_categorical = [col for col in categorical_features if col in self.df.columns]

        logger.info(f"Wybrano {len(available_features)} cech numerycznych")
        logger.info(f"Wybrano {len(available_categorical)} cech kategorycznych")

        # Zakoduj cechy kategoryczne
        df_encoded = self.df[available_features].copy()

        for col in available_categorical:
            if col in self.df.columns:
                # One-hot encoding
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)

        # Uzupełnij braki wartości medianą
        df_encoded = df_encoded.fillna(df_encoded.median())

        self.feature_names = df_encoded.columns.tolist()
        logger.info(f"Łączna liczba cech po przetworzeniu: {len(self.feature_names)}")

        return df_encoded

    def prepare_train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """
        Przygotowuje podział na zbiór treningowy i testowy

        Args:
            test_size: Proporcja zbioru testowego (0.2 = 20%)
            random_state: Seed dla powtarzalności wyników
        """
        if self.df is None:
            self.load_data()

        # Przygotuj cechy
        X = self.select_features()

        # Zmienna docelowa (target)
        y = self.df['calories']

        # Podział na train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        logger.info(f"Podział danych:")
        logger.info(f"  Zbiór treningowy: {len(self.X_train)} próbek ({(1-test_size)*100:.0f}%)")
        logger.info(f"  Zbiór testowy: {len(self.X_test)} próbek ({test_size*100:.0f}%)")

    def scale_features(self):
        """
        Normalizuje cechy używając StandardScaler
        """
        logger.info("Normalizacja cech...")

        # Dopasuj scaler do danych treningowych
        self.scaler.fit(self.X_train)

        # Transformuj oba zbiory
        self.X_train_scaled = pd.DataFrame(
            self.scaler.transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )

        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )

        logger.info("✓ Normalizacja zakończona")

    def save_datasets(self):
        """
        Zapisuje przygotowane zbiory danych do plików
        """
        logger.info("Zapisuję przetworzone zbiory danych...")

        # Zapisz dane nieskalowane
        self.X_train.to_csv(self.output_dir / "X_train.csv", index=False)
        self.X_test.to_csv(self.output_dir / "X_test.csv", index=False)
        self.y_train.to_csv(self.output_dir / "y_train.csv", index=False, header=['calories'])
        self.y_test.to_csv(self.output_dir / "y_test.csv", index=False, header=['calories'])

        # Zapisz dane skalowane
        self.X_train_scaled.to_csv(self.output_dir / "X_train_scaled.csv", index=False)
        self.X_test_scaled.to_csv(self.output_dir / "X_test_scaled.csv", index=False)

        # Zapisz scaler
        joblib.dump(self.scaler, self.output_dir / "scaler.pkl")

        # Zapisz listę cech
        with open(self.output_dir / "feature_names.txt", 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")

        logger.info(f"✓ Zapisano wszystkie pliki do: {self.output_dir}")

    def get_data_statistics(self):
        """
        Wyświetla statystyki przygotowanych danych
        """
        logger.info("\n" + "="*60)
        logger.info("STATYSTYKI PRZYGOTOWANYCH DANYCH ML")
        logger.info("="*60)

        logger.info(f"\nCechy (features): {len(self.feature_names)}")
        logger.info("\nLista cech:")
        for i, feature in enumerate(self.feature_names, 1):
            logger.info(f"  {i}. {feature}")

        logger.info(f"\nZmienna docelowa (target): calories")
        logger.info(f"  Min: {self.y_train.min():.2f} kcal")
        logger.info(f"  Max: {self.y_train.max():.2f} kcal")
        logger.info(f"  Średnia: {self.y_train.mean():.2f} kcal")
        logger.info(f"  Mediana: {self.y_train.median():.2f} kcal")
        logger.info(f"  Odchylenie std: {self.y_train.std():.2f} kcal")

        logger.info(f"\nRozmiary zbiorów:")
        logger.info(f"  X_train: {self.X_train.shape}")
        logger.info(f"  X_test: {self.X_test.shape}")
        logger.info(f"  y_train: {self.y_train.shape}")
        logger.info(f"  y_test: {self.y_test.shape}")

        logger.info("="*60 + "\n")

    def create_data_info_file(self):
        """
        Tworzy plik z informacjami o danych
        """
        info_file = self.output_dir / "data_info.txt"

        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("INFORMACJE O DANYCH DO UCZENIA MASZYNOWEGO\n")
            f.write("="*60 + "\n\n")

            f.write("1. STRUKTURA DANYCH\n")
            f.write("-"*60 + "\n")
            f.write(f"Liczba cech (features): {len(self.feature_names)}\n")
            f.write(f"Zmienna docelowa (target): calories [kcal]\n\n")

            f.write("2. LISTA CECH\n")
            f.write("-"*60 + "\n")
            for i, feature in enumerate(self.feature_names, 1):
                f.write(f"{i:2d}. {feature}\n")
            f.write("\n")

            f.write("3. PODZIAŁ DANYCH\n")
            f.write("-"*60 + "\n")
            f.write(f"Zbiór treningowy: {len(self.X_train)} próbek (80%)\n")
            f.write(f"Zbiór testowy: {len(self.X_test)} próbek (20%)\n\n")

            f.write("4. STATYSTYKI ZMIENNEJ DOCELOWEJ (zbiór treningowy)\n")
            f.write("-"*60 + "\n")
            f.write(f"Min:             {self.y_train.min():.2f} kcal\n")
            f.write(f"Max:             {self.y_train.max():.2f} kcal\n")
            f.write(f"Średnia:         {self.y_train.mean():.2f} kcal\n")
            f.write(f"Mediana:         {self.y_train.median():.2f} kcal\n")
            f.write(f"Odchylenie std:  {self.y_train.std():.2f} kcal\n\n")

            f.write("5. PLIKI WYJŚCIOWE\n")
            f.write("-"*60 + "\n")
            f.write("X_train.csv         - Cechy treningowe (nieskalowane)\n")
            f.write("X_test.csv          - Cechy testowe (nieskalowane)\n")
            f.write("X_train_scaled.csv  - Cechy treningowe (skalowane)\n")
            f.write("X_test_scaled.csv   - Cechy testowe (skalowane)\n")
            f.write("y_train.csv         - Etykiety treningowe (kalorie)\n")
            f.write("y_test.csv          - Etykiety testowe (kalorie)\n")
            f.write("scaler.pkl          - Obiekt StandardScaler do normalizacji\n")
            f.write("feature_names.txt   - Lista nazw cech\n\n")

            f.write("6. UŻYCIE W MODELACH ML\n")
            f.write("-"*60 + "\n")
            f.write("Do modeli liniowych, SVM, sieci neuronowych:\n")
            f.write("  - Użyj X_train_scaled.csv i X_test_scaled.csv\n\n")
            f.write("Do modeli drzewiastych (Random Forest, XGBoost, LightGBM):\n")
            f.write("  - Użyj X_train.csv i X_test.csv\n\n")

        logger.info(f"✓ Zapisano plik informacyjny: {info_file}")

    def prepare_complete_dataset(self, test_size: float = 0.2, random_state: int = 42):
        """
        Kompletne przygotowanie danych do ML

        Args:
            test_size: Proporcja zbioru testowego
            random_state: Seed dla powtarzalności
        """
        logger.info("\n" + "="*60)
        logger.info("PRZYGOTOWANIE DANYCH DO UCZENIA MASZYNOWEGO")
        logger.info("="*60 + "\n")

        # Wczytaj dane
        self.load_data()

        # Przygotuj podział train/test
        self.prepare_train_test_split(test_size=test_size, random_state=random_state)

        # Skaluj cechy
        self.scale_features()

        # Zapisz zbiory
        self.save_datasets()

        # Wyświetl statystyki
        self.get_data_statistics()

        # Utwórz plik info
        self.create_data_info_file()

        logger.info("="*60)
        logger.info("✓ PRZYGOTOWANIE DANYCH ZAKOŃCZONE!")
        logger.info(f"✓ Pliki zapisane w: {self.output_dir}")
        logger.info("="*60 + "\n")


def main():
    """Główna funkcja przygotowania danych ML"""
    try:
        preparator = MLDataPreparator()

        # Przygotuj kompletny zestaw danych
        preparator.prepare_complete_dataset(
            test_size=0.2,  # 80% train, 20% test
            random_state=42  # Seed dla powtarzalności wyników
        )

    except FileNotFoundError:
        logger.error("Nie znaleziono pliku z przetworzonymi danymi. Uruchom najpierw data_processor.py")
    except Exception as e:
        logger.error(f"Wystąpił błąd: {e}")


if __name__ == "__main__":
    main()
