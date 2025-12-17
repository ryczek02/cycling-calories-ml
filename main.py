#!/usr/bin/env python3
"""
Cycling Calories ML - Główny skrypt uruchomieniowy
Autor: Łukasz Ryczko

Ten skrypt uruchamia pełny pipeline przetwarzania danych:
1. Pobieranie danych ze Strava
2. Przetwarzanie danych
3. Tworzenie wizualizacji
4. Przygotowanie danych do ML
"""

import argparse
import sys
from pathlib import Path

# Dodaj src do ścieżki
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.strava_client import StravaClient
from src.data_processor import StravaDataProcessor
from src.visualization import CyclingVisualizer
from src.ml_preparation import MLDataPreparator
from src.train_models import ModelTrainer

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """Uruchamia pełny pipeline przetwarzania danych"""
    logger.info("\n" + "="*70)
    logger.info("CYCLING CALORIES ML - PEŁNY PIPELINE")
    logger.info("="*70 + "\n")

    try:
        # Krok 1: Pobieranie danych ze Strava
        logger.info("KROK 1/4: Pobieranie danych ze Strava")
        logger.info("-"*70)
        client = StravaClient()
        client.download_all_cycling_data(include_streams=True, max_activities=None)
        logger.info("✓ Krok 1 zakończony\n")

        # Krok 2: Przetwarzanie danych
        logger.info("KROK 2/4: Przetwarzanie danych")
        logger.info("-"*70)
        processor = StravaDataProcessor()
        processor.load_raw_data()
        df = processor.process_all_activities()
        df = processor.clean_data(df)
        processor.save_processed_data(df)
        processor.get_data_summary(df)
        logger.info("✓ Krok 2 zakończony\n")

        # Krok 3: Tworzenie wizualizacji
        logger.info("KROK 3/4: Tworzenie wizualizacji")
        logger.info("-"*70)
        visualizer = CyclingVisualizer()
        visualizer.create_all_visualizations()
        logger.info("✓ Krok 3 zakończony\n")

        # Krok 4: Przygotowanie danych do ML
        logger.info("KROK 4/5: Przygotowanie danych do ML")
        logger.info("-"*70)
        preparator = MLDataPreparator()
        preparator.prepare_complete_dataset(test_size=0.2, random_state=42)
        logger.info("✓ Krok 4 zakończony\n")

        # Krok 5: Trenowanie modeli ML
        logger.info("KROK 5/5: Trenowanie modeli ML")
        logger.info("-"*70)
        trainer = ModelTrainer()
        trainer.load_data()
        trainer.initialize_models()
        trainer.train_all_models()
        trainer.create_comparison_table()
        trainer.save_models()
        trainer.create_all_visualizations()
        logger.info("✓ Krok 5 zakończony\n")

        logger.info("="*70)
        logger.info("✓✓✓ PEŁNY PIPELINE ZAKOŃCZONY POMYŚLNIE! ✓✓✓")
        logger.info("="*70 + "\n")

        logger.info("Wyniki:")
        logger.info("1. Wizualizacje danych: data/visualizations/")
        logger.info("2. Dane ML: data/ml_ready/")
        logger.info("3. Modele ML: data/ml_models/")
        logger.info("4. Wizualizacje ML: data/ml_visualizations/")
        logger.info("\nAby przewidzieć kalorie z pliku GPX użyj:")
        logger.info("  python -m src.predict_from_gpx <plik.gpx> --weight 75")

    except Exception as e:
        logger.error(f"Błąd w pipeline: {e}")
        sys.exit(1)


def run_step_1():
    """Krok 1: Pobieranie danych ze Strava"""
    logger.info("Uruchamiam: Pobieranie danych ze Strava...")
    client = StravaClient()
    client.download_all_cycling_data(include_streams=True, max_activities=None)
    logger.info("✓ Pobieranie zakończone")


def run_step_2():
    """Krok 2: Przetwarzanie danych"""
    logger.info("Uruchamiam: Przetwarzanie danych...")
    processor = StravaDataProcessor()
    processor.load_raw_data()
    df = processor.process_all_activities()
    df = processor.clean_data(df)
    processor.save_processed_data(df)
    processor.get_data_summary(df)
    logger.info("✓ Przetwarzanie zakończone")


def run_step_3():
    """Krok 3: Tworzenie wizualizacji"""
    logger.info("Uruchamiam: Tworzenie wizualizacji...")
    visualizer = CyclingVisualizer()
    visualizer.create_all_visualizations()
    logger.info("✓ Wizualizacje gotowe")


def run_step_4():
    """Krok 4: Przygotowanie danych do ML"""
    logger.info("Uruchamiam: Przygotowanie danych do ML...")
    preparator = MLDataPreparator()
    preparator.prepare_complete_dataset(test_size=0.2, random_state=42)
    logger.info("✓ Dane ML gotowe")


def run_step_5():
    """Krok 5: Trenowanie modeli ML"""
    logger.info("Uruchamiam: Trenowanie modeli ML...")
    trainer = ModelTrainer()
    trainer.load_data()
    trainer.initialize_models()
    trainer.train_all_models()
    trainer.create_comparison_table()
    trainer.save_models()
    trainer.create_all_visualizations()
    logger.info("✓ Modele wytrenowane i zapisane")


def main():
    parser = argparse.ArgumentParser(
        description="Cycling Calories ML - System predykcji spalania kalorii",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python main.py --all              # Uruchom pełny pipeline
  python main.py --step 1           # Tylko pobieranie danych
  python main.py --step 2           # Tylko przetwarzanie danych
  python main.py --step 3           # Tylko wizualizacje
  python main.py --step 4           # Tylko przygotowanie danych ML
  python main.py --step 5           # Tylko trenowanie modeli ML

Kolejność kroków:
  1. Pobieranie danych ze Strava API
  2. Przetwarzanie i czyszczenie danych
  3. Tworzenie wizualizacji (heatmapy, wykresy)
  4. Przygotowanie danych do uczenia maszynowego
  5. Trenowanie modeli ML i tworzenie wizualizacji
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Uruchom pełny pipeline (wszystkie kroki)'
    )

    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Uruchom konkretny krok (1-5)'
    )

    args = parser.parse_args()

    # Jeśli nie podano argumentów, pokaż pomoc
    if not args.all and args.step is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.all:
            run_full_pipeline()
        elif args.step == 1:
            run_step_1()
        elif args.step == 2:
            run_step_2()
        elif args.step == 3:
            run_step_3()
        elif args.step == 4:
            run_step_4()
        elif args.step == 5:
            run_step_5()

    except FileNotFoundError as e:
        logger.error(f"Błąd: Nie znaleziono pliku: {e}")
        logger.error("Upewnij się, że:")
        logger.error("1. Plik config/config.yaml istnieje i zawiera dane Strava")
        logger.error("2. Uruchomiłeś wcześniejsze kroki w odpowiedniej kolejności")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Wystąpił błąd: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
