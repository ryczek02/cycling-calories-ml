"""
Moduł do wizualizacji danych treningowych
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ustawienia stylu wykresów
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class CyclingVisualizer:
    """Klasa do wizualizacji danych treningowych"""

    def __init__(self, data_file: str = "data/processed/processed_activities.csv"):
        """
        Inicjalizacja wizualizera

        Args:
            data_file: Ścieżka do pliku z przetworzonymi danymi
        """
        self.data_file = data_file
        self.df = None
        self.output_dir = Path("data/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Wczytuje przetworzone dane"""
        self.df = pd.read_csv(self.data_file)
        logger.info(f"Wczytano {len(self.df)} aktywności z {self.data_file}")
        return self.df

    def create_heatmap_distance_elevation_calories(self):
        """
        Tworzy heatmapę przedstawiającą zależność między dystansem,
        nachyleniem i spalonymi kaloriami
        """
        logger.info("Tworzę heatmapę: dystans vs nachylenie vs kalorie...")

        # Przygotuj dane do heatmapy
        # Podziel dystans na bins
        distance_bins = pd.cut(self.df['distance_km'], bins=10)
        elevation_bins = pd.cut(self.df['elevation_per_km'], bins=10)

        # Grupuj i agreguj kalorie
        heatmap_data = self.df.groupby([distance_bins, elevation_bins])['calories'].mean().unstack()

        # Twórz wykres
        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
                   cbar_kws={'label': 'Średnie spalone kalorie [kcal]'})
        plt.title('Heatmapa: Dystans vs Nachylenie vs Spalone Kalorie', fontsize=16, pad=20)
        plt.xlabel('Nachylenie na km [m/km]', fontsize=12)
        plt.ylabel('Dystans [km]', fontsize=12)
        plt.tight_layout()

        output_file = self.output_dir / "heatmap_distance_elevation_calories.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Zapisano: {output_file}")

    def create_heatmap_speed_time_calories(self):
        """
        Tworzy heatmapę przedstawiającą zależność między prędkością,
        czasem trwania i spalonymi kaloriami
        """
        logger.info("Tworzę heatmapę: prędkość vs czas vs kalorie...")

        # Przygotuj dane
        speed_bins = pd.cut(self.df['average_speed_kmh'], bins=10)
        time_bins = pd.cut(self.df['moving_time_min'], bins=10)

        # Grupuj i agreguj kalorie
        heatmap_data = self.df.groupby([time_bins, speed_bins])['calories'].mean().unstack()

        # Twórz wykres
        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='viridis',
                   cbar_kws={'label': 'Średnie spalone kalorie [kcal]'})
        plt.title('Heatmapa: Czas trwania vs Prędkość średnia vs Spalone Kalorie',
                 fontsize=16, pad=20)
        plt.xlabel('Prędkość średnia [km/h]', fontsize=12)
        plt.ylabel('Czas trwania [min]', fontsize=12)
        plt.tight_layout()

        output_file = self.output_dir / "heatmap_speed_time_calories.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Zapisano: {output_file}")

    def create_heatmap_distance_speed_calories(self):
        """
        Tworzy heatmapę: dystans vs prędkość vs kalorie
        """
        logger.info("Tworzę heatmapę: dystans vs prędkość vs kalorie...")

        distance_bins = pd.cut(self.df['distance_km'], bins=8)
        speed_bins = pd.cut(self.df['average_speed_kmh'], bins=8)

        heatmap_data = self.df.groupby([distance_bins, speed_bins])['calories'].mean().unstack()

        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='coolwarm',
                   cbar_kws={'label': 'Średnie spalone kalorie [kcal]'})
        plt.title('Heatmapa: Dystans vs Prędkość vs Spalone Kalorie', fontsize=16, pad=20)
        plt.xlabel('Prędkość średnia [km/h]', fontsize=12)
        plt.ylabel('Dystans [km]', fontsize=12)
        plt.tight_layout()

        output_file = self.output_dir / "heatmap_distance_speed_calories.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Zapisano: {output_file}")

    def create_correlation_heatmap(self):
        """
        Tworzy heatmapę korelacji między różnymi zmiennymi
        """
        logger.info("Tworzę heatmapę korelacji...")

        # Wybierz tylko kolumny numeryczne
        numeric_cols = [
            'distance_km', 'moving_time_min', 'total_elevation_gain',
            'average_speed_kmh', 'max_speed_kmh', 'avg_grade',
            'calorie_burn_rate', 'calories_per_km', 'elevation_per_km',
            'calories', 'average_heartrate', 'max_heartrate'
        ]

        # Filtruj tylko istniejące kolumny
        available_cols = [col for col in numeric_cols if col in self.df.columns]

        # Oblicz macierz korelacji
        corr_matrix = self.df[available_cols].corr()

        # Twórz wykres
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1,
                   cbar_kws={'label': 'Współczynnik korelacji'})
        plt.title('Macierz Korelacji Zmiennych Treningowych', fontsize=16, pad=20)
        plt.tight_layout()

        output_file = self.output_dir / "correlation_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Zapisano: {output_file}")

    def create_distribution_plots(self):
        """
        Tworzy wykresy rozkładu kluczowych zmiennych
        """
        logger.info("Tworzę wykresy rozkładów zmiennych...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Rozkłady Zmiennych Treningowych', fontsize=16)

        # Dystans
        axes[0, 0].hist(self.df['distance_km'], bins=30, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Rozkład dystansu')
        axes[0, 0].set_xlabel('Dystans [km]')
        axes[0, 0].set_ylabel('Liczba treningów')

        # Czas
        axes[0, 1].hist(self.df['moving_time_min'], bins=30, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Rozkład czasu trwania')
        axes[0, 1].set_xlabel('Czas [min]')
        axes[0, 1].set_ylabel('Liczba treningów')

        # Kalorie
        axes[0, 2].hist(self.df['calories'], bins=30, color='coral', edgecolor='black')
        axes[0, 2].set_title('Rozkład spalonych kalorii')
        axes[0, 2].set_xlabel('Kalorie [kcal]')
        axes[0, 2].set_ylabel('Liczba treningów')

        # Prędkość
        axes[1, 0].hist(self.df['average_speed_kmh'], bins=30, color='gold', edgecolor='black')
        axes[1, 0].set_title('Rozkład prędkości średniej')
        axes[1, 0].set_xlabel('Prędkość [km/h]')
        axes[1, 0].set_ylabel('Liczba treningów')

        # Nachylenie
        axes[1, 1].hist(self.df['avg_grade'], bins=30, color='plum', edgecolor='black')
        axes[1, 1].set_title('Rozkład nachylenia średniego')
        axes[1, 1].set_xlabel('Nachylenie [%]')
        axes[1, 1].set_ylabel('Liczba treningów')

        # Przewyższenie
        axes[1, 2].hist(self.df['total_elevation_gain'], bins=30, color='lightcoral', edgecolor='black')
        axes[1, 2].set_title('Rozkład przewyższenia')
        axes[1, 2].set_xlabel('Przewyższenie [m]')
        axes[1, 2].set_ylabel('Liczba treningów')

        plt.tight_layout()

        output_file = self.output_dir / "distribution_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Zapisano: {output_file}")

    def create_scatter_plots(self):
        """
        Tworzy wykresy rozrzutu pokazujące zależności między zmiennymi
        """
        logger.info("Tworzę wykresy rozrzutu...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Zależności między Zmiennymi a Spalonymi Kaloriami', fontsize=16)

        # Dystans vs Kalorie
        axes[0, 0].scatter(self.df['distance_km'], self.df['calories'], alpha=0.5, color='blue')
        axes[0, 0].set_title('Dystans vs Kalorie')
        axes[0, 0].set_xlabel('Dystans [km]')
        axes[0, 0].set_ylabel('Kalorie [kcal]')

        # Czas vs Kalorie
        axes[0, 1].scatter(self.df['moving_time_min'], self.df['calories'], alpha=0.5, color='green')
        axes[0, 1].set_title('Czas trwania vs Kalorie')
        axes[0, 1].set_xlabel('Czas [min]')
        axes[0, 1].set_ylabel('Kalorie [kcal]')

        # Prędkość vs Kalorie
        axes[1, 0].scatter(self.df['average_speed_kmh'], self.df['calories'], alpha=0.5, color='red')
        axes[1, 0].set_title('Prędkość średnia vs Kalorie')
        axes[1, 0].set_xlabel('Prędkość [km/h]')
        axes[1, 0].set_ylabel('Kalorie [kcal]')

        # Przewyższenie vs Kalorie
        axes[1, 1].scatter(self.df['total_elevation_gain'], self.df['calories'], alpha=0.5, color='purple')
        axes[1, 1].set_title('Przewyższenie vs Kalorie')
        axes[1, 1].set_xlabel('Przewyższenie [m]')
        axes[1, 1].set_ylabel('Kalorie [kcal]')

        plt.tight_layout()

        output_file = self.output_dir / "scatter_plots.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Zapisano: {output_file}")

    def create_time_series_plot(self):
        """
        Tworzy wykres czasowy pokazujący aktywność w czasie
        """
        logger.info("Tworzę wykres czasowy...")

        # Konwertuj datę
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Aktywność Treningowa w Czasie', fontsize=16)

        # Kalorie w czasie
        axes[0].plot(self.df['date'], self.df['calories'], marker='o', linestyle='-', alpha=0.7)
        axes[0].set_title('Spalone kalorie w czasie')
        axes[0].set_ylabel('Kalorie [kcal]')
        axes[0].grid(True, alpha=0.3)

        # Dystans w czasie
        axes[1].plot(self.df['date'], self.df['distance_km'], marker='o', linestyle='-',
                    alpha=0.7, color='green')
        axes[1].set_title('Dystans przejazdów w czasie')
        axes[1].set_ylabel('Dystans [km]')
        axes[1].grid(True, alpha=0.3)

        # Prędkość w czasie
        axes[2].plot(self.df['date'], self.df['average_speed_kmh'], marker='o', linestyle='-',
                    alpha=0.7, color='red')
        axes[2].set_title('Średnia prędkość w czasie')
        axes[2].set_xlabel('Data')
        axes[2].set_ylabel('Prędkość [km/h]')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = self.output_dir / "time_series_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Zapisano: {output_file}")

    def create_all_visualizations(self):
        """
        Tworzy wszystkie wizualizacje
        """
        if self.df is None:
            self.load_data()

        logger.info("\n" + "="*50)
        logger.info("TWORZENIE WIZUALIZACJI")
        logger.info("="*50)

        self.create_heatmap_distance_elevation_calories()
        self.create_heatmap_speed_time_calories()
        self.create_heatmap_distance_speed_calories()
        self.create_correlation_heatmap()
        self.create_distribution_plots()
        self.create_scatter_plots()
        self.create_time_series_plot()

        logger.info("="*50)
        logger.info(f"✓ Wszystkie wizualizacje zapisane w: {self.output_dir}")
        logger.info("="*50 + "\n")


def main():
    """Główna funkcja tworzenia wizualizacji"""
    try:
        visualizer = CyclingVisualizer()
        visualizer.create_all_visualizations()

    except FileNotFoundError:
        logger.error("Nie znaleziono pliku z przetworzonymi danymi. Uruchom najpierw data_processor.py")
    except Exception as e:
        logger.error(f"Wystąpił błąd: {e}")


if __name__ == "__main__":
    main()
