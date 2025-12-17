"""
Moduł do pobierania danych z Strava API
"""

import requests
import json
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StravaClient:
    """Klient do komunikacji z Strava API"""

    BASE_URL = "https://www.strava.com/api/v3"

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicjalizacja klienta Strava

        Args:
            config_path: Ścieżka do pliku konfiguracyjnego
        """
        self.config = self._load_config(config_path)
        self.access_token = self.config['strava']['access_token']
        self.headers = {'Authorization': f'Bearer {self.access_token}'}

    def _load_config(self, config_path: str) -> Dict:
        """Wczytuje konfigurację z pliku YAML"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_athlete_info(self) -> Dict:
        """
        Pobiera informacje o zalogowanym sportowcu

        Returns:
            Słownik z danymi sportowca
        """
        url = f"{self.BASE_URL}/athlete"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            logger.info("Pomyślnie pobrano informacje o sportowcu")
            return response.json()
        else:
            logger.error(f"Błąd pobierania danych sportowca: {response.status_code}")
            raise Exception(f"Błąd API: {response.status_code}")

    def get_activities(self, page: int = 1, per_page: int = 30) -> List[Dict]:
        """
        Pobiera listę aktywności

        Args:
            page: Numer strony
            per_page: Liczba aktywności na stronę (max 200)

        Returns:
            Lista aktywności
        """
        url = f"{self.BASE_URL}/athlete/activities"
        params = {
            'page': page,
            'per_page': per_page
        }

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            activities = response.json()
            logger.info(f"Pobrano {len(activities)} aktywności ze strony {page}")
            return activities
        else:
            logger.error(f"Błąd pobierania aktywności: {response.status_code}")
            raise Exception(f"Błąd API: {response.status_code}")

    def get_all_activities(self, max_activities: Optional[int] = None) -> List[Dict]:
        """
        Pobiera wszystkie aktywności z paginacją

        Args:
            max_activities: Maksymalna liczba aktywności do pobrania (None = wszystkie)

        Returns:
            Lista wszystkich aktywności
        """
        all_activities = []
        page = 1
        per_page = 200  # Maksymalna wartość

        while True:
            activities = self.get_activities(page=page, per_page=per_page)

            if not activities:
                break

            all_activities.extend(activities)
            logger.info(f"Łącznie pobrano: {len(all_activities)} aktywności")

            if max_activities and len(all_activities) >= max_activities:
                all_activities = all_activities[:max_activities]
                break

            if len(activities) < per_page:
                break

            page += 1
            time.sleep(0.5)  # Unikanie rate limit

        logger.info(f"Zakończono pobieranie. Łącznie: {len(all_activities)} aktywności")
        return all_activities

    def get_activity_details(self, activity_id: int) -> Dict:
        """
        Pobiera szczegółowe dane pojedynczej aktywności

        Args:
            activity_id: ID aktywności

        Returns:
            Szczegółowe dane aktywności
        """
        url = f"{self.BASE_URL}/activities/{activity_id}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Błąd pobierania szczegółów aktywności {activity_id}: {response.status_code}")
            raise Exception(f"Błąd API: {response.status_code}")

    def get_activity_streams(self, activity_id: int,
                            keys: List[str] = None) -> Dict:
        """
        Pobiera strumienie danych dla aktywności (GPS, tętno, moc, etc.)

        Args:
            activity_id: ID aktywności
            keys: Lista kluczy do pobrania (np. ['time', 'latlng', 'distance', 'altitude'])

        Returns:
            Słownik ze strumieniami danych
        """
        if keys is None:
            keys = ['time', 'latlng', 'distance', 'altitude', 'velocity_smooth',
                   'heartrate', 'cadence', 'watts', 'temp', 'moving', 'grade_smooth']

        url = f"{self.BASE_URL}/activities/{activity_id}/streams"
        params = {
            'keys': ','.join(keys),
            'key_by_type': True
        }

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Brak danych streams dla aktywności {activity_id}")
            return {}

    def filter_cycling_activities(self, activities: List[Dict]) -> List[Dict]:
        """
        Filtruje tylko aktywności kolarskie (Ride)

        Args:
            activities: Lista wszystkich aktywności

        Returns:
            Lista aktywności kolarskich
        """
        cycling = [a for a in activities if a.get('type') == 'Ride']
        logger.info(f"Znaleziono {len(cycling)} przejazdów rowerowych")
        return cycling

    def save_activities_to_file(self, activities: List[Dict],
                                filename: str = "data/raw/strava_activities.json"):
        """
        Zapisuje aktywności do pliku JSON

        Args:
            activities: Lista aktywności
            filename: Nazwa pliku
        """
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(activities, f, indent=2, ensure_ascii=False)

        logger.info(f"Zapisano {len(activities)} aktywności do {filename}")

    def download_all_cycling_data(self, include_streams: bool = True,
                                  max_activities: Optional[int] = None):
        """
        Kompletne pobieranie danych kolarskich ze Stravy

        Args:
            include_streams: Czy pobierać szczegółowe strumienie danych
            max_activities: Maksymalna liczba aktywności
        """
        logger.info("Rozpoczynam pobieranie danych ze Stravy...")

        # Pobierz informacje o użytkowniku
        athlete = self.get_athlete_info()
        self.save_activities_to_file([athlete], "data/raw/athlete_info.json")

        # Pobierz wszystkie aktywności
        all_activities = self.get_all_activities(max_activities=max_activities)

        # Filtruj aktywności kolarskie
        cycling_activities = self.filter_cycling_activities(all_activities)

        # Zapisz podstawowe dane
        self.save_activities_to_file(cycling_activities, "data/raw/strava_cycling_activities.json")

        # Pobierz szczegółowe dane jeśli wymagane
        if include_streams:
            logger.info("Pobieram szczegółowe dane streams dla każdej aktywności...")
            detailed_activities = []

            for i, activity in enumerate(cycling_activities, 1):
                activity_id = activity['id']
                logger.info(f"Pobieram szczegóły {i}/{len(cycling_activities)}: {activity.get('name', 'Brak nazwy')}")

                try:
                    # Pobierz szczegóły aktywności
                    details = self.get_activity_details(activity_id)

                    # Pobierz streams
                    streams = self.get_activity_streams(activity_id)

                    # Połącz dane
                    details['streams'] = streams
                    detailed_activities.append(details)

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"Błąd przy pobieraniu aktywności {activity_id}: {e}")
                    continue

            # Zapisz szczegółowe dane
            self.save_activities_to_file(detailed_activities,
                                        "data/raw/strava_detailed_activities.json")

        logger.info("✓ Pobieranie danych zakończone!")
        logger.info(f"Znaleziono {len(cycling_activities)} przejazdów rowerowych")


def main():
    """Główna funkcja do uruchomienia pobierania danych"""
    try:
        client = StravaClient()

        # Pobierz wszystkie dane kolarskie ze streams
        client.download_all_cycling_data(
            include_streams=True,
            max_activities=None  # None = wszystkie
        )

    except FileNotFoundError:
        logger.error("Nie znaleziono pliku config/config.yaml. Upewnij się, że plik istnieje i zawiera dane dostępowe Strava.")
    except Exception as e:
        logger.error(f"Wystąpił błąd: {e}")


if __name__ == "__main__":
    main()
