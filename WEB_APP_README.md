# Cycling Calories Web App - Instrukcja

## Opis

Aplikacja webowa FastAPI z graficznym interfejsem do predykcji spalonych kalorii na podstawie plików GPX. Aplikacja umożliwia:

- Upload plików GPX przez przeglądarkę
- Wizualizację trasy na interaktywnej mapie Leaflet z kolorami pokazującymi prędkość
- Predykcję spalonych kalorii przy użyciu modelu ML
- Wyświetlanie szczegółowych wykresów i statystyk trasy

## Funkcje

### 1. Upload i analiza GPX
- Wgrywanie plików GPX przez formularz webowy
- Ustawienie wagi sportowca (domyślnie 75 kg)
- Automatyczna analiza trasy

### 2. Wizualizacja mapy (Leaflet)
- Interaktywna mapa z trasą
- Kolorowanie segmentów według prędkości:
  - Czerwony: wolno (0-15 km/h)
  - Żółty: średnio (15-25 km/h)
  - Zielony: szybko (25+ km/h)
- Markery startu (S) i mety (M)
- Popupy z informacjami o dystansie, prędkości i wysokości

### 3. Wykresy i wizualizacje
- **Profil wysokościowy** - wykres wysokości w funkcji dystansu
- **Profil prędkości** - wykres prędkości w funkcji dystansu ze średnią
- **Histogram nachylenia** - rozkład nachylenia trasy

### 4. Statystyki
- Dystans całkowity
- Przewyższenie (↑) i zjazd (↓)
- Czas trwania
- Średnia i maksymalna prędkość
- Średnie nachylenie
- Waga sportowca

### 5. Predykcja kalorii
- Wykorzystanie modelu Random Forest
- Wyświetlenie przewidywanej liczby spalonych kalorii

## Instalacja

1. Zainstaluj wymagane biblioteki:
```bash
pip3 install -r requirements.txt
```

lub tylko niezbędne dla web app:
```bash
pip3 install fastapi "uvicorn[standard]" python-multipart jinja2
```

## Uruchomienie

### Sposób 1: Przez Python
```bash
python3 app.py
```

### Sposób 2: Przez Uvicorn (zalecane)
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Parametry:
- `--reload` - automatyczne przeładowanie przy zmianach w kodzie
- `--host 0.0.0.0` - serwer dostępny z sieci lokalnej
- `--port 8000` - port serwera

## Użycie

1. Uruchom serwer (patrz wyżej)
2. Otwórz przeglądarkę i wejdź na: **http://localhost:8000**
3. Wgraj plik GPX
4. Ustaw swoją wagę
5. Kliknij "Analizuj"
6. Przeglądaj wyniki:
   - Mapa z trasą
   - Przewidywane spalenie kalorii
   - Szczegółowe statystyki
   - Wykresy analizy

## Struktura projektu

```
cycling_calories_ml/
├── app.py                     # Główna aplikacja FastAPI
├── templates/
│   └── index.html            # Frontend HTML z Leaflet i wizualizacjami
├── data/
│   ├── uploads/              # Wgrane pliki GPX (tworzone automatycznie)
│   ├── predictions/          # Wyniki predykcji
│   └── ml_models/            # Modele ML
│       └── random_forest.pkl # Model Random Forest
├── src/
│   ├── predict_from_gpx.py   # Moduł predykcji
│   └── gpx_parser.py         # Parser plików GPX
└── requirements.txt          # Wymagane biblioteki
```

## API Endpointy

### GET /
Strona główna z interfejsem użytkownika

### POST /upload
Upload i analiza pliku GPX

**Parametry:**
- `file` (multipart/form-data): Plik GPX
- `weight` (float): Waga sportowca w kg

**Odpowiedź:** JSON z:
- `predicted_calories`: Przewidywane kalorie
- `route_summary`: Statystyki trasy
- `route_points`: Punkty trasy dla mapy
- `charts`: Wykresy jako base64

### GET /health
Health check endpoint

## Technologie

### Backend
- **FastAPI** - framework webowy
- **Uvicorn** - serwer ASGI
- **scikit-learn** - modele ML
- **pandas, numpy** - przetwarzanie danych
- **matplotlib, seaborn** - generowanie wykresów

### Frontend
- **Leaflet.js** - interaktywne mapy
- **Bootstrap 5** - stylowanie UI
- **Vanilla JavaScript** - logika frontendu

## Uwagi

- Model ML (Random Forest) musi być wytrenowany i zapisany w `data/ml_models/random_forest.pkl`
- Pliki GPX są zapisywane w `data/uploads/` z timestampem
- Wykresy są generowane dynamicznie i zwracane jako obrazy base64
- Aplikacja działa najlepiej z plikami GPX zawierającymi dane o wysokości i czasie

## Rozwiązywanie problemów

### Model nie ładuje się
Upewnij się, że plik `data/ml_models/random_forest.pkl` istnieje. Jeśli nie, wytrenuj model używając:
```bash
python3 example_train_model.py
```

### Port zajęty
Zmień port w `app.py` lub użyj innego portu w uvicorn:
```bash
uvicorn app:app --port 8080
```

### Błąd przy uploadu GPX
Sprawdź czy plik GPX jest poprawny i zawiera dane GPS (współrzędne, wysokość, czas)

## Przykładowe pliki GPX

Możesz użyć przykładowych plików z `data/sample_data/` (jeśli dostępne)
