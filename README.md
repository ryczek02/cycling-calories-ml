# Cycling Calories ML

**Predykcja spalania kalorii przez rowerzystów z wykorzystaniem uczenia maszynowego**

Praca inżynierska - Łukasz Ryczko (14621)
Wyższa Szkoła Ekonomii i Informatyki w Krakowie

---

## 📋 Spis treści

- [Opis projektu](#-opis-projektu)
- [Funkcjonalności](#-funkcjonalności)
- [Struktura projektu](#-struktura-projektu)
- [Wymagania](#-wymagania)
- [Instalacja](#-instalacja)
- [Konfiguracja Strava API](#-konfiguracja-strava-api)
- [Użycie](#-użycie)
- [Dane wyjściowe](#-dane-wyjściowe)
- [Rozwiązywanie problemów](#-rozwiązywanie-problemów)

---

## 📖 Opis projektu

System do pobierania, przetwarzania i analizy danych treningowych kolarskich ze Stravy, z zaawansowanym przygotowaniem danych do modeli uczenia maszynowego predykcji spalonych kalorii.

**Cechy systemu:**
- Automatyczne pobieranie danych ze Strava API
- Przetwarzanie i czyszczenie danych treningowych
- Zaawansowana inżynieria cech (feature engineering)
- Wizualizacje (heatmapy, wykresy korelacji, rozkłady)
- Przygotowanie zbiorów train/test do ML
- Modułowa architektura - łatwe rozszerzanie

---

## ✨ Funkcjonalności

### 1. Pobieranie danych ze Strava
- Automatyczne pobieranie wszystkich przejazdów rowerowych
- Pobieranie szczegółowych danych streams (GPS, tętno, moc, temperatura)
- Zapisywanie w formacie JSON

### 2. Przetwarzanie danych
- Ekstrakcja podstawowych i pochodnych cech
- Obliczanie metryk: prędkość, nachylenie, intensywność spalania
- Czyszczenie outliers i braków danych
- Analiza zmiennych czasowych

### 3. Wizualizacje
- **Heatmapy** - zależności dystans/nachylenie/kalorie, prędkość/czas/kalorie
- **Macierz korelacji** - wszystkie zmienne
- **Rozkłady** - dystans, czas, kalorie, prędkość, nachylenie
- **Scatter plots** - relacje między zmiennymi
- **Wykresy czasowe** - progres treningów

### 4. Przygotowanie danych ML
- Podział train/test (80/20)
- Normalizacja cech (StandardScaler)
- Zapisywanie w CSV gotowych do użycia
- Dokumentacja cech i statystyk

---

## 📁 Struktura projektu

```
cycling_calories_ml/
├── README.md                    # Ten plik
├── requirements.txt             # Wymagane biblioteki Python
├── main.py                      # Główny skrypt uruchomieniowy
├── .gitignore                   # Pliki ignorowane przez git
│
├── config/
│   ├── config.example.yaml      # Przykładowa konfiguracja
│   └── config.yaml              # Twoja konfiguracja (do uzupełnienia)
│
├── src/                         # Kod źródłowy
│   ├── __init__.py
│   ├── strava_client.py         # Pobieranie danych ze Strava API
│   ├── data_processor.py        # Przetwarzanie danych
│   ├── visualization.py         # Tworzenie wizualizacji
│   └── ml_preparation.py        # Przygotowanie danych do ML
│
├── data/                        # Dane
│   ├── raw/                     # Surowe dane ze Strava (JSON)
│   ├── processed/               # Przetworzone dane (CSV)
│   ├── ml_ready/                # Dane gotowe do ML (train/test)
│   └── visualizations/          # Wykresy i heatmapy (PNG)
│
├── models/                      # Katalog na przyszłe modele ML
│
└── notebooks/                   # Jupyter notebooks (opcjonalnie)
```

---

## 🔧 Wymagania

- **Python 3.8+**
- **Konto Strava** z aktywnością kolarską
- **Strava API credentials** (Client ID, Client Secret, Access Token)

---

## 📦 Instalacja

### 1. Sklonuj repozytorium lub rozpakuj projekt

```bash
cd cycling_calories_ml
```

### 2. Utwórz środowisko wirtualne (zalecane)

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Zainstaluj wymagane biblioteki

```bash
pip install -r requirements.txt
```

---

## 🔑 Konfiguracja Strava API

### Krok 1: Utwórz aplikację Strava

1. Zaloguj się na **[Strava](https://www.strava.com)**
2. Przejdź do **[Strava API Settings](https://www.strava.com/settings/api)**
3. Kliknij **"Create an App"** (lub "My API Application")

**Wypełnij formularz:**
- **Application Name**: `Cycling Calories ML`
- **Category**: `Data Importer` lub `Research`
- **Club**: (pozostaw puste)
- **Website**: `http://localhost` (lub dowolny URL)
- **Authorization Callback Domain**: `localhost`
- **Description**: `System predykcji spalanych kalorii`

4. Kliknij **"Create"**

### Krok 2: Skopiuj Client ID i Client Secret

Po utworzeniu aplikacji zobaczysz:
- **Client ID** (np. `12345`)
- **Client Secret** (np. `abcdef1234567890abcdef1234567890abcdef12`)

**Zachowaj te dane!**

### Krok 3: Wygeneruj Access Token

#### Opcja A: Używając przeglądarki (prostsze)

1. W przeglądarce wklej poniższy URL (zamień `YOUR_CLIENT_ID`):

```
https://www.strava.com/oauth/authorize?client_id=YOUR_CLIENT_ID&response_type=code&redirect_uri=http://localhost&approval_prompt=force&scope=activity:read_all
```

2. Zaloguj się i kliknij **"Authorize"**

3. Zostaniesz przekierowany na `http://localhost/?code=XXXXXX`

4. Skopiuj wartość `code` z URL (to jest Twój **authorization code**)

5. Użyj tego kodu aby uzyskać **Access Token** i **Refresh Token**:

```bash
curl -X POST https://www.strava.com/oauth/token \
  -d client_id=YOUR_CLIENT_ID \
  -d client_secret=YOUR_CLIENT_SECRET \
  -d code=YOUR_AUTHORIZATION_CODE \
  -d grant_type=authorization_code
```

**Odpowiedź będzie zawierać:**
```json
{
  "access_token": "your_access_token_here",
  "refresh_token": "your_refresh_token_here",
  "expires_at": 1234567890
}
```

#### Opcja B: Używając Python (bardziej automatyczne)

Stwórz plik `get_token.py`:

```python
import requests

CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
AUTHORIZATION_CODE = "YOUR_AUTHORIZATION_CODE"  # Z kroku 4 powyżej

response = requests.post(
    "https://www.strava.com/oauth/token",
    data={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": AUTHORIZATION_CODE,
        "grant_type": "authorization_code"
    }
)

print(response.json())
```

Uruchom:
```bash
python get_token.py
```

### Krok 4: Skonfiguruj config.yaml

1. Skopiuj przykładowy plik konfiguracyjny:

```bash
cp config/config.example.yaml config/config.yaml
```

2. Edytuj `config/config.yaml` i uzupełnij swoje dane:

```yaml
strava:
  access_token: "TWÓJ_ACCESS_TOKEN"
  client_id: "TWÓJ_CLIENT_ID"
  client_secret: "TWÓJ_CLIENT_SECRET"
  refresh_token: "TWÓJ_REFRESH_TOKEN"
```

**UWAGA:** Plik `config/config.yaml` jest w `.gitignore` - nie zostanie dodany do repo (bezpieczeństwo!)

---

## 🚀 Użycie

### Opcja 1: Uruchom pełny pipeline (zalecane)

```bash
python main.py --all
```

To uruchomi wszystkie 5 kroków:
1. ✅ Pobieranie danych ze Strava
2. ✅ Przetwarzanie danych
3. ✅ Tworzenie wizualizacji
4. ✅ Przygotowanie danych ML
5. ✅ Trenowanie modeli ML

### Opcja 2: Uruchamiaj kroki osobno

```bash
# Krok 1: Pobierz dane ze Strava
python main.py --step 1

# Krok 2: Przetwórz dane
python main.py --step 2

# Krok 3: Stwórz wizualizacje
python main.py --step 3

# Krok 4: Przygotuj dane do ML
python main.py --step 4

# Krok 5: Wytrenuj modele ML
python main.py --step 5
```

### Opcja 3: Uruchamiaj moduły bezpośrednio

```bash
# Pobieranie danych
python -m src.strava_client

# Przetwarzanie
python -m src.data_processor

# Wizualizacje
python -m src.visualization

# Przygotowanie ML
python -m src.ml_preparation
```

---

## 📊 Dane wyjściowe

Po uruchomieniu pełnego pipeline otrzymasz:

### 1. Surowe dane (data/raw/)
- `athlete_info.json` - informacje o Twoim koncie Strava
- `strava_cycling_activities.json` - lista wszystkich przejazdów
- `strava_detailed_activities.json` - szczegółowe dane z streams

### 2. Przetworzone dane (data/processed/)
- `processed_activities.csv` - pełny zbiór danych z wszystkimi cechami

### 3. Wizualizacje (data/visualizations/)
- `heatmap_distance_elevation_calories.png` - dystans vs nachylenie vs kalorie
- `heatmap_speed_time_calories.png` - prędkość vs czas vs kalorie
- `heatmap_distance_speed_calories.png` - dystans vs prędkość vs kalorie
- `correlation_heatmap.png` - macierz korelacji
- `distribution_plots.png` - rozkłady zmiennych
- `scatter_plots.png` - wykresy rozrzutu
- `time_series_plot.png` - progres w czasie

### 4. Dane ML (data/ml_ready/)
- `X_train.csv` - cechy treningowe (nieskalowane)
- `X_test.csv` - cechy testowe (nieskalowane)
- `X_train_scaled.csv` - cechy treningowe (skalowane)
- `X_test_scaled.csv` - cechy testowe (skalowane)
- `y_train.csv` - etykiety treningowe (kalorie)
- `y_test.csv` - etykiety testowe (kalorie)
- `scaler.pkl` - obiekt StandardScaler
- `feature_names.txt` - lista nazw cech
- `data_info.txt` - szczegółowy opis danych

---

## 🧠 Wykorzystanie danych ML

### Dla modeli liniowych, SVM, sieci neuronowych:
Użyj **skalowanych** danych:
```python
import pandas as pd

X_train = pd.read_csv("data/ml_ready/X_train_scaled.csv")
X_test = pd.read_csv("data/ml_ready/X_test_scaled.csv")
y_train = pd.read_csv("data/ml_ready/y_train.csv")
y_test = pd.read_csv("data/ml_ready/y_test.csv")
```

### Dla modeli drzewiastych (Random Forest, XGBoost, LightGBM):
Użyj **nieskalowanych** danych:
```python
import pandas as pd

X_train = pd.read_csv("data/ml_ready/X_train.csv")
X_test = pd.read_csv("data/ml_ready/X_test.csv")
y_train = pd.read_csv("data/ml_ready/y_train.csv")
y_test = pd.read_csv("data/ml_ready/y_test.csv")
```

### Przykład: Trening modelu

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Wczytaj dane
X_train = pd.read_csv("data/ml_ready/X_train.csv")
X_test = pd.read_csv("data/ml_ready/X_test.csv")
y_train = pd.read_csv("data/ml_ready/y_train.csv").values.ravel()
y_test = pd.read_csv("data/ml_ready/y_test.csv").values.ravel()

# Trenuj model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcja
y_pred = model.predict(X_test)

# Ocena
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f} kcal")
print(f"RMSE: {rmse:.2f} kcal")
```

---

## 🤖 Trenowanie Modeli ML i Predykcja

### Krok 5: Trenowanie modeli

Po przygotowaniu danych (krok 4), możesz wytrenować modele uczenia maszynowego:

```bash
# Automatycznie trenuje wszystkie modele
python main.py --step 5

# lub bezpośrednio
python -m src.train_models
```

**Co robi ten krok:**
- Trenuje 6 modeli ML: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM
- Porównuje ich wydajność (MAE, RMSE, R², MAPE)
- Tworzy **wykresy do pracy dyplomowej**:
  - Porównanie predykcji wszystkich modeli
  - Wykresy residuals (błędów)
  - Feature importance (istotność cech)
  - Learning curves (krzywe uczenia)
  - Analiza regresji liniowej (Q-Q plot, rozkład residuals)
  - Porównanie metryk
- Zapisuje wytrenowane modele w `data/ml_models/`

**Pliki wyjściowe:**

`data/ml_models/`
- `random_forest.pkl` - model Random Forest (zazwyczaj najlepszy)
- `xgboost.pkl` - model XGBoost
- `linear_regression.pkl` - regresja liniowa
- inne modele...
- `model_comparison.txt` - tabela porównawcza

`data/ml_visualizations/`
- `01_predictions_comparison.png` - porównanie predykcji
- `02_residuals_plot.png` - wykresy residuals
- `03_feature_importance.png` - istotność cech
- `04_learning_curves.png` - krzywe uczenia
- `05_linear_regression_analysis.png` - analiza regresji
- `06_metrics_comparison.png` - porównanie metryk

---

## 🚴 Predykcja z pliku GPX

### Jak używać wytrenowanego modelu

Po wytrenowaniu modeli możesz przewidywać spalenie kalorii i prędkość z dowolnego pliku GPX:

```bash
# Podstawowe użycie (domyślna waga 75 kg)
python -m src.predict_from_gpx twoja_trasa.gpx

# Podaj swoją wagę
python -m src.predict_from_gpx twoja_trasa.gpx --weight 80

# Użyj innego modelu
python -m src.predict_from_gpx twoja_trasa.gpx --weight 75 --model data/ml_models/xgboost.pkl
```

**Co otrzymasz:**

1. **Wizualizację trasy** (`data/predictions/prediction_<nazwa>.png`):
   - Mapę trasy z kolorami pokazującymi prędkość w różnych punktach
   - Profil wysokościowy
   - Wykres prędkości w funkcji dystansu
   - Podsumowanie ze **przewidywanym spaleniem kalorii**

2. **Raport tekstowy** (`data/predictions/report_<nazwa>.txt`):
   - Charakterystyka trasy (dystans, przewyższenie, czas)
   - Statystyki prędkości i nachylenia
   - **Przewidywane spalenie kalorii**
   - Informacje o dokładności modelu

### Przykład wyjścia:

```
═══════════════════════════════════════════════════════════════════
WYNIKI PREDYKCJI
═══════════════════════════════════════════════════════════════════
Dystans: 45.20 km
Przewyższenie: 680 m
Średnia prędkość: 24.5 km/h
Czas trwania: 110 min

🔥 PRZEWIDYWANE SPALENIE: 1250 kcal
═══════════════════════════════════════════════════════════════════
```

### Skąd wziąć pliki GPX?

- **Strava**: Otwórz aktywność → menu (⋮) → "Export GPX"
- **Garmin Connect**: Aktywność → ⚙️ → "Export to GPX"
- **Komoot**, **RideWithGPS**: każda trasa ma opcję "Download GPX"
- **Planowanie trasy**: Użyj narzędzi online (Komoot, Strava Route Builder) aby stworzyć planowaną trasę

---

## 📊 Interpretacja wyników ML

### Metryki modelu

- **MAE (Mean Absolute Error)**: Średni błąd predykcji w kcal. Im niższy, tym lepiej.
  - Przykład: MAE = 50 kcal oznacza że średnio model myli się o 50 kcal

- **RMSE (Root Mean Squared Error)**: Bardziej karze większe błędy. Im niższy, tym lepiej.

- **R² (R-squared)**: Dopasowanie modelu (0-1). Im bliżej 1, tym lepiej.
  - R² = 0.95 oznacza że model wyjaśnia 95% wariancji danych

- **MAPE (Mean Absolute Percentage Error)**: Błąd procentowy.
  - MAPE = 5% oznacza że średni błąd to 5% wartości rzeczywistej

### Który model wybrać?

System automatycznie wybiera **najlepszy model** (według MAE).

Zazwyczaj:
- **Random Forest** - najlepszy stosunek dokładności do szybkości
- **XGBoost** - często najdokładniejszy, ale wolniejszy
- **LightGBM** - bardzo szybki, dobra dokładność
- **Linear Regression** - baseline, do porównania

### Feature Importance

Wykres pokazuje **które cechy są najważniejsze** dla predykcji:

Typowo najważniejsze cechy:
1. `distance_km` - dystans
2. `moving_time_min` - czas
3. `total_elevation_gain` - przewyższenie
4. `average_speed_kmh` - prędkość średnia
5. `elevation_per_km` - nachylenie na km

---

## 🔍 Rozwiązywanie problemów

### Problem: `FileNotFoundError: config/config.yaml`

**Rozwiązanie:**
```bash
cp config/config.example.yaml config/config.yaml
# Następnie edytuj config/config.yaml i dodaj swoje dane Strava
```

### Problem: `Błąd API: 401 Unauthorized`

**Przyczyna:** Nieprawidłowy lub wygasły Access Token

**Rozwiązanie:**
1. Wygeneruj nowy Access Token (patrz sekcja "Konfiguracja Strava API")
2. Zaktualizuj `config/config.yaml`

### Problem: `Nie znaleziono pliku z surowymi danymi`

**Rozwiązanie:** Uruchom kroki po kolei:
```bash
python main.py --step 1  # Najpierw pobierz dane
python main.py --step 2  # Potem przetwórz
```

### Problem: `ModuleNotFoundError: No module named 'requests'`

**Rozwiązanie:**
```bash
pip install -r requirements.txt
```

### Problem: Mało danych treningowych

**Przyczyna:** Nowe konto Strava lub mało aktywności

**Rozwiązanie:**
- Upewnij się, że masz minimum 20-30 przejazdów rowerowych
- Sprawdź czy Twoje aktywności są publiczne/widoczne przez API
- Użyj zakładki "Upload" na Strava aby zaimportować stare treningi

### Problem: Access Token wygasa po 6 godzinach

**Rozwiązanie:** Implementacja automatycznego odświeżania tokenu (TODO dla przyszłej wersji)

---

## 📝 Licencja

Projekt edukacyjny - Praca inżynierska
Autor: Łukasz Ryczko
WSEI Kraków 2026

---

## 🙏 Podziękowania

- **Strava API** - za dostęp do danych treningowych
- **dr hab. Dariusz Put** - za opiekę naukową

---

## 📧 Kontakt

W razie problemów lub pytań:
- Sprawdź sekcję "Rozwiązywanie problemów" powyżej
- Przejrzyj logi - system wyświetla szczegółowe informacje o błędach
- Upewnij się, że wszystkie kroki zostały wykonane poprawnie

---

**Powodzenia z projektem!** 🚴‍♂️📊🤖
