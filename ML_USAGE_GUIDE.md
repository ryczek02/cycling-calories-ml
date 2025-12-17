# 🤖 Przewodnik Użycia Modeli ML

## Kompletny przewodnik predykcji spalania kalorii

---

## 📋 Spis treści

1. [Pełny przepływ pracy](#pełny-przepływ-pracy)
2. [Trenowanie modeli](#trenowanie-modeli)
3. [Predykcja z GPX](#predykcja-z-gpx)
4. [Interpretacja wyników](#interpretacja-wyników)
5. [Przykłady użycia](#przykłady-użycia)

---

## 🔄 Pełny przepływ pracy

### Krok po kroku:

```bash
# 1. Przygotuj środowisko
pip install -r requirements.txt

# 2. Skonfiguruj Strava API (patrz README.md)
cp config/config.example.yaml config/config.yaml
# Edytuj config.yaml i dodaj swoje dane

# 3. Pobierz dane ze Stravy
python main.py --step 1

# 4. Przetwórz dane
python main.py --step 2

# 5. Stwórz wizualizacje danych
python main.py --step 3

# 6. Przygotuj dane do ML
python main.py --step 4

# 7. Wytrenuj modele ML
python main.py --step 5

# 8. Przewiduj kalorie z GPX
python -m src.predict_from_gpx twoja_trasa.gpx --weight 75
```

### Lub wszystko na raz:

```bash
python main.py --all
```

---

## 🏋️ Trenowanie modeli

### Automatyczne trenowanie

```bash
python main.py --step 5
```

### Ręczne trenowanie

```bash
python -m src.train_models
```

### Co się dzieje podczas trenowania?

1. **Wczytywanie danych** z `data/ml_ready/`
2. **Trenowanie 7 modeli**:
   - Linear Regression (baseline)
   - Ridge Regression
   - Lasso Regression
   - Random Forest (zazwyczaj najlepszy)
   - Gradient Boosting
   - XGBoost (jeśli zainstalowany)
   - LightGBM (jeśli zainstalowany)

3. **Ewaluacja** każdego modelu:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R² (R-squared)
   - MAPE (Mean Absolute Percentage Error)
   - 5-fold Cross-Validation

4. **Generowanie wizualizacji**:
   - `01_predictions_comparison.png` - porównanie predykcji wszystkich modeli
   - `02_residuals_plot.png` - analiza błędów
   - `03_feature_importance.png` - istotność cech (dla modeli drzewiastych)
   - `04_learning_curves.png` - krzywe uczenia
   - `05_linear_regression_analysis.png` - szczegółowa analiza regresji liniowej
   - `06_metrics_comparison.png` - porównanie metryk

5. **Zapisywanie modeli** w `data/ml_models/`

### Wyniki trenowania

Po zakończeniu otrzymasz raport w konsoli:

```
═══════════════════════════════════════════════════════════════════════════════════════
PORÓWNANIE MODELI
═══════════════════════════════════════════════════════════════════════════════════════
Model                       MAE       RMSE       MAPE         R²       CV MAE
-----------------------------------------------------------------------------------
Linear Regression         45.23      58.12      5.2%     0.9523      46.1±3.2
Ridge Regression          45.18      58.05      5.1%     0.9524      46.0±3.1
Lasso Regression          46.89      59.23      5.4%     0.9512      47.2±3.3
Random Forest             38.45      49.67      4.3%     0.9689      39.2±2.8
Gradient Boosting         39.12      50.34      4.4%     0.9678      40.1±2.9
XGBoost                   37.89      48.92      4.2%     0.9701      38.5±2.7
LightGBM                  38.12      49.23      4.3%     0.9695      39.0±2.8
═══════════════════════════════════════════════════════════════════════════════════════

🏆 NAJLEPSZY MODEL (według MAE): XGBoost
   MAE = 37.89 kcal
   R² = 0.9701
```

---

## 🚴 Predykcja z GPX

### Podstawowe użycie

```bash
# Predykcja z domyślną wagą (75 kg)
python -m src.predict_from_gpx moja_trasa.gpx

# Podaj swoją wagę
python -m src.predict_from_gpx moja_trasa.gpx --weight 80

# Użyj konkretnego modelu
python -m src.predict_from_gpx moja_trasa.gpx --weight 75 --model data/ml_models/xgboost.pkl
```

### Test z przykładową trasą

```bash
# Używa dołączonej przykładowej trasy
python -m src.predict_from_gpx example_route.gpx --weight 75
```

### Co otrzymasz?

#### 1. Wizualizacja trasy (`data/predictions/prediction_<nazwa>.png`)

Wykres 2x2 zawierający:

- **Mapa trasy z prędkościami** - każdy punkt GPS pokolorowany według prędkości (czerwony = wolno, zielony = szybko)
- **Profil wysokościowy** - zmiana wysokości w funkcji dystansu
- **Wykres prędkości** - prędkość w czasie z zaznaczoną średnią
- **Podsumowanie** - kluczowe statystyki i **przewidywane spalenie kalorii**

#### 2. Raport tekstowy (`data/predictions/report_<nazwa>.txt`)

```
═══════════════════════════════════════════════════════════════════
RAPORT PREDYKCJI SPALONYCH KALORII
═══════════════════════════════════════════════════════════════════

Plik GPX: moja_trasa.gpx
Model: data/ml_models/random_forest.pkl
Waga sportowca: 75 kg

CHARAKTERYSTYKA TRASY
----------------------------------------------------------------------
Dystans całkowity:        45.20 km
Przewyższenie (↑):        680 m
Zjazd (↓):                620 m
Czas trwania:             110 min (1.8 h)

PRĘDKOŚĆ
----------------------------------------------------------------------
Średnia prędkość:         24.5 km/h
Maksymalna prędkość:      48.2 km/h

NACHYLENIE
----------------------------------------------------------------------
Średnie nachylenie:       1.2%
Maksymalne nachylenie:    8.5%
Minimalne nachylenie:     -7.2%
Przewyższenie na km:      15.0 m/km

═══════════════════════════════════════════════════════════════════
WYNIK PREDYKCJI
═══════════════════════════════════════════════════════════════════
🔥 Przewidywane spalenie: 1250 kcal
═══════════════════════════════════════════════════════════════════

DOKŁADNOŚĆ MODELU
----------------------------------------------------------------------
MAE (błąd średni):        38.45 kcal
RMSE:                     49.67 kcal
R² (dopasowanie):         0.9689
MAPE (błąd %):            4.3%
```

---

## 📊 Interpretacja wyników

### Metryki modelu

#### MAE (Mean Absolute Error)
- **Co to jest**: Średni bezwzględny błąd predykcji
- **Interpretacja**: MAE = 40 kcal oznacza, że średnio model myli się o 40 kcal
- **Im niższy, tym lepiej**

#### RMSE (Root Mean Squared Error)
- **Co to jest**: Pierwiastek ze średniego kwadratu błędów
- **Interpretacja**: Bardziej karze większe błędy niż MAE
- **Im niższy, tym lepiej**

#### R² (R-squared)
- **Co to jest**: Współczynnik determinacji (0-1)
- **Interpretacja**:
  - R² = 0.95 → model wyjaśnia 95% wariancji danych
  - R² = 1.00 → idealne dopasowanie
  - R² = 0.00 → model nie lepszy niż średnia
- **Im bliżej 1, tym lepiej**

#### MAPE (Mean Absolute Percentage Error)
- **Co to jest**: Średni procentowy błąd bezwzględny
- **Interpretacja**: MAPE = 5% oznacza średni błąd 5% wartości rzeczywistej
- **Im niższy, tym lepiej**

### Feature Importance

Najważniejsze cechy dla predykcji (typowo):

1. **distance_km** (30-40%) - dystans ma największy wpływ
2. **moving_time_min** (20-30%) - czas trwania
3. **total_elevation_gain** (15-25%) - przewyższenie
4. **average_speed_kmh** (10-15%) - prędkość średnia
5. **elevation_per_km** (5-10%) - nachylenie na km

---

## 💡 Przykłady użycia

### Scenariusz 1: Planowanie trasy

```bash
# 1. Stwórz trasę w Komoot/Strava Route Builder
# 2. Eksportuj jako GPX
# 3. Przewiduj spalenie:
python -m src.predict_from_gpx planowana_trasa.gpx --weight 80

# Sprawdź wizualizację i raport w data/predictions/
```

### Scenariusz 2: Analiza przeszłych treningów

```bash
# 1. Pobierz GPX ze Stravy (Aktywność → Export GPX)
# 2. Porównaj rzeczywiste spalenie z predykcją:
python -m src.predict_from_gpx wczorajszy_trening.gpx --weight 75

# 3. Sprawdź jak dobrze model przewidział
```

### Scenariusz 3: Praca dyplomowa - wizualizacje

```bash
# 1. Wytrenuj wszystkie modele
python main.py --step 5

# 2. Sprawdź wykresy w data/ml_visualizations/
# 3. Użyj ich w pracy dyplomowej:
#    - 01_predictions_comparison.png → rozdział "Porównanie modeli"
#    - 03_feature_importance.png → rozdział "Analiza cech"
#    - 04_learning_curves.png → rozdział "Ewaluacja modeli"
#    - 05_linear_regression_analysis.png → rozdział "Regresja liniowa"
```

### Scenariusz 4: Optymalizacja diety

```bash
# 1. Przewiduj spalenie dla planowanej trasy
python -m src.predict_from_gpx trasa_100km.gpx --weight 70

# 2. Zaplanuj odpowiednie odżywianie:
#    - Przewidywane spalenie: 2500 kcal
#    - Zalecany przyrost kalorii: 2500 + 500 = 3000 kcal dziennie
```

---

## 🎯 Wskazówki

### Jak poprawić dokładność modelu?

1. **Więcej danych** - zbierz minimum 50-100 treningów
2. **Różnorodne trasy** - równinne, górskie, miejskie
3. **Dokładne dane** - upewnij się że Strava ma poprawną wagę i wiek
4. **Feature engineering** - dodaj nowe cechy (np. temperatura, wiatr)

### Który model wybrać?

- **Random Forest** - najlepszy stosunek dokładność/szybkość, stabilny
- **XGBoost** - najdokładniejszy, ale wolniejszy
- **LightGBM** - bardzo szybki, dobra dokładność dla dużych zbiorów
- **Linear Regression** - szybki, prosty, dobry baseline

### Skąd wziąć pliki GPX?

- **Strava**: Aktywność → ⋮ → "Export GPX"
- **Garmin Connect**: Aktywność → ⚙️ → "Export to GPX"
- **Komoot**: Trasa → "Download" → GPX
- **RideWithGPS**: Route → "Export" → GPX
- **Strava Route Builder**: Utwórz trasę → "Export GPX"

---

## ❓ FAQ

**Q: Model przewiduje zbyt wysokie/niskie kalorie?**
A: Sprawdź czy podałeś poprawną wagę (--weight). Model jest wytrenowany na danych ze Stravy które mogą mieć własne przeszacowania.

**Q: Czy mogę użyć pliku GPX bez danych czasowych?**
A: Tak, model estymuje czas na podstawie dystansu i przewyższenia. Podaj prędkość planowaną.

**Q: Jak często powinienem retrenować model?**
A: Co 20-30 nowych treningów, aby model dostosował się do Twojego postępu kondycyjnego.

**Q: Czy model uwzględnia wiatr/pogodę?**
A: Jeśli dane ze Stravy zawierają temperaturę - tak. Wiatr niestety nie jest standardowo dostępny w GPX.

---

## 📚 Dodatkowe zasoby

- **README.md** - pełna dokumentacja projektu
- **QUICK_START.md** - szybki start
- **INSTRUKCJA.txt** - instrukcja tekstowa
- **data/ml_visualizations/** - wykresy do pracy dyplomowej

---

**Powodzenia z analizą i pracą dyplomową!** 🚴‍♂️📊🎓
