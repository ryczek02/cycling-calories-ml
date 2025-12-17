# 🚀 QUICK START - Szybki Start

## Uruchomienie w 5 krokach

### 1️⃣ Zainstaluj wymagania
```bash
pip install -r requirements.txt
```

### 2️⃣ Utwórz aplikację Strava
- Przejdź do: https://www.strava.com/settings/api
- Kliknij "Create an App"
- Zapisz **Client ID** i **Client Secret**

### 3️⃣ Wygeneruj Access Token

W przeglądarce wklej (zamień YOUR_CLIENT_ID):
```
https://www.strava.com/oauth/authorize?client_id=YOUR_CLIENT_ID&response_type=code&redirect_uri=http://localhost&approval_prompt=force&scope=activity:read_all
```

Po autoryzacji skopiuj `code` z URL i wykonaj:
```bash
curl -X POST https://www.strava.com/oauth/token \
  -d client_id=YOUR_CLIENT_ID \
  -d client_secret=YOUR_CLIENT_SECRET \
  -d code=YOUR_CODE \
  -d grant_type=authorization_code
```

### 4️⃣ Skonfiguruj config.yaml
```bash
cp config/config.example.yaml config/config.yaml
# Edytuj config/config.yaml i wklej swoje dane
```

### 5️⃣ Uruchom pipeline
```bash
python main.py --all
```

---

## 📁 Gdzie znajdę wyniki?

- **Wizualizacje danych**: `data/visualizations/*.png`
- **Dane ML**: `data/ml_ready/*.csv`
- **Modele ML**: `data/ml_models/*.pkl`
- **Wizualizacje ML**: `data/ml_visualizations/*.png`
- **Predykcje GPX**: `data/predictions/*.png`

---

## 🤖 Trenowanie modeli i predykcja

### Automatyczne trenowanie
Pipeline (krok 5) automatycznie trenuje wszystkie modele:
```bash
python main.py --all  # Pełny pipeline z trenowaniem
```

### Predykcja z pliku GPX
Po wytrenowaniu modeli użyj ich do predykcji:

```bash
# Przewiduj kalorie z trasy GPX
python -m src.predict_from_gpx twoja_trasa.gpx --weight 75

# Test z przykładową trasą
python -m src.predict_from_gpx example_route.gpx --weight 75
```

**Otrzymasz:**
- 🗺️ Mapę trasy z kolorami pokazującymi prędkość
- 📊 Profil wysokościowy
- 🔥 **Przewidywane spalenie kalorii**
- 📄 Szczegółowy raport tekstowy

---

Szczegółowe instrukcje w **README.md**
